"""
EDA 90단계 — 파이프라인이 **실제로 본** 텍스트로 오답을 진단한다

89단계는 rapid 와 retry 의 텍스트를 합쳐서 후보를 만들었다. 그래서 "정답이
후보에 있는데 규칙이 잘못 골랐다"고 읽힌 것이 많았는데, 재현해 보니 틀렸다.
    000378  합집합에는 2026-05-29 가 있는데 파이프라인은 2025-05-29 를 냈다
    002816  파이프라인이 낸 2021-01-28 은 합집합 후보에 아예 없다
파이프라인은 계단을 한 칸씩 오르며 **그 칸의 텍스트만** 규칙에 넣는다.
합집합에 답이 있다고 해서 파이프라인이 그걸 본 것은 아니다.

여기서는 `CombinedPipeline._rapid` 와 같은 순서로 칸을 오르며
각 칸의 텍스트·규칙 결과를 남긴다. 그래야 "어느 칸에서 무엇을 보고 무엇을
골랐는지"가 나온다.

가르는 기준 (파이프라인이 채택한 칸의 텍스트 기준)
    정답이 그 칸의 후보에 있었나
      있다 → 규칙이 잘못 골랐다        ← 여기만 규칙 문제
      없다 → 그 칸이 정답을 못 읽었다   ← OCR 문제
    그리고 다른 칸에 정답이 있었는지도 함께 본다(더 좋은 칸이 있었는지).

    python eda/90_stage_trace.py --workers 4
"""
import os
import sys
import json
import argparse
import warnings
from collections import Counter

warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

IMG_DIR = os.path.join(ROOT, "images", "상품사진입니다")
OUT = os.path.join(ROOT, "eda", "out")
RAPID_DIR = os.path.join(ROOT, "weights", "rapidocr")

_S = None


def setup():
    global _S
    if _S is None:
        import cv2
        import combined
        import team2_rules
        rapid, _ = combined.build_rapid_ocr(RAPID_DIR)
        retry, _ = combined.build_rapid_retry(RAPID_DIR)
        _S = (cv2, combined, team2_rules, rapid, retry)
    return _S


def work(f):
    """CombinedPipeline._rapid 와 같은 순서로 칸을 오르며 기록한다."""
    cv2, combined, R, rapid, retry = setup()
    img = combined.imread_any(os.path.join(IMG_DIR, f))
    if img is None:
        return None
    small = combined.downscale_max(img, 1024)

    def texts_of(eng, arr):
        try:
            r = eng(arr)
            return list(r.txts) if getattr(r, "txts", None) else []
        except Exception:
            return []

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    adap = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 31, 11)

    stages = [("rapid", rapid, small), ("adaptive", rapid, adap)]
    if retry is not None:
        stages.append(("retry", retry, small))

    rows, taken = [], None
    for name, eng, arr in stages:
        txt = texts_of(eng, arr)
        full = R.choose_expiration_date(txt) if txt else None
        rows.append({"stage": name, "n": len(txt), "full": full,
                     "frags": [t for t in txt if sum(c.isdigit() for c in t) >= 4][:12]})
        if full is not None and taken is None:
            taken = name          # 파이프라인은 여기서 멈춘다
            break                 # 뒤 칸은 실제로 돌지 않으므로 우리도 멈춘다

    return {"file": f, "taken": taken, "stages": rows}


def s3(v):
    if not v or v == "NONE":
        return ("NONE",) * 3
    q = str(v).split("-")
    return tuple(q) if len(q) == 3 else ("NONE",) * 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    D = json.load(open(os.path.join(OUT, "88_all_pred.json"), encoding="utf-8"))
    lab, pred = D["label"], D["pred"]
    wrong = [f for f in lab if s3(lab[f]) != s3(pred[f])]
    print(f"오답 {len(wrong)}장을 계단째 추적", flush=True)

    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        res = {r["file"]: r for r in ex.map(work, wrong, chunksize=1) if r}

    rows, tally = [], Counter()
    for f in wrong:
        r = res.get(f)
        if not r:
            continue
        truth = lab[f]
        used = next((s for s in r["stages"] if s["stage"] == r["taken"]), None)

        # 채택된 칸이 정답을 만들 수 있었나 — 그 칸의 조각만 다시 규칙에 넣어 본다
        if used is None:
            why = "어느 칸도 날짜를 못 만듦"
        else:
            # 그 칸의 조각 중 정답을 내는 것이 있는지
            import team2_rules as R2
            has = False
            for t in used["frags"]:
                if R2.choose_expiration_date([t]) == truth:
                    has = True
                    break
            why = "규칙이 잘못 고름" if has else "그 칸이 정답을 못 읽음"
        tally[why] += 1
        rows.append({"f": f, "truth": truth, "pred": pred[f],
                     "taken": r["taken"], "why": why, "stages": r["stages"]})

    L = ["# EDA 90 — 파이프라인이 실제로 본 텍스트로 본 오답\n",
         "89단계는 rapid 와 retry 의 텍스트를 **합쳐서** 봤다. 파이프라인은",
         "계단을 한 칸씩 오르며 그 칸의 텍스트만 규칙에 넣으므로, 합집합에",
         "답이 있다고 해서 파이프라인이 본 것은 아니다. 여기서 바로잡는다.",
         "",
         "| 막힌 지점 | 장수 |",
         "| --- | ---: |"]
    for k, v in tally.most_common():
        L.append(f"| {k} | {v} |")
    L.append("")
    L.append(f"**규칙 문제 {tally['규칙이 잘못 고름']}장 / "
             f"OCR 문제 {sum(v for k, v in tally.items() if k != '규칙이 잘못 고름')}장**")
    L.append("")

    for why, _ in tally.most_common():
        sub = [x for x in rows if x["why"] == why]
        L.append(f"## {why} — {len(sub)}장\n")
        for x in sub:
            L.append(f"### `{x['f']}` — 정답 {x['truth']} / 출력 {x['pred']} "
                     f"(채택 칸: {x['taken']})\n")
            for s in x["stages"]:
                L.append(f"- **{s['stage']}** 조각 {s['n']}개 → 규칙 `{s['full']}`")
                if s["frags"]:
                    L.append(f"  - `{' | '.join(s['frags'])[:170]}`")
            L.append("")

    with open(os.path.join(OUT, "90_stage_trace.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    json.dump(rows, open(os.path.join(OUT, "90_stage_trace.json"), "w",
                         encoding="utf-8"), ensure_ascii=False, indent=1)
    print()
    print("\n".join(L[5:13]))
    print("\n→ eda/out/90_stage_trace.md")


if __name__ == "__main__":
    main()
