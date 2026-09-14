"""
EDA 40단계 — 2차 홀드아웃 오답 심층 분해

홀드아웃 200장에서 부분점수 64.5%, 미검출 55장(27.8%), 오탐 0장.
튜닝셋(12.4% 미검출)의 두 배가 넘는다. 무엇을 못 읽는지 유형화한다.

각 오답에 대해 OCR 원문까지 뽑아, 못 읽은 것(검출/인식 실패)과
읽었는데 못 뽑은 것(파싱 실패)을 구분한다. 후자는 규칙으로 고칠 수 있다.
"""
import os
import sys
import json
import time
import argparse
import warnings
from collections import Counter

warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "eda"))
import pipeline as P
import ocrlib

IMG_DIR = os.path.join(ROOT, "images", "상품사진입니다")
OUT_DIR = os.path.join(ROOT, "eda", "out")

_PIPE = None


def get_pipe():
    global _PIPE
    if _PIPE is None:
        _PIPE = P.DatePipeline(ocrlib.build_ocr(det_side=640, threads=1),
                               long_side=640, rotations=(0, 90),
                               adaptive=True, keep=None)
    return _PIPE


def work(f):
    """예측 + OCR 원문(모든 해상도 단계) 수집."""
    pipe = get_pipe()
    img = P.imread(os.path.join(IMG_DIR, f))
    out = {"file": f, "text": "", "pred": ("NONE", "NONE", "NONE")}
    if img is None:
        return out
    try:
        fin, cands, anc = pipe.predict_image(img)
        r = P.to_row("x", fin)
        out["pred"] = (r["year"], r["month"], r["day"])
        out["cands"] = sorted(cands)
        out["anchors"] = sorted(anc)
    except Exception:
        out["cands"], out["anchors"] = [], []
    # 진단용: 모든 해상도에서 읽은 원문을 전부 모은다
    texts = []
    for side in (640, 512, 448, 384):
        try:
            _, _, _, _, t = pipe._scan(img, side)
            texts.append(t)
        except Exception:
            pass
    out["text"] = "\n".join(texts)
    return out


def split3(v):
    if v is None or v == "NONE":
        return ("NONE", "NONE", "NONE")
    p = str(v).split("-")
    return tuple(p) if len(p) == 3 else ("NONE", "NONE", "NONE")


def digits_only(s):
    return "".join(ch for ch in s if ch.isdigit())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    lab = json.load(open(os.path.join(OUT_DIR, "truth_holdout2.json"),
                         encoding="utf-8"))["labels"]

    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rs = list(ex.map(work, list(lab), chunksize=2))
    print(f"수집 {time.time()-t0:.0f}s", flush=True)

    rows = []
    for r in rs:
        g = split3(lab[r["file"]])
        p = r["pred"]
        if g == p:
            continue
        gy, gm, gd = g
        # 정답 날짜의 숫자열이 OCR 원문 안에 (구분자 무시하고) 존재하는가
        txt_d = digits_only(r["text"])
        pats = []
        if gy != "NONE" and gm != "NONE" and gd != "NONE":
            pats = [gy + gm + gd, gy[2:] + gm + gd, gd + gm + gy[2:], gd + gm + gy]
        elif gm != "NONE" and gd != "NONE":
            pats = [gm + gd]
        elif gy != "NONE" and gm != "NONE":
            pats = [gy + gm, gm + gy]
        readable = any(x in txt_d for x in pats) if pats else False

        if p == ("NONE", "NONE", "NONE"):
            kind = "미검출-읽긴읽음" if readable else "미검출-못읽음"
        elif g[1:] == p[1:]:
            kind = "연도만 오독"
        elif g[0] == p[0]:
            kind = "월/일 오독"
        else:
            kind = "완전히 다름"
        rows.append({"file": r["file"], "truth": lab[r["file"]],
                     "pred": "-".join(p), "kind": kind, "readable": readable,
                     "cands": r.get("cands", []), "anchors": r.get("anchors", []),
                     "text": r["text"]})

    import csv
    with open(os.path.join(OUT_DIR, "holdout2_errors.csv"), "w",
              encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "truth", "pred", "kind", "readable", "cands", "anchors"])
        for r in rows:
            w.writerow([r["file"], r["truth"], r["pred"], r["kind"], r["readable"],
                        "|".join(r["cands"]), "|".join(r["anchors"])])

    L = ["# EDA 40 — 2차 홀드아웃 오답 심층 분해\n",
         f"- 오답 **{len(rows)}장** / 200장",
         "- `읽긴읽음` = 정답 날짜의 숫자열이 OCR 원문 안에 존재 → **규칙으로 고칠 수 있다**",
         "- `못읽음` = OCR 이 아예 못 읽음 → 인식 한계",
         "",
         "| 유형 | 장수 | 고칠 수 있나 |",
         "| --- | ---: | --- |"]
    can = {"미검출-읽긴읽음": "✅ 파싱 규칙", "미검출-못읽음": "❌ 인식 한계",
           "연도만 오독": "△ 일부", "월/일 오독": "❌ 인식 오류",
           "완전히 다름": "△ 선택 규칙"}
    for k, v in Counter(r["kind"] for r in rows).most_common():
        L.append(f"| {k} | {v} | {can.get(k,'')} |")
    L.append("")

    fixable = [r for r in rows if r["kind"] == "미검출-읽긴읽음"]
    L.append(f"## 규칙으로 고칠 수 있는 {len(fixable)}장\n")
    L.append("정답 숫자열이 OCR 원문에 있는데 파싱이 못 잡은 경우다.\n")
    L.append("| 파일 | 정답 | OCR 원문에서 해당 부분 |")
    L.append("| --- | --- | --- |")
    for r in fixable:
        gy, gm, gd = split3(r["truth"])
        hit = ""
        for line in r["text"].split("\n"):
            dd = digits_only(line)
            for pat in ([gy + gm + gd, gy[2:] + gm + gd, gd + gm + gy[2:], gd + gm + gy]
                        if gy != "NONE" else [gm + gd]):
                if pat and pat in dd:
                    hit = line.strip()[:50]
                    break
            if hit:
                break
        L.append(f"| `{r['file']}` | {r['truth']} | `{hit}` |")
    L.append("")

    L.append("## 그 외 오답\n")
    L.append("| 파일 | 정답 | 출력 | 유형 |")
    L.append("| --- | --- | --- | --- |")
    for r in rows:
        if r["kind"] != "미검출-읽긴읽음":
            L.append(f"| `{r['file']}` | {r['truth']} | {r['pred']} | {r['kind']} |")

    with open(os.path.join(OUT_DIR, "40_holdout2_errors.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L[:40]))


if __name__ == "__main__":
    main()
