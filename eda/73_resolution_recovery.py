"""
EDA 73단계 — 실패 271장을 더 높은 해상도로 다시 읽으면 몇 장이 회수되는가

72단계에서 실패 표본을 눈으로 본 결과, 네 장 모두 소비기한이 선명하게 인쇄돼
있었다. 라벨이 없어서 못 배운 게 아니라 **읽지를 못한** 것이다.

가설: 주력 경로가 긴 변을 1024px 로 줄이는 것이 원인이다.
      4032px 원본을 1024px 로 줄이면 4배 축소다. 캔 뚜껑 각인이나 점자 인쇄는
      그 과정에서 뭉개진다. 실제로 3024x4032 원본(000288)에서 OCR 조각이 0개였다.

이 실험이 라벨링보다 먼저인 이유: 라벨링은 사람 시간이 3~4시간이고 이건 자동이다.
회수되는 만큼 라벨링 대상이 줄고, 남는 것이 진짜 어려운 케이스가 된다.

비용 구조상 안전하다: 고해상도 재시도는 **이미 실패한 8.6%에만** 건다.
성공한 이미지에는 닿지 않으므로 오답이 늘 위험이 없다(기존 사다리와 같은 논리).

    python eda/73_resolution_recovery.py --workers 4
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

import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "eda"))

IMG_DIR = os.path.join(ROOT, "images", "상품사진입니다")
OUT_DIR = os.path.join(ROOT, "eda", "out")
RAPID_DIR = os.path.join(ROOT, "weights", "rapidocr")

# 1024 는 현재 값이라 기준선. 위로 올려 가며 본다. 0 은 원본(4096 상한).
SIDES = [1024, 1536, 2048, 3072]

_S = None


def setup():
    global _S
    if _S is None:
        import combined
        import team2_rules
        rapid, _ = combined.build_rapid_ocr(RAPID_DIR)
        _S = (combined, team2_rules, rapid)
    return _S


def read_at(rapid, rules, img, side):
    h, w = img.shape[:2]
    if max(h, w) > side:
        s = side / max(h, w)
        a = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))),
                       interpolation=cv2.INTER_AREA)
    else:
        a = img            # 원본이 더 작으면 키우지 않는다
    try:
        r = rapid(a)
        texts = list(r.txts) if getattr(r, "txts", None) else []
    except Exception:
        return None, 0
    if not texts:
        return None, 0
    full = rules.choose_expiration_date(texts)
    if full:
        y, m, d = full.split("-")
        return {"year": y, "month": m, "day": d}, len(texts)
    return rules.extract_partial_date(texts), len(texts)


def work(fname):
    combined, rules, rapid = setup()
    img = combined.imread_any(os.path.join(IMG_DIR, fname))
    if img is None:
        return None
    h, w = img.shape[:2]
    out = {"file": fname, "w": w, "h": h, "first_ok": None, "by_side": {}}
    for side in SIDES:
        if side > 1024 and max(h, w) <= 1024:
            out["by_side"][str(side)] = {"ok": False, "sec": 0.0, "n": 0}
            continue          # 원본이 작으면 올릴 여지가 없다
        t0 = time.time()
        got, n = read_at(rapid, rules, img, side)
        el = time.time() - t0
        ok = got is not None and any(v != "NONE" for v in got.values())
        out["by_side"][str(side)] = {"ok": bool(ok), "sec": round(el, 3), "n": n}
        if ok:
            out["by_side"][str(side)]["pred"] = "-".join(
                str(got.get(k, "NONE") or "NONE") for k in ("year", "month", "day"))
            if out["first_ok"] is None:
                out["first_ok"] = side
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    rows = json.load(open(os.path.join(OUT_DIR, "69_failures.json"), encoding="utf-8"))
    fails = [r for r in rows if r["stage"] == "none" and r["cause"] != "load_fail"]
    files = [r["file"] for r in fails]
    cause = {r["file"]: r["cause"] for r in fails}
    if a.limit:
        files = files[:a.limit]
    print(f"실패 {len(files)}장 × 해상도 {SIDES}", flush=True)

    lab = {}
    for fn in ("truth_v4.json", "truth_holdout.json"):
        p = os.path.join(OUT_DIR, fn)
        if os.path.exists(p):
            lab.update(json.load(open(p, encoding="utf-8"))["labels"])

    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    res = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(work, files, chunksize=2), 1):
            if r:
                res.append(r)
            if i % 50 == 0:
                print(f"  {i}/{len(files)}  {time.time()-t0:.0f}s", flush=True)
    wall = time.time() - t0

    json.dump(res, open(os.path.join(OUT_DIR, "73_resolution.json"), "w",
                        encoding="utf-8"), ensure_ascii=False)

    n = len(res)

    def s3(v):
        if not v or v == "NONE":
            return ("NONE",) * 3
        q = str(v).split("-")
        return tuple(q) if len(q) == 3 else ("NONE",) * 3

    # 해상도별 누적 회수
    L = ["# EDA 73 — 고해상도 재시도로 실패 이미지를 회수할 수 있는가\n",
         "72단계에서 실패 표본을 눈으로 보니 네 장 모두 소비기한이 선명하게 인쇄돼",
         "있었다. 라벨이 없어서 못 배운 것이 아니라 **읽지 못한** 것이다.",
         "",
         "주력 경로는 긴 변을 1024px 로 줄인다. 4032px 원본이면 4배 축소이고,",
         "캔 뚜껑 각인이나 점자 인쇄는 그 과정에서 뭉개진다. 해상도만 올려 다시 읽어 본다.",
         "",
         f"- 대상 {n}장, {wall/60:.1f}분 ({a.workers}워커)",
         "",
         "## 해상도별 회수\n",
         "| 긴 변 | 이 해상도에서 날짜가 나온 장수 | 누적 회수 | 장당 소요 |",
         "| ---: | ---: | ---: | ---: |"]
    cum = set()
    for side in SIDES:
        k = str(side)
        okf = {r["file"] for r in res if r["by_side"].get(k, {}).get("ok")}
        cum |= okf
        secs = [r["by_side"][k]["sec"] for r in res
                if r["by_side"].get(k, {}).get("sec", 0) > 0]
        avg = sum(secs) / len(secs) if secs else 0
        L.append(f"| {side}px | {len(okf)} | **{len(cum)} ({len(cum)/n*100:.1f}%)** | "
                 f"{avg:.2f}초 |")
    L.append("")

    base = {r["file"] for r in res if r["by_side"].get("1024", {}).get("ok")}
    gain = cum - base
    L.append(f"> 1024px 는 현재 설정이라 기준선이다. 여기서 이미 {len(base)}장이 나오는 것은")
    L.append(f"> 69단계 실행과 미세한 차이(전처리 경로) 때문이다.")
    L.append(f"> **해상도를 올려 새로 회수한 것은 {len(gain)}장 ({len(gain)/n*100:.1f}%)** 이다.")
    L.append("")

    # 처음 성공한 해상도 분포
    fo = Counter(r["first_ok"] for r in res if r["first_ok"])
    L.append("## 어느 해상도에서 처음 읽혔나\n")
    L.append("| 긴 변 | 장수 |")
    L.append("| ---: | ---: |")
    for side in SIDES:
        L.append(f"| {side}px | {fo.get(side,0)} |")
    L.append(f"| 끝내 실패 | {n - sum(fo.values())} |")
    L.append("")

    # 라벨이 있는 것에 한해 정답 여부
    lab_rec = [r for r in res if r["first_ok"] and r["file"] in lab]
    if lab_rec:
        okc = 0
        rowsl = []
        for r in lab_rec:
            pred = r["by_side"][str(r["first_ok"])].get("pred", "NONE-NONE-NONE")
            g, p = s3(lab[r["file"]]), s3(pred)
            hit = sum(g[i] == p[i] for i in range(3))
            okc += hit == 3
            rowsl.append((r["file"], lab[r["file"]], pred, r["first_ok"], hit))
        L.append(f"## 회수한 것이 맞는가 (라벨 있는 {len(lab_rec)}장)\n")
        L.append(f"완전일치 **{okc}/{len(lab_rec)}장**. 회수가 곧 정답은 아니므로 함께 본다.")
        L.append("")
        L.append("| 파일 | 정답 | 회수값 | 해상도 | 맞은 필드 |")
        L.append("| --- | --- | --- | ---: | ---: |")
        for f, g, p, s, h in rowsl[:25]:
            L.append(f"| `{f}` | {g} | {p} | {s}px | {h}/3 |")
        L.append("")

    # 원인별 회수율
    L.append("## 69단계 원인별 회수율\n")
    KO = {"no_text": "글자 못 찾음", "text_no_digits": "숫자 줄 없음",
          "digits_no_date": "숫자는 있으나 날짜 아님"}
    L.append("| 원인 | 대상 | 회수 | 회수율 |")
    L.append("| --- | ---: | ---: | ---: |")
    for c in ("digits_no_date", "text_no_digits", "no_text"):
        sub = [r for r in res if cause.get(r["file"]) == c]
        if not sub:
            continue
        rec = sum(1 for r in sub if r["first_ok"])
        L.append(f"| {KO[c]} | {len(sub)} | {rec} | {rec/len(sub)*100:.1f}% |")
    L.append("")

    # 비용 추정
    extra = sum(r["by_side"].get(k, {}).get("sec", 0)
                for r in res for k in ("1536", "2048", "3072"))
    L.append("## 비용\n")
    L.append(f"- 실패 {n}장에 고해상도 3단계를 모두 걸면 총 {extra:.0f}초 추가")
    L.append(f"- 500장 환산: 실패율 8.6% 가정 시 약 {extra/n*0.086*500:.0f}초 추가")
    L.append("- 현재 최종 구성이 500장 460초(제한의 19%)이므로 여유가 있다")
    L.append("")
    L.append("## 판단\n")
    if len(gain) / n >= 0.15:
        L.append(f"**해상도 사다리를 주력 경로에 추가할 가치가 있다.** {len(gain)}장이 코드만으로 회수된다.")
        L.append("라벨링 대상은 나머지로 줄어들고, 남은 것이 진짜 어려운 케이스다.")
    else:
        L.append(f"회수가 {len(gain)}장({len(gain)/n*100:.1f}%)에 그친다. 해상도는 주원인이 아니다.")
        L.append("라벨링·엔진 쪽으로 방향을 돌리는 것이 맞다.")
    L.append("")
    L.append("> ⚠️ 이 수치는 실패 집합 안에서의 회수율이다. 전체 성능 개선폭은")
    L.append("> **동결한 홀드아웃2**에 적용해 따로 재야 한다.")

    with open(os.path.join(OUT_DIR, "73_resolution_recovery.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L))
    print()
    print("\n".join(L[8:30]))
    print("\n→ eda/out/73_resolution_recovery.md")


if __name__ == "__main__":
    main()
