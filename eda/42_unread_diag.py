"""
EDA 42단계 — '못 읽음' 44장의 원인 규명

오답 이미지를 눈으로 보니 상당수가 육안으로는 선명한데 못 읽었다
(000876 '2021.07.01 까지', 002172 '20.06.24 J').

가설: 날짜 영역이 전체 이미지 대비 너무 작아, 640px 로 줄이면 글자가 뭉개진다.
      → 원본 해상도가 높을수록 손해라는 뜻이고, 이는 '작게 넣을수록 좋다'는
        EDA 23/24 의 결론과 정면으로 충돌한다. 평균적으로는 작게가 맞지만
        이 부분집합에서는 반대일 수 있다.

검증: 못 읽은 44장과 맞춘 116장의 원본 해상도 분포를 비교하고,
      못 읽은 것들을 **원본 해상도 그대로** 다시 돌려본다.
"""
import os
import sys
import json
import time
import argparse
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "eda"))
import pipeline as P
import ocrlib

IMG_DIR = os.path.join(ROOT, "images", "상품사진입니다")
OUT_DIR = os.path.join(ROOT, "eda", "out")

_OCR = None


def get_ocr(side):
    global _OCR
    if _OCR is None:
        _OCR = ocrlib.build_ocr(det_side=side, threads=1)
    return _OCR


def work(args):
    f, side = args
    ocr = get_ocr(2560)          # det 상한을 넉넉히 두고 long_side 로 제어
    pipe = P.DatePipeline(ocr, long_side=side, rotations=(0, 90),
                          adaptive=True, keep=None)
    pipe.fallback_sides = ()     # 단일 해상도 효과만 본다
    try:
        fin, _, _ = pipe.predict_path(os.path.join(IMG_DIR, f))
    except Exception:
        fin = None
    r = P.to_row("x", fin)
    return {"file": f, "side": side,
            "pred": (r["year"], r["month"], r["day"])}


def split3(v):
    if v is None or v == "NONE":
        return ("NONE", "NONE", "NONE")
    p = str(v).split("-")
    return tuple(p) if len(p) == 3 else ("NONE", "NONE", "NONE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--sides", default="640,1280,1920")
    a = ap.parse_args()

    lab = json.load(open(os.path.join(OUT_DIR, "truth_holdout2.json"),
                         encoding="utf-8"))["labels"]
    err = pd.read_csv(os.path.join(OUT_DIR, "holdout2_errors.csv"))
    unread = err[err["kind"] == "미검출-못읽음"]["file"].tolist()
    meta = pd.read_csv(os.path.join(OUT_DIR, "metadata.csv")).set_index("file")

    L = ["# EDA 42 — '못 읽음' 44장의 원인 규명\n",
         "오답 이미지를 눈으로 보니 상당수가 육안으로는 선명하다.",
         "`000876`(`2021.07.01 까지`), `002172`(`20.06.24 J`) 처럼.",
         "",
         "가설: **날짜 영역이 전체 이미지 대비 너무 작아 640px 축소 시 뭉개진다.**",
         ""]

    # 1. 해상도 분포 비교
    ok_files = [f for f in lab if f not in set(err["file"])]
    def mp(fs):
        v = [meta.loc[f, "mp"] for f in fs if f in meta.index]
        return np.array(v) if v else np.array([0.0])
    u, o = mp(unread), mp(ok_files)
    L += ["## 1. 원본 해상도 분포\n",
          "| 집합 | 장수 | 중앙 MP | 9MP 초과 비율 |",
          "| --- | ---: | ---: | ---: |",
          f"| 못 읽음 | {len(unread)} | {np.median(u):.2f} | {(u > 9).mean()*100:.0f}% |",
          f"| 맞춤 | {len(ok_files)} | {np.median(o):.2f} | {(o > 9).mean()*100:.0f}% |",
          ""]
    if np.median(u) > np.median(o):
        L.append("> 못 읽은 쪽의 원본 해상도가 **더 높다**. 축소 손실 가설과 일치한다.")
    else:
        L.append("> 해상도 차이가 뚜렷하지 않다. 다른 원인일 수 있다.")
    L.append("")

    # 2. 해상도를 올려 다시 돌려본다
    sides = [int(s) for s in a.sides.split(",")]
    from concurrent.futures import ProcessPoolExecutor
    res = {}
    for side in sides:
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            rs = list(ex.map(work, [(f, side) for f in unread], chunksize=2))
        hit = sum(1 for r in rs
                  if r["pred"] == split3(lab[r["file"]]))
        got = sum(1 for r in rs if r["pred"] != ("NONE",) * 3)
        res[side] = {"hit": hit, "got": got, "sec": (time.time() - t0) / len(unread),
                     "rows": rs}
        print(f"  {side}px: 정확 {hit}/{len(unread)}, 뭐라도 출력 {got}, "
              f"{(time.time()-t0)/len(unread):.2f}s/장", flush=True)

    L += ["## 2. 못 읽은 44장을 해상도별로 다시 돌린 결과\n",
          "| long_side | 정확히 맞춤 | 날짜를 내긴 냄 | 초/장 |",
          "| ---: | ---: | ---: | ---: |"]
    for side in sides:
        r = res[side]
        L.append(f"| {side}px | **{r['hit']}/{len(unread)}** | {r['got']} | {r['sec']:.2f} |")
    L.append("")
    best = max(sides, key=lambda s: res[s]["hit"])
    if res[best]["hit"] > res[sides[0]]["hit"]:
        L.append(f"> ✅ **{best}px 에서 {res[best]['hit']}장을 복구한다** "
                 f"(640px 에서는 {res[sides[0]]['hit']}장). "
                 "날짜 영역이 작은 이미지에는 고해상도가 필요하다는 뜻이다.")
        L.append(">")
        L.append("> EDA 23/24 의 '작게 넣을수록 좋다'는 **평균적으로만** 맞았다. "
                 "부분집합에서는 정반대이며, 두 결론이 공존한다.")
    else:
        L.append("> ❌ 해상도를 올려도 복구되지 않는다. 축소 손실이 원인이 아니다.")
    L.append("")

    L.append("### 고해상도에서 복구된 이미지\n")
    L.append("| 파일 | 정답 | 640px | %dpx |" % best)
    L.append("| --- | --- | --- | --- |")
    b640 = {r["file"]: r["pred"] for r in res[sides[0]]["rows"]}
    bbest = {r["file"]: r["pred"] for r in res[best]["rows"]}
    for f in unread:
        g = split3(lab[f])
        if bbest.get(f) == g and b640.get(f) != g:
            L.append(f"| `{f}` | {lab[f]} | {'-'.join(b640.get(f, ()))} | "
                     f"**{'-'.join(bbest[f])}** |")

    with open(os.path.join(OUT_DIR, "42_unread_diag.md"), "w", encoding="utf-8") as fo:
        fo.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
