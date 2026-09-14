"""
EDA 22단계 — 단일 프로세스 속도 보정

지금까지의 속도 측정은 전부 **4 프로세스 병렬**이었다. 그런데 predict.ipynb 는
단일 프로세스 순차 루프다. 사다리의 각 단계가 단일 프로세스에서 실제로 몇 초/장인지
재지 않으면 시간 예산 제어기가 도달 불가능한 목표를 쫓게 된다.

실제로 3,352장 실행이 두 번 연속 2,410초로 타임아웃을 넘겼다.
원인 후보: 박스 수 하한(FILTER_MIN_BOXES)이 너무 높아 keep 을 낮춰도 절감이 없다.
"""
import os
import sys
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--min-boxes", type=int, default=4)
    ap.add_argument("--out", default="22_calibrate_speed.md")
    a = ap.parse_args()

    P.FILTER_MIN_BOXES = a.min_boxes

    meta = pd.read_csv(os.path.join(OUT_DIR, "metadata.csv"))
    rng = np.random.default_rng(4242)
    files = [os.path.join(IMG_DIR, f)
             for f in rng.choice(meta["file"].values, a.n, replace=False)]

    ocr = ocrlib.build_ocr(det_side=960, threads=0)   # 0 = paddle 기본(코어 전부)
    rows = []
    for keep in P.DatePipeline.__init__.__defaults__ and [None, 0.60, 0.40, 0.30, 0.20, 0.12, 0.08]:
        pipe = P.DatePipeline(ocr, long_side=960, rotations=(0, 90),
                              adaptive=True, keep=keep)
        t0 = time.time()
        found = 0
        nbox = 0
        for f in files:
            final, cands, _ = pipe.predict_path(f)
            found += bool(final)
        el = time.time() - t0
        per = el / len(files)
        rows.append({"keep": keep, "per": per, "cov": found / len(files),
                     "total_3352": per * 3352})
        print(f"  keep={keep}: {per:.3f}s/img, 3352장 {per*3352:.0f}s, "
              f"coverage {found/len(files)*100:.0f}%", flush=True)

    L = ["# EDA 22 — 단일 프로세스 속도 보정\n",
         f"- 무작위 **{a.n}장**, **단일 프로세스** (predict.ipynb 와 동일 조건)",
         f"- `FILTER_MIN_BOXES = {a.min_boxes}`",
         "",
         "> 이전 속도 측정은 전부 4 프로세스 병렬이었다. 노트북은 순차 루프이므로",
         "> 사다리 각 단계의 실제 도달 속도를 여기서 확정한다.",
         "",
         "| keep | 초/장 | 3,352장 환산 | 타임아웃(2400s) 대비 | coverage |",
         "| --- | ---: | ---: | ---: | ---: |"]
    for r in rows:
        k = "없음" if r["keep"] is None else f"{r['keep']:.2f}"
        mark = "✅" if r["total_3352"] < 2400 else "❌"
        L.append(f"| {k} | **{r['per']:.3f}** | {r['total_3352']:.0f}s {mark} | "
                 f"{r['total_3352']/2400*100:.0f}% | {r['cov']*100:.0f}% |")
    L.append("")
    ok = [r for r in rows if r["total_3352"] < 1800]
    if ok:
        b = ok[0]
        L.append(f"> 예산 1,800초를 만족하는 가장 약한 필터: "
                 f"**keep={b['keep']}** ({b['per']:.3f}s/장, {b['total_3352']:.0f}s)")
    else:
        L.append("> ❌ **어떤 keep 도 1,800초 안에 들어오지 못한다.** "
                 "사다리만으로는 부족하고 다른 절감 수단이 필요하다.")
    with open(os.path.join(OUT_DIR, a.out), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
