"""
EDA 23단계 — det 입력 크기 하향 실험

EDA 5 는 960px **이상**만 시험했다(3200px 까지 올려도 recall 불변).
그런데 rec 를 필터로 줄이고 나니 이제 **det 가 지배적 비용**이다(EDA 22).
det 비용은 입력 면적에 비례하므로 960 → 768 이면 0.64배, 640 이면 0.44배다.

960 미만에서 정확도가 얼마나 떨어지는지는 한 번도 재지 않았다. 여기서 잰다.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed-n", type=int, default=80)
    ap.add_argument("--sides", default="640,768,960")
    ap.add_argument("--out", default="23_det_side.md")
    a = ap.parse_args()

    truth = json.load(open(os.path.join(OUT_DIR, "truth_v2.json"), encoding="utf-8"))
    meta = pd.read_csv(os.path.join(OUT_DIR, "metadata.csv"))
    rng = np.random.default_rng(777)
    pool = [f for f in meta["file"] if f not in truth]
    speed_files = [os.path.join(IMG_DIR, f)
                   for f in rng.choice(pool, a.speed_n, replace=False)]

    rows = []
    for side in [int(s) for s in a.sides.split(",")]:
        ocr = ocrlib.build_ocr(det_side=side, threads=0)
        pipe = P.DatePipeline(ocr, long_side=side, rotations=(0, 90),
                              adaptive=True, keep=None)
        # 정확도
        hit = 0
        for f, gt in truth.items():
            final, _, _ = pipe.predict_path(os.path.join(IMG_DIR, f))
            hit += (final == gt)
        # 속도
        t0 = time.time()
        for f in speed_files:
            pipe.predict_path(f)
        per = (time.time() - t0) / len(speed_files)
        rows.append({"side": side, "hit": hit, "n": len(truth),
                     "per": per, "total": per * 3352})
        print(f"  det={side}: 정확도 {hit}/{len(truth)}, {per:.3f}s/img, "
              f"3352장 {per*3352:.0f}s", flush=True)

    L = ["# EDA 23 — det 입력 크기 하향\n",
         f"- 정확도: 사람 정답 {len(truth)}장 / 속도: 무작위 {a.speed_n}장",
         "- **단일 프로세스**, 박스 필터 없음(keep=None), 0°+90° 적응형",
         "",
         "> EDA 5 는 960px 이상만 봤다. rec 를 필터로 줄이고 나면 det 가 지배적",
         "> 비용이 되므로, 960 **미만**에서의 손실을 여기서 처음 잰다.",
         "",
         "| det 입력 | 정확도 | 초/장 | 3,352장 | 960 대비 |",
         "| ---: | ---: | ---: | ---: | ---: |"]
    base = [r for r in rows if r["side"] == 960]
    b = base[0] if base else rows[-1]
    for r in rows:
        L.append(f"| {r['side']}px | **{r['hit']}/{r['n']} ({r['hit']/r['n']*100:.1f}%)** | "
                 f"{r['per']:.3f} | {r['total']:.0f}s | {r['per']/b['per']*100:.0f}% |")
    L.append("")
    best = max(rows, key=lambda r: (r["hit"], -r["per"]))
    L.append(f"> 정확도 최고: **{best['side']}px** ({best['hit']}/{best['n']})")
    cheap = [r for r in rows if r["hit"] >= b["hit"] - 1]
    if cheap:
        c = min(cheap, key=lambda r: r["per"])
        L.append(f">")
        L.append(f"> 정확도 손실 1장 이내에서 가장 빠른 설정: **{c['side']}px** "
                 f"({c['per']:.3f}s/장, 960px 대비 {c['per']/b['per']*100:.0f}%)")
    with open(os.path.join(OUT_DIR, a.out), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
