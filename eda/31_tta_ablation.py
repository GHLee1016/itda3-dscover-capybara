"""
EDA 31단계 — TTA / 회전 확장 ablation (정답 137장)

최대 손실원은 미검출 27장(79 필드, 전체 손실의 55%)이다.
예산은 제한의 19% 밖에 안 쓰므로(581장 447초) 더 쓸 여유가 충분하다.

이제 정답이 137장이라 **커버리지가 아니라 실제 부분점수로** 판단할 수 있다.
(EDA 19 에서 커버리지만 보고 판단했다가 무의미한 규칙을 채택할 뻔했다.)
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

CONFIGS = [
    {"name": "A. 640 단독",            "side": 640, "fb": (),            "rot": (0, 90)},
    {"name": "B. 640→512→448 (현재)",  "side": 640, "fb": (512, 448),    "rot": (0, 90)},
    {"name": "C. B + 180/270 회전",    "side": 640, "fb": (512, 448),    "rot": (0, 90, 180, 270)},
    {"name": "D. 640→512→448→384",    "side": 640, "fb": (512, 448, 384), "rot": (0, 90)},
]

_OCR = None


def get_ocr():
    global _OCR
    if _OCR is None:
        _OCR = ocrlib.build_ocr(det_side=640, threads=1)
    return _OCR


def work(args):
    path, ci = args
    cfg = CONFIGS[ci]
    ocr = get_ocr()
    pipe = P.DatePipeline(ocr, long_side=cfg["side"], rotations=cfg["rot"],
                          adaptive=True, keep=None)
    pipe.fallback_sides = cfg["fb"]
    t = time.time()
    try:
        final, _, _ = pipe.predict_path(path)
    except Exception:
        final = None
    row = P.to_row(os.path.splitext(os.path.basename(path))[0], final)
    row["sec"] = time.time() - t
    return row


def split3(v):
    if v is None or v == "NONE":
        return ("NONE", "NONE", "NONE")
    p = str(v).split("-")
    return tuple(p) if len(p) == 3 else ("NONE", "NONE", "NONE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    t = json.load(open(os.path.join(OUT_DIR, "truth_v4.json"), encoding="utf-8"))
    labels = t["labels"]
    mem = t["membership"]
    evalset = set(mem["eval_random"])
    files = [os.path.join(IMG_DIR, f) for f in labels]

    from concurrent.futures import ProcessPoolExecutor
    results = {}
    for ci, cfg in enumerate(CONFIGS):
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            rs = list(ex.map(work, [(f, ci) for f in files], chunksize=2))
        wall = time.time() - t0
        results[cfg["name"]] = {"rows": rs, "wall": wall}
        print(f"  {cfg['name']}: {wall:.0f}s ({wall/len(files):.3f}s/장)", flush=True)

    L = ["# EDA 31 — TTA / 회전 확장 ablation\n",
         f"- 정답 **{len(labels)}장** 기준. 부분점수(필드 평균)로 판정",
         f"- {a.workers} 프로세스 병렬 wall-clock",
         "",
         "> 최대 손실원은 미검출 27장(79 필드, 전체 손실의 55%)이다.",
         "> 예산은 제한의 19% 밖에 안 쓰므로 더 쓸 여유가 있다.",
         "",
         "| 설정 | 부분점수 (전체) | 부분점수 (무작위셋) | 미검출 | 초/장 | 500장 환산 |",
         "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for cfg in CONFIGS:
        r = results[cfg["name"]]
        pred = {x["image_id"]: (x["year"], x["month"], x["day"]) for x in r["rows"]}
        ok = tot = 0
        ok_e = tot_e = 0
        miss = 0
        for f, gv in labels.items():
            iid = os.path.splitext(f)[0]
            if iid not in pred:
                continue
            g, p = split3(gv), pred[iid]
            hit = sum(g[i] == p[i] for i in range(3))
            ok += hit
            tot += 3
            if f in evalset:
                ok_e += hit
                tot_e += 3
            if p == ("NONE", "NONE", "NONE") and g != ("NONE", "NONE", "NONE"):
                miss += 1
        per = r["wall"] / len(files)
        L.append(f"| {cfg['name']} | **{ok/tot*100:.1f}%** | {ok_e/max(tot_e,1)*100:.1f}% | "
                 f"{miss} | {per:.2f} | {per*500:.0f}초 |")
    L.append("")
    L.append("> 500장 환산이 2,400초를 넘지 않아야 한다. 모두 여유가 크다.")
    L.append("")
    L.append("> ⚠️ 무작위셋(43장)이 대표 수치지만 표본이 작아 ±15%p 수준이다. "
             "전체 137장 수치가 설정 간 **비교**에는 더 안정적이다.")

    with open(os.path.join(OUT_DIR, "31_tta_ablation.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
