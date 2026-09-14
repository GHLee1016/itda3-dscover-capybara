"""
EDA 10단계 — 전처리 실험 (recall 상한 34.5% 를 올리려는 시도)

09단계까지의 결론: 비용 문제는 박스 필터로 풀리지만, recall 상한은 그대로다.
실패 원인은 검출 누락 · 원본 해상도 부족 · 인접 텍스트 간섭 세 갈래였고,
이 중 **검출 누락**(세로 잉크젯 · 저대비 · 곡면 조명)이 전처리로 공략 가능한 부분이다.

두 축을 함께 본다:
  변형(variant) — preprocs.VARIANTS
  적용범위(mode)
    both     : det 와 rec 모두 전처리본으로
    det_only : det 만 전처리본으로, rec 크롭은 **원본**에서
               (이진화처럼 det 에는 도움되지만 rec 에는 해로운 변형을 살리기 위함)

지표
  coverage    날짜가 하나라도 잡힌 이미지 비율 (사람 라벨 불필요)
  plausible   잡힌 날짜의 연도가 2015~2035 인 비율 (정밀도 대용)
  truth       육안 정답 10장에 대한 정확 recall
  sec/img     0°+90° 2회 합계
"""
import os
import sys
import json
import time
import argparse
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "eda"))
from datex import find_dates
from preprocs import VARIANTS
import ocrlib

IMG_DIR = os.path.join(ROOT, "images", "상품사진입니다")
OUT_DIR = os.path.join(ROOT, "eda", "out")

_OCR = None


def get_ocr():
    global _OCR
    if _OCR is None:
        _OCR = ocrlib.build_ocr(det_side=960, threads=1)
    return _OCR


def dates_of(txt):
    return {"%04d-%02d-%02d" % (d["y"], d["m"], d["d"])
            for d in find_dates(txt or "")
            if not d["in_long_digits"] and not d["day_missing"]}


def run_one(orig_small, variant_fn, mode, ocr):
    """0°+90° 합집합으로 날짜 후보를 뽑는다."""
    det, rec = ocr.text_detector, ocr.text_recognizer
    got, el = set(), 0.0
    for deg in (0, 90):
        base = ocrlib.rotate(orig_small, deg)
        t = time.time()
        proc = variant_fn(base)
        boxes, _ = det(proc)
        if boxes is not None and len(boxes):
            src = base if mode == "det_only" else proc
            crops = []
            for b in boxes:
                c = ocrlib.crop_quad(src, b)
                if c is None:
                    continue
                c = ocrlib.fit_rec_height(c)
                if c is not None:
                    crops.append(c)
            if crops:
                rr = rec(crops)
                rr = rr[0] if isinstance(rr, tuple) else rr
                got |= dates_of("\n".join(r[0] for r in rr))
        el += time.time() - t
    return got, el


def work(args):
    path, configs = args
    ocr = get_ocr()
    img = ocrlib.imread(path)
    if img is None:
        return None
    small = ocrlib.downscale(img, 960)
    out = {"file": os.path.basename(path)}
    for name, mode in configs:
        got, el = run_one(small, VARIANTS[name], mode, ocr)
        out[f"{name}|{mode}"] = {"dates": sorted(got), "t": round(el, 3)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120, help="평가 표본 장수")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--variants", default="all")
    ap.add_argument("--modes", default="both,det_only")
    ap.add_argument("--out", default="10_preprocess.md")
    ap.add_argument("--cache", default="preproc_raw.json")
    a = ap.parse_args()

    truth = json.load(open(os.path.join(OUT_DIR, "truth.json"), encoding="utf-8"))
    meta = pd.read_csv(os.path.join(OUT_DIR, "metadata.csv"))
    rng = np.random.default_rng(2026)
    pool = [f for f in meta["file"] if f not in truth]
    sample = list(rng.choice(pool, size=min(a.n, len(pool)), replace=False))
    files = list(truth.keys()) + sample

    names = list(VARIANTS) if a.variants == "all" else a.variants.split(",")
    modes = a.modes.split(",")
    configs = []
    for nm in names:
        for md in modes:
            if nm == "none" and md == "det_only":
                continue                      # none 은 both 와 동일
            configs.append((nm, md))
    print(f"이미지 {len(files)}장 (정답 {len(truth)} + 표본 {len(sample)}) × "
          f"설정 {len(configs)}개 × 회전 2 = {len(files)*len(configs)*2:,} OCR 통과", flush=True)

    cache = os.path.join(OUT_DIR, a.cache)
    t0 = time.time()
    results = []
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(work, [(os.path.join(IMG_DIR, f), configs) for f in files],
                                     chunksize=1), 1):
            if r:
                results.append(r)
            if i % 10 == 0:
                print(f"  {i}/{len(files)}  {time.time()-t0:.0f}s", flush=True)
    json.dump(results, open(cache, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"수집 완료 {time.time()-t0:.0f}s", flush=True)

    # ---------------- 집계 ----------------
    byfile = {r["file"]: r for r in results}
    rows = []
    for nm, md in configs:
        key = f"{nm}|{md}"
        cov = pl_ok = pl_tot = 0
        tsum = 0.0
        hit = 0
        n_eval = 0
        for f, r in byfile.items():
            if key not in r:
                continue
            d = r[key]
            tsum += d["t"]
            if f in truth:
                hit += truth[f] in d["dates"]
                continue                       # 정답셋은 coverage 통계에서 제외
            n_eval += 1
            if d["dates"]:
                cov += 1
            for s in d["dates"]:
                pl_tot += 1
                pl_ok += 2015 <= int(s[:4]) <= 2035
        rows.append({
            "variant": nm, "mode": md,
            "coverage": cov / max(n_eval, 1),
            "plausible": pl_ok / max(pl_tot, 1),
            "n_dates": pl_tot,
            "truth": hit, "truth_n": len(truth),
            "sec": tsum / max(len(byfile), 1),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "preprocess.csv"), index=False, encoding="utf-8-sig")

    base = df[(df["variant"] == "none")].iloc[0]
    df["d_cov"] = (df["coverage"] - base["coverage"]) * 100
    df = df.sort_values("coverage", ascending=False)

    L = ["# EDA 10 — 전처리 실험\n",
         f"- 평가 표본 **{len([f for f in byfile if f not in truth])}장** (무작위) "
         f"+ 육안 정답 **{len(truth)}장**",
         "- det 입력 960px, 0°+90° 합집합, 사람 라벨 없이 측정 가능한 지표 사용",
         "",
         "| 지표 | 뜻 |",
         "| --- | --- |",
         "| coverage | 날짜가 하나라도 잡힌 이미지 비율 — **높을수록 좋다** |",
         "| plausible | 잡힌 날짜의 연도가 2015~2035 인 비율 — 정밀도 대용, **낮아지면 오탐 증가** |",
         "| truth | 육안 정답 10장 중 맞춘 수 |",
         "| 초/장 | 0°+90° 2회 합계 (단일 프로세스) |",
         "",
         "> `both` = det·rec 모두 전처리본 / `det_only` = det 만 전처리본, rec 는 원본 크롭",
         "",
         "## 결과\n",
         "| 변형 | 적용 | coverage | Δ | plausible | truth | 초/장 |",
         "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for _, r in df.iterrows():
        mark = "**" if r["d_cov"] > 0.5 else ""
        L.append(f"| `{r['variant']}` | {r['mode']} | {mark}{r['coverage']*100:.1f}%{mark} | "
                 f"{r['d_cov']:+.1f}%p | {r['plausible']*100:.1f}% | "
                 f"{int(r['truth'])}/{int(r['truth_n'])} | {r['sec']:.2f} |")
    L.append("")
    b = df.iloc[0]
    L.append(f"- 기준선 `none`: coverage **{base['coverage']*100:.1f}%**, "
             f"plausible {base['plausible']*100:.1f}%, truth {int(base['truth'])}/10, "
             f"{base['sec']:.2f}초")
    L.append(f"- 최고 coverage: `{b['variant']}` ({b['mode']}) "
             f"**{b['coverage']*100:.1f}%** ({b['d_cov']:+.1f}%p)")
    L.append("")
    gain = df[df["d_cov"] > 0.5]
    if len(gain) == 0:
        L.append("> ❌ **coverage 를 유의미하게 올린 전처리가 없다.** "
                 "검출 누락은 전역 전처리로 풀리지 않는다는 뜻이다.")
    else:
        L.append(f"> coverage 를 올린 변형 {len(gain)}개. 단 `plausible` 이 함께 떨어지면 "
                 "오탐이 늘어난 것이므로 실질 개선이 아니다 — 두 지표를 같이 봐야 한다.")
    with open(os.path.join(OUT_DIR, a.out), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
