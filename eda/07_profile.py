"""
EDA 7단계 — 비용 분해 프로파일링

채점 예산 0.72초/장(4 vCPU) 을 맞추려면 시간이 어디로 가는지 알아야 한다.
 - JPEG 디코딩 / 리사이즈
 - det (텍스트 검출) 1회
 - rec (텍스트 인식) × 박스 수  ← 여기가 지배적일 것으로 예상

추가로 비용 절감 수단의 효과를 측정한다.
 - IMREAD_REDUCED_COLOR_N (DCT 영역 축소 디코딩)
 - rec_batch_num 증가
 - '0° 먼저 → 실패 시에만 90°' 적응형 회전
"""
import os
import sys
import time
import json
import argparse
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "eda"))
from datex import find_dates

IMG_DIR = os.path.join(ROOT, "images", "상품사진입니다")
OUT_DIR = os.path.join(ROOT, "eda", "out")
MODELS = r"C:\Users\Public\ocr_models"


def build(rec_batch=6):
    from paddleocr import PaddleOCR
    return PaddleOCR(
        use_angle_cls=False, lang="korean", show_log=False, use_gpu=False,
        det_limit_side_len=960, det_limit_type="max", rec_batch_num=rec_batch,
        det_model_dir=os.path.join(MODELS, "det", "ml", "Multilingual_PP-OCRv3_det_infer"),
        rec_model_dir=os.path.join(MODELS, "rec", "korean", "korean_PP-OCRv4_rec_infer"),
        cls_model_dir=os.path.join(MODELS, "cls", "ch_ppocr_mobile_v2.0_cls_infer"),
    )


def decode_plain(path):
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)


def decode_reduced(path, target=960):
    """JPEG DCT 영역에서 1/2, 1/4, 1/8 로 축소 디코딩 — 전체 디코딩보다 훨씬 싸다."""
    buf = np.fromfile(path, np.uint8)
    # 헤더만 읽어 원본 크기 파악
    hdr = cv2.imdecode(buf, cv2.IMREAD_REDUCED_COLOR_8)
    if hdr is None:
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    long8 = max(hdr.shape[:2]) * 8
    flag = cv2.IMREAD_COLOR
    for f, div in ((cv2.IMREAD_REDUCED_COLOR_8, 8), (cv2.IMREAD_REDUCED_COLOR_4, 4),
                   (cv2.IMREAD_REDUCED_COLOR_2, 2)):
        if long8 / div >= target:
            flag = f
            break
    return cv2.imdecode(buf, flag)


def fit(a, long_side=960):
    h, w = a.shape[:2]
    s = long_side / max(h, w)
    if s < 1.0:
        a = cv2.resize(a, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return a


def dates_of(lines):
    txt = "\n".join(l[1][0] for l in lines)
    return {"%04d-%02d-%02d" % (d["y"], d["m"], d["d"])
            for d in find_dates(txt)
            if not d["in_long_digits"] and not d["day_missing"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--out", default="07_profile.md")
    a = ap.parse_args()

    meta = pd.read_csv(os.path.join(OUT_DIR, "metadata.csv"))
    sam = meta.sample(min(a.n, len(meta)), random_state=7)
    files = [os.path.join(IMG_DIR, f) for f in sam["file"]]

    ocr = build()
    det = ocr.text_detector
    rec = ocr.text_recognizer

    from importlib.util import spec_from_file_location, module_from_spec
    sp = spec_from_file_location("ts", os.path.join(ROOT, "eda", "06_twostage.py"))
    ts = module_from_spec(sp)
    sys.argv = ["x"]
    sp.loader.exec_module(ts)

    print(f"profiling {len(files)} images ...", flush=True)
    R = []
    for p in files:
        r = {"file": os.path.basename(p)}
        # --- 디코딩 비교 ---
        t = time.time(); a_full = decode_plain(p); r["t_decode_full"] = time.time() - t
        t = time.time(); a_red = decode_reduced(p); r["t_decode_reduced"] = time.time() - t
        r["mp"] = a_full.shape[0] * a_full.shape[1] / 1e6

        t = time.time(); img = fit(a_full); r["t_resize"] = time.time() - t
        # --- det ---
        t = time.time(); boxes, _ = det(img); r["t_det"] = time.time() - t
        r["n_box"] = 0 if boxes is None else len(boxes)
        # --- rec (전체 박스) ---
        crops = []
        if boxes is not None:
            for b in boxes:
                c = ts.crop_quad(img, b, pad=0.0)
                if c is not None:
                    c = ts.fit_rec_height(c)
                    if c is not None:
                        crops.append(c)
        t = time.time()
        rr = rec(crops) if crops else ([],)
        r["t_rec_all"] = time.time() - t
        r["n_crop"] = len(crops)
        recs = rr[0] if isinstance(rr, tuple) else rr
        lines = [[boxes[i], recs[i]] for i in range(min(len(recs), r["n_box"]))]
        r["found_0"] = len(dates_of(lines)) > 0
        R.append(r)

    df = pd.DataFrame(R)
    df.to_csv(os.path.join(OUT_DIR, "profile.csv"), index=False, encoding="utf-8-sig")

    tot = df["t_decode_full"] + df["t_resize"] + df["t_det"] + df["t_rec_all"]
    L = ["# EDA 7 — 비용 분해 프로파일링\n",
         f"- 표본 {len(df)}장, 단일 프로세스, det 입력 960px, 0° 1회 기준",
         f"- CPU: AMD Ryzen 5 7500F (채점 환경 4 vCPU 보다 코어당 빠름 → 낙관적 하한)\n",
         "## 1회 통과(0°) 비용 분해\n",
         "| 단계 | 평균 | 중앙값 | 비중 |",
         "| --- | ---: | ---: | ---: |"]
    for col, lab in [("t_decode_full", "JPEG 디코딩"), ("t_resize", "리사이즈"),
                     ("t_det", "det (검출)"), ("t_rec_all", "rec (인식)")]:
        L.append(f"| {lab} | {df[col].mean()*1000:.0f} ms | {df[col].median()*1000:.0f} ms | "
                 f"{df[col].sum()/tot.sum()*100:.1f}% |")
    L.append(f"| **합계** | **{tot.mean()*1000:.0f} ms** | {tot.median()*1000:.0f} ms | 100% |")
    L.append("")
    L.append(f"- 이미지당 박스 수: 평균 {df['n_box'].mean():.0f}개 (중앙값 {df['n_box'].median():.0f})")
    L.append(f"- rec 1박스당 비용: 약 {df['t_rec_all'].sum()/max(df['n_crop'].sum(),1)*1000:.1f} ms")
    L.append("")

    L.append("## 절감 수단 1 — 축소 디코딩 (IMREAD_REDUCED_COLOR_N)\n")
    L.append("| 방식 | 평균 | 중앙값 |")
    L.append("| --- | ---: | ---: |")
    L.append(f"| 전체 디코딩 | {df['t_decode_full'].mean()*1000:.0f} ms | {df['t_decode_full'].median()*1000:.0f} ms |")
    L.append(f"| 축소 디코딩 | {df['t_decode_reduced'].mean()*1000:.0f} ms | {df['t_decode_reduced'].median()*1000:.0f} ms |")
    save = df["t_decode_full"].mean() - df["t_decode_reduced"].mean()
    L.append("")
    L.append(f"> 절감 **{save*1000:.0f} ms/장** ({save/tot.mean()*100:.1f}% of 1회 통과). "
             "9MP 이상 고해상도 이미지에서 특히 크다.")
    L.append("")
    hi = df[df["mp"] > 9]
    if len(hi):
        L.append(f"- 9MP 초과 {len(hi)}장 기준: 전체 {hi['t_decode_full'].mean()*1000:.0f} ms → "
                 f"축소 {hi['t_decode_reduced'].mean()*1000:.0f} ms")
    L.append("")

    L.append("## 절감 수단 2 — 적응형 회전\n")
    p0 = df["found_0"].mean()
    one, two = tot.mean(), tot.mean() * 2
    adaptive = tot.mean() * (1 + (1 - p0))
    L.append(f"- 0° 만으로 날짜를 찾은 비율: **{p0*100:.1f}%**")
    L.append("")
    L.append("| 회전 정책 | 기대 비용/장 | 예산(0.72s) 대비 |")
    L.append("| --- | ---: | ---: |")
    L.append(f"| 0° 만 | {one:.2f} s | {one/0.72*100:.0f}% |")
    L.append(f"| 0°+90° 항상 | {two:.2f} s | {two/0.72*100:.0f}% |")
    L.append(f"| **0° → 실패시만 90°** | **{adaptive:.2f} s** | {adaptive/0.72*100:.0f}% |")
    L.append("")
    L.append(f"> 적응형은 항상 2회 대비 **{(two-adaptive)/two*100:.0f}% 절감**하면서 "
             "합집합 recall 을 그대로 유지한다 (0°에서 찾으면 어차피 90°는 볼 필요가 없다).")
    L.append("")
    L.append("> ⚠️ 단, 위 수치는 **단일 프로세스** 기준이다. 4 vCPU 에 4개 프로세스를 "
             "병렬로 돌리면 이론상 1/4 이지만 메모리 대역폭 경합으로 실제로는 그보다 못하다.")

    with open(os.path.join(OUT_DIR, a.out), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L[:40]))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
