"""
EDA 6단계 — 2단계(검출 → 원본해상도 재인식) 구조 검증

가설: 실패 원인은 '검출 실패'가 아니라 '축소본에서의 인식 오류'다.
  근거 (05단계): 001201 은 날짜 박스를 정확히 잡고도 '20221?16' 로 읽었고,
                003341 은 '29/SEP/2022' 를 '2948P/2022' 로 읽었다.
                해상도·임계값을 아무리 올려도 recall 이 5/10 에서 움직이지 않았다.

방법:
  1단계  축소본(960px)에서 det 만 수행 → 박스 좌표 확보 (싸다)
  2단계  숫자가 있을 법한 박스만 골라, **원본 해상도**에서 잘라내 rec 재수행 (박스 몇 개뿐)

비용: rec 대상이 이미지당 30~70개 → 2~6개로 줄어든다.
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
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "eda"))
from datex import find_dates

IMG_DIR = os.path.join(ROOT, "images", "상품사진입니다")
OUT_DIR = os.path.join(ROOT, "eda", "out")
MODELS = r"C:\Users\Public\ocr_models"

REC_H = 48          # PP-OCRv4 rec 입력 높이
PAD = 0.12          # 크롭 여유 (박스가 숫자를 살짝 자르는 경우 대비)


def build(det_side=960):
    from paddleocr import PaddleOCR
    return PaddleOCR(
        use_angle_cls=False, lang="korean", show_log=False, use_gpu=False,
        det_limit_side_len=det_side, det_limit_type="max",
        det_model_dir=os.path.join(MODELS, "det", "ml", "Multilingual_PP-OCRv3_det_infer"),
        rec_model_dir=os.path.join(MODELS, "rec", "korean", "korean_PP-OCRv4_rec_infer"),
        cls_model_dir=os.path.join(MODELS, "cls", "ch_ppocr_mobile_v2.0_cls_infer"),
    )


def crop_quad(img, quad, pad=PAD):
    """4점 박스를 원근 보정해 잘라낸다 (PaddleOCR get_rotate_crop_image 와 동일 개념)."""
    q = np.array(quad, dtype=np.float32)
    w = int(max(np.linalg.norm(q[0] - q[1]), np.linalg.norm(q[2] - q[3])))
    h = int(max(np.linalg.norm(q[0] - q[3]), np.linalg.norm(q[1] - q[2])))
    if w < 4 or h < 4:
        return None
    # 여유 확장
    c = q.mean(axis=0)
    q = c + (q - c) * (1.0 + pad)
    w, h = int(w * (1 + pad)), int(h * (1 + pad))
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(q, dst)
    out = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    # 세로로 긴 박스는 눕힌다
    if h > w * 1.5:
        out = cv2.rotate(out, cv2.ROTATE_90_CLOCKWISE)
    return out


def fit_rec_height(crop, target=REC_H, max_w=1600):
    """rec 입력 높이에 맞춰 스케일. 원본이 크면 축소, 작으면 확대(보간)."""
    h, w = crop.shape[:2]
    if h < 1:
        return None
    s = target / h
    nw = max(8, min(max_w, int(w * s)))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    return cv2.resize(crop, (nw, target), interpolation=interp)


def digit_like(txt):
    """1단계 저해상도 인식 결과로 '날짜일 가능성'을 싸게 판정."""
    if not txt:
        return False
    d = sum(c.isdigit() for c in txt)
    if d < 3:
        return False
    # 순수 한글 문장은 제외
    han = sum("\uac00" <= c <= "\ud7a3" for c in txt)
    return d >= 3 and d >= han


def run_baseline(ocr, img):
    """기존 방식: 축소본 통째로 det+rec."""
    t = time.time()
    res = ocr.ocr(img, cls=False)
    el = time.time() - t
    lines = res[0] if res and res[0] else []
    txt = "\n".join(l[1][0] for l in lines)
    return txt, el, len(lines), lines


def run_twostage(ocr, small, orig, scale, lines_small):
    """2단계: 1단계 박스 중 숫자 후보만 원본에서 다시 인식."""
    t = time.time()
    crops = []
    for ln in lines_small:
        if not digit_like(ln[1][0]):
            continue
        quad = np.array(ln[0], dtype=np.float32) / scale     # 원본 좌표로 환원
        c = crop_quad(orig, quad)
        if c is None:
            continue
        c = fit_rec_height(c)
        if c is not None:
            crops.append(c)
    texts = []
    if crops:
        rec = ocr.text_recognizer(crops)
        rec = rec[0] if isinstance(rec, tuple) else rec
        texts = [r[0] for r in rec]
    el = time.time() - t
    return "\n".join(texts), el, len(crops)


def dates_of(txt):
    return {"%04d-%02d-%02d" % (d["y"], d["m"], d["d"])
            for d in find_dates(txt)
            if not d["in_long_digits"] and not d["day_missing"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", default=os.path.join(OUT_DIR, "truth.json"))
    ap.add_argument("--rot", default="0,90")
    ap.add_argument("--det-side", type=int, default=960)
    ap.add_argument("--out", default="06_twostage.md")
    a = ap.parse_args()

    truth = json.load(open(a.truth, encoding="utf-8"))
    rots = [int(x) for x in a.rot.split(",")]
    ocr = build(a.det_side)

    ROT = {0: None, 90: cv2.ROTATE_90_CLOCKWISE,
           180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}

    rows = []
    for f, gt in truth.items():
        orig0 = cv2.imdecode(np.fromfile(os.path.join(IMG_DIR, f), np.uint8), cv2.IMREAD_COLOR)
        base_txt, two_txt = [], []
        t_base = t_two = 0.0
        nbox = ncrop = 0
        for r in rots:
            orig = orig0 if r == 0 else cv2.rotate(orig0, ROT[r])
            h, w = orig.shape[:2]
            scale = min(1.0, a.det_side / max(h, w))
            small = orig if scale >= 1.0 else cv2.resize(
                orig, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            txt, el, nb, lines = run_baseline(ocr, small)
            base_txt.append(txt); t_base += el; nbox += nb
            txt2, el2, nc = run_twostage(ocr, small, orig, scale, lines)
            two_txt.append(txt2); t_two += el2; ncrop += nc
        b = dates_of("\n".join(base_txt))
        s = dates_of("\n".join(base_txt) + "\n" + "\n".join(two_txt))
        rows.append({"file": f, "gt": gt, "base_hit": gt in b, "two_hit": gt in s,
                     "base": sorted(b), "two": sorted(s - b),
                     "t_base": t_base, "t_two": t_two,
                     "nbox": nbox, "ncrop": ncrop})
        mark = "O" if gt in s else "X"
        gain = " ★복구" if (gt in s and gt not in b) else ""
        print(f"{mark} {f:14s} 정답={gt} 기존={sorted(b)} 추가={sorted(s-b)}"
              f"  박스{nbox}→크롭{ncrop}  {t_base:.2f}s+{t_two:.2f}s{gain}", flush=True)

    n = len(rows)
    bh = sum(r["base_hit"] for r in rows)
    th = sum(r["two_hit"] for r in rows)
    tb = sum(r["t_base"] for r in rows) / n
    tt = sum(r["t_two"] for r in rows) / n
    nb = sum(r["nbox"] for r in rows) / n
    nc = sum(r["ncrop"] for r in rows) / n

    L = ["# EDA 6 — 2단계(검출 → 원본해상도 재인식) 검증\n",
         f"- 정답 집합 {n}장, 회전 {rots} 합집합, det 입력 {a.det_side}px\n",
         "| 방식 | recall | 초/장 | rec 대상 |",
         "| --- | ---: | ---: | ---: |",
         f"| 기존 (축소본 통째 det+rec) | **{bh}/{n} ({bh/n*100:.0f}%)** | {tb:.2f} | {nb:.0f}개 |",
         f"| + 2단계 재인식 | **{th}/{n} ({th/n*100:.0f}%)** | {tb+tt:.2f} | +{nc:.0f}개 |",
         "",
         f"- 2단계 추가 비용: **{tt:.2f}초/장** (rec 대상이 {nb:.0f}개 중 {nc:.0f}개로 걸러짐)",
         f"- 복구된 이미지: **{th-bh}장**",
         ""]
    L.append("| 파일 | 정답 | 기존 후보 | 2단계가 추가한 후보 | 결과 |")
    L.append("| --- | --- | --- | --- | --- |")
    for r in rows:
        res = "★ 복구" if (r["two_hit"] and not r["base_hit"]) else ("O" if r["two_hit"] else "X")
        L.append(f"| `{r['file']}` | {r['gt']} | {r['base'] or '-'} | {r['two'] or '-'} | {res} |")
    with open(os.path.join(OUT_DIR, a.out), "w", encoding="utf-8") as fo:
        fo.write("\n".join(L))
    print(f"\n기존 {bh}/{n} → 2단계 {th}/{n}   |   {tb:.2f}s → {tb+tt:.2f}s/장")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
