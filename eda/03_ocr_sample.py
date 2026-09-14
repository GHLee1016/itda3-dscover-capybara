"""
EDA 3단계 — 표본 OCR 실행

층화 표본에 대해 0° / 90° 두 방향으로 OCR 을 돌려
 - 원문 텍스트 (후속 분석용)
 - 방향별 점수 → 90° 회전이 필요한 이미지 비율
 - 해상도별 CPU 소요시간 (속도 예산 검증)
를 수집해 ocr_sample.jsonl 로 저장한다.
"""
import os
import sys
import json
import time
import argparse
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("FLAGS_use_mkldnn", "1")

import numpy as np
import pandas as pd
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(ROOT, "images", "상품사진입니다")
OUT_DIR = os.path.join(ROOT, "eda", "out")
os.makedirs(OUT_DIR, exist_ok=True)

# Paddle 의 C++ 로더가 한글 경로를 못 열어 모델은 ASCII 경로에 둔다 (로컬 전용 우회)
MODELS = r"C:\Users\Public\ocr_models"

_OCR = None
HANGUL = lambda s: sum("\uac00" <= c <= "\ud7a3" for c in s)


def get_ocr(threads=1):
    global _OCR
    if _OCR is None:
        from paddleocr import PaddleOCR
        _OCR = PaddleOCR(
            use_angle_cls=True, lang="korean", show_log=False, use_gpu=False,
            cpu_threads=threads,
            det_model_dir=os.path.join(MODELS, "det", "ml", "Multilingual_PP-OCRv3_det_infer"),
            rec_model_dir=os.path.join(MODELS, "rec", "korean", "korean_PP-OCRv4_rec_infer"),
            cls_model_dir=os.path.join(MODELS, "cls", "ch_ppocr_mobile_v2.0_cls_infer"),
        )
    return _OCR


def load(path, long_side):
    """cv2.imdecode 는 EXIF orientation 을 자동 적용한다 (검증 완료, EDA2 §3)."""
    a = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    if a is None:
        return None, None
    h, w = a.shape[:2]
    s = long_side / max(h, w)
    if s < 1.0:
        a = cv2.resize(a, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    return a, (w, h)


def run(ocr, img):
    t = time.time()
    res = ocr.ocr(img, cls=True)
    el = time.time() - t
    out = []
    lines = res[0] if res and res[0] else []
    for ln in lines:
        box = np.array(ln[0], dtype=float)
        txt, conf = ln[1]
        out.append({
            "t": txt,
            "c": round(float(conf), 3),
            "box": [round(float(v), 1) for v in box.reshape(-1).tolist()],
        })
    return out, el


def score(lines):
    """방향 판별용 점수: 신뢰도 가중 한글+숫자 글자수."""
    s = 0.0
    for l in lines:
        txt = l["t"]
        useful = HANGUL(txt) + sum(ch.isdigit() for ch in txt)
        s += l["c"] * useful
    return s


def work(args):
    path, long_side, threads = args
    name = os.path.basename(path)
    ocr = get_ocr(threads)
    rec = {"file": name, "image_id": os.path.splitext(name)[0]}
    try:
        img, orig = load(path, long_side)
        if img is None:
            rec["ok"] = False
            return rec
        rec["orig_w"], rec["orig_h"] = orig
        rec["proc_w"], rec["proc_h"] = img.shape[1], img.shape[0]

        l0, t0 = run(ocr, img)
        l90, t90 = run(ocr, cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
        s0, s90 = score(l0), score(l90)

        rec["t_0"], rec["t_90"] = round(t0, 3), round(t90, 3)
        rec["score_0"], rec["score_90"] = round(s0, 1), round(s90, 1)
        rec["n_0"], rec["n_90"] = len(l0), len(l90)
        rec["best"] = 90 if s90 > s0 * 1.15 else 0     # 15% 이상 우세할 때만 회전 판정
        rec["lines_0"] = l0
        rec["lines_90"] = l90
        rec["ok"] = True
    except Exception as e:
        rec["ok"] = False
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400, help="표본 장수 (0=전체)")
    ap.add_argument("--long-side", type=int, default=1280)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=1, help="paddle cpu_threads (프로세스당)")
    ap.add_argument("--out", default="ocr_sample.jsonl")
    a = ap.parse_args()

    meta = pd.read_csv(os.path.join(OUT_DIR, "metadata.csv"))
    if a.n and a.n < len(meta):
        # 해상도 클러스터 × id 구간으로 층화 추출
        cluster = pd.Series(
            np.where((meta["w"] == 640) & (meta["h"] == 640), "sq640",
                     np.where(meta["mp"] > 5, "hires", "mid")),
            index=meta.index, dtype="object")
        meta["strat"] = cluster + "_" + (meta["id_num"] // 500).astype(int).astype(str)
        frac = a.n / len(meta)
        sam = (meta.groupby("strat", group_keys=False)
                   .apply(lambda g: g.sample(max(1, round(len(g) * frac)), random_state=42)))
        sam = sam.sample(min(a.n, len(sam)), random_state=42)
    else:
        sam = meta
    files = [os.path.join(IMG_DIR, f) for f in sam["file"]]
    print(f"OCR on {len(files)} images | long_side={a.long_side} "
          f"workers={a.workers} threads={a.threads}", flush=True)

    outp = os.path.join(OUT_DIR, a.out)
    t0 = time.time()
    done = 0
    with open(outp, "w", encoding="utf-8") as fo:
        if a.workers <= 1:
            for p in files:
                fo.write(json.dumps(work((p, a.long_side, a.threads)), ensure_ascii=False) + "\n")
                done += 1
                if done % 25 == 0:
                    print(f"  {done}/{len(files)}  {time.time()-t0:.0f}s", flush=True)
        else:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=a.workers) as ex:
                for rec in ex.map(work, [(p, a.long_side, a.threads) for p in files], chunksize=4):
                    fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    done += 1
                    if done % 25 == 0:
                        print(f"  {done}/{len(files)}  {time.time()-t0:.0f}s", flush=True)
    el = time.time() - t0
    print(f"\ndone in {el:.0f}s  ({el/max(done,1):.2f}s/img wall, {a.workers} workers)")
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
