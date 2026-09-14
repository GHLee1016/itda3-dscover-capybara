"""
EDA 5단계 — 검출 파라미터 / 전처리 ablation

정답을 아는 이미지 집합(truth.json)에 대해 설정 격자를 돌려
 recall(정답 날짜가 후보에 포함된 비율) 과 소요시간을 함께 측정한다.

핵심 검증 대상:
  - det_limit_side_len : PaddleOCR 기본 960 이 잉크젯 날짜를 잘라먹는가
  - det_db_box_thresh  : 낮추면 흐린 날짜 박스를 더 잡는가
  - unclip_ratio       : 박스를 키우면 잘린 숫자가 살아나는가
  - 전처리(CLAHE/샤프닝) : 저대비 잉크젯에 효과가 있는가
"""
import os
import sys
import json
import time
import argparse
import warnings
import itertools

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


def build(det_side=960, box_thresh=0.6, unclip=1.5, dilation=False, rec_batch=6):
    from paddleocr import PaddleOCR
    return PaddleOCR(
        use_angle_cls=False, lang="korean", show_log=False, use_gpu=False,
        det_limit_side_len=det_side, det_limit_type="max",
        det_db_box_thresh=box_thresh, det_db_unclip_ratio=unclip,
        use_dilation=dilation, rec_batch_num=rec_batch,
        det_model_dir=os.path.join(MODELS, "det", "ml", "Multilingual_PP-OCRv3_det_infer"),
        rec_model_dir=os.path.join(MODELS, "rec", "korean", "korean_PP-OCRv4_rec_infer"),
        cls_model_dir=os.path.join(MODELS, "cls", "ch_ppocr_mobile_v2.0_cls_infer"),
    )


# ------------------------------------------------------------------ 전처리
def pp_none(a):
    return a


def pp_clahe(a):
    lab = cv2.cvtColor(a, cv2.COLOR_BGR2LAB)
    l, u, v = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, u, v]), cv2.COLOR_LAB2BGR)


def pp_sharpen(a):
    blur = cv2.GaussianBlur(a, (0, 0), 2.0)
    return cv2.addWeighted(a, 1.6, blur, -0.6, 0)


def pp_clahe_sharpen(a):
    return pp_sharpen(pp_clahe(a))


PREPROCS = {"none": pp_none, "clahe": pp_clahe, "sharpen": pp_sharpen,
            "clahe+sharpen": pp_clahe_sharpen}


def load(path, long_side):
    a = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    h, w = a.shape[:2]
    s = long_side / max(h, w)
    if s < 1.0:
        a = cv2.resize(a, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return a


def candidates(ocr, img, rotations):
    got, el, nbox = set(), 0.0, 0
    views = {0: img,
             90: cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
             180: cv2.rotate(img, cv2.ROTATE_180),
             270: cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)}
    for r in rotations:
        t = time.time()
        res = ocr.ocr(views[r], cls=False)
        el += time.time() - t
        lines = res[0] if res and res[0] else []
        nbox += len(lines)
        txt = "\n".join(l[1][0] for l in lines)
        for d in find_dates(txt):
            if not d["in_long_digits"] and not d["day_missing"]:
                got.add("%04d-%02d-%02d" % (d["y"], d["m"], d["d"]))
    return got, el, nbox


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", default=os.path.join(OUT_DIR, "truth.json"))
    ap.add_argument("--rot", default="0,90", help="쉼표구분 회전 각도")
    ap.add_argument("--grid", default="det_side",
                    choices=["det_side", "thresh", "preproc", "full"])
    ap.add_argument("--out", default="05_ablation.md")
    a = ap.parse_args()

    truth = json.load(open(a.truth, encoding="utf-8"))
    rots = [int(x) for x in a.rot.split(",")]
    print(f"truth set: {len(truth)}장, 회전 {rots}", flush=True)

    if a.grid == "det_side":
        configs = [dict(long_side=ls, det_side=ds, preproc="none")
                   for ls, ds in [(960, 960), (1280, 960), (1280, 1280),
                                  (1920, 1920), (2560, 2560), (3200, 3200)]]
    elif a.grid == "thresh":
        # 해상도는 960 으로 고정 (det_side 실험에서 그 이상은 무의미하다고 확인됨)
        configs = [dict(long_side=1280, det_side=960, preproc="none",
                        box_thresh=bt, unclip=uc)
                   for bt in (0.6, 0.45, 0.3, 0.15) for uc in (1.5, 2.0, 2.5)]
    elif a.grid == "preproc":
        configs = [dict(long_side=1280, det_side=960, preproc=p, box_thresh=bt)
                   for p in PREPROCS for bt in (0.6, 0.3)]
    else:
        configs = [dict(long_side=ls, det_side=ls, preproc=p, box_thresh=bt)
                   for ls in (1280, 1920) for p in ("none", "clahe+sharpen")
                   for bt in (0.6, 0.3)]

    rows = []
    for cfg in configs:
        ocr = build(det_side=cfg.get("det_side", 960),
                    box_thresh=cfg.get("box_thresh", 0.6),
                    unclip=cfg.get("unclip", 1.5),
                    dilation=cfg.get("dilation", False))
        fn = PREPROCS[cfg["preproc"]]
        hit, tot_t, tot_box, miss = 0, 0.0, 0, []
        for f, gt in truth.items():
            img = fn(load(os.path.join(IMG_DIR, f), cfg["long_side"]))
            got, el, nbox = candidates(ocr, img, rots)
            tot_t += el
            tot_box += nbox
            if gt in got:
                hit += 1
            else:
                miss.append(f)
        n = len(truth)
        rows.append({**cfg, "recall": hit / n, "hit": hit, "n": n,
                     "sec_per_img": tot_t / n, "box_per_img": tot_box / n,
                     "miss": miss})
        print(f"  {cfg} -> recall {hit}/{n} ({hit/n*100:.0f}%)  "
              f"{tot_t/n:.2f}s/장  박스 {tot_box/n:.0f}개", flush=True)

    L = ["# EDA 5 — 검출 파라미터 / 전처리 ablation\n",
         f"- 정답 집합: **{len(truth)}장** (육안 확인)",
         f"- 회전: {rots} 합집합",
         "- recall = 정답 날짜가 후보 집합에 포함된 비율 (선택 규칙 이전 단계)\n"]
    keys = [k for k in ("long_side", "det_side", "preproc", "box_thresh", "unclip")
            if any(k in r for r in rows)]
    L.append("| " + " | ".join(keys) + " | recall | 초/장 | 박스/장 |")
    L.append("| " + " | ".join(["---"] * len(keys)) + " | ---: | ---: | ---: |")
    for r in rows:
        cells = [str(r.get(k, "-")) for k in keys]
        L.append("| " + " | ".join(cells) +
                 f" | **{r['recall']*100:.0f}%** ({r['hit']}/{r['n']}) | "
                 f"{r['sec_per_img']:.2f} | {r['box_per_img']:.0f} |")
    L.append("")
    best = max(rows, key=lambda r: (r["recall"], -r["sec_per_img"]))
    L.append(f"- 최고 recall 설정: `{ {k: best.get(k) for k in keys} }` "
             f"→ {best['recall']*100:.0f}%, {best['sec_per_img']:.2f}초/장")
    L.append(f"- 이 설정에서도 놓친 이미지: `{', '.join(best['miss']) or '없음'}`")
    with open(os.path.join(OUT_DIR, a.out), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
