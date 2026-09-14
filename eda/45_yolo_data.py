"""
EDA 45단계 — YOLO 학습 데이터 생성

문제: 정답 385장은 날짜 '값'만 있고 '위치'가 없다. YOLO 를 학습하려면 박스가 필요하다.

방법: 기존 파이프라인을 돌려 **정답 날짜를 담은 det 박스**를 찾아 그 좌표를 쓴다.
      약지도(weak supervision) 이며, 원본 회전 각도까지 포함해 좌표를 기록한다.

⚠️ 근본적 한계: 이렇게 모은 박스는 **기존 det 가 이미 성공한 케이스**뿐이다.
   정작 YOLO 가 구해줘야 할 실패 케이스는 학습 데이터에 없다.
   그래도 시도할 가치는 있다 — YOLO 는 '날짜 영역의 생김새'(포장지 위에 고립된
   짧은 숫자열)를 배우므로, 범용 텍스트 검출기와 다른 기준으로 일반화할 수 있다.
   확보 박스가 150개 미만이면 학습이 무의미하므로 먼저 수량을 센다.
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
import pipeline as P
import ocrlib

IMG_DIR = os.path.join(ROOT, "images", "상품사진입니다")
OUT_DIR = os.path.join(ROOT, "eda", "out")
DS = os.path.join(ROOT, "yolo_ds")

_OCR = None


def get_ocr():
    global _OCR
    if _OCR is None:
        _OCR = ocrlib.build_ocr(det_side=640, threads=1)
    return _OCR


def digits(s):
    return "".join(c for c in str(s) if c.isdigit())


def patterns_of(truth):
    """정답 날짜가 문자열로 나타날 수 있는 형태들."""
    if truth == "NONE":
        return []
    p = truth.split("-")
    if len(p) != 3:
        return []
    y, m, d = p
    out = []
    if y != "NONE" and m != "NONE" and d != "NONE":
        out += [y + m + d, y[2:] + m + d, d + m + y[2:], d + m + y, m + d + y[2:]]
    elif m != "NONE" and d != "NONE":
        out.append(m + d)
    elif y != "NONE" and m != "NONE":
        out += [y + m, m + y]
    return [x for x in out if x]


def work(args):
    f, truth = args
    ocr = get_ocr()
    img = P.imread(os.path.join(IMG_DIR, f))
    if img is None:
        return None
    pats = patterns_of(truth)
    if not pats:
        return None
    out = {"file": f, "truth": truth, "boxes": []}
    # 여러 해상도·회전에서 시도해 최대한 많이 건진다
    for side in (640, 512, 448):
        for deg in (0, 90):
            small = P.downscale(img, side)
            view = small if deg == 0 else cv2.rotate(small, cv2.ROTATE_90_CLOCKWISE)
            res = ocr.ocr(view, cls=False)
            lines = res[0] if res and res[0] else []
            for ln in lines:
                if any(p in digits(ln[1][0]) for p in pats):
                    q = np.asarray(ln[0], dtype=float).reshape(4, 2)
                    out["boxes"].append({
                        "side": side, "deg": deg,
                        "w": view.shape[1], "h": view.shape[0],
                        "quad": q.tolist(), "text": ln[1][0],
                        "conf": float(ln[1][1]),
                    })
            if out["boxes"]:
                break
        if out["boxes"]:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    lab = {}
    for fn in ("truth_v4.json", "truth_holdout.json", "truth_holdout2.json"):
        p = os.path.join(OUT_DIR, fn)
        if os.path.exists(p):
            lab.update(json.load(open(p, encoding="utf-8"))["labels"])
    lab = {k: v for k, v in lab.items() if v != "NONE"}
    print(f"정답 {len(lab)}장에서 박스 추출 시도", flush=True)

    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rs = [r for r in ex.map(work, list(lab.items()), chunksize=2) if r]
    got = [r for r in rs if r["boxes"]]
    print(f"\n추출 완료 {time.time()-t0:.0f}s")
    print(f"  박스를 찾은 이미지 : {len(got)}/{len(lab)} ({len(got)/len(lab)*100:.0f}%)")
    print(f"  총 박스 수         : {sum(len(r['boxes']) for r in got)}")
    print(f"  해상도별           : {Counter(b['side'] for r in got for b in r['boxes'])}")
    print(f"  회전별             : {Counter(b['deg'] for r in got for b in r['boxes'])}")

    json.dump(got, open(os.path.join(OUT_DIR, "yolo_boxes.json"), "w",
                        encoding="utf-8"), ensure_ascii=False)
    print(f"\n→ eda/out/yolo_boxes.json")
    if len(got) < 150:
        print("\n⚠️ 150장 미만이다. YOLO 학습 데이터로는 부족할 수 있다.")


if __name__ == "__main__":
    main()
