"""
EDA 49단계 — YOLO 예측 시각화

mAP 로는 "기존 det 가 놓친 날짜 영역을 YOLO 가 찾았는가" 를 알 수 없다.
둘은 다른 질문이다.
  - mAP : 약지도로 만든 정답 박스(= det 가 이미 성공한 곳)를 얼마나 맞추나
  - 우리가 알고 싶은 것 : det 가 실패한 이미지에서 날짜를 짚어내나

그래서 **오답 이미지**에 직접 그려 본다.
  초록 = 기존 PP-OCR det 박스   빨강 = YOLO 박스(신뢰도 표기)
사람이 보면 30초 안에 판단이 끝난다.

결과는 `yolo_시각화/` 에 저장한다.
"""
import os
import sys
import json
import argparse
import warnings

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
VIS = os.path.join(ROOT, "yolo_시각화")
# 추론은 ONNX 로 한다 — torch 를 싣지 않기 위해서다(EDA 52~53).
WEIGHTS = os.path.join(ROOT, "weights", "yolo", "date_v1.onnx")


def imwrite(path, img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if ok:
        buf.tofile(path)      # cv2.imwrite 는 한글 경로에서 조용히 실패한다
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.20)
    ap.add_argument("--limit", type=int, default=40)
    a = ap.parse_args()

    if not os.path.exists(WEIGHTS):
        print(f"가중치 없음: {WEIGHTS}")
        return

    lab = json.load(open(os.path.join(OUT_DIR, "truth_holdout2.json"),
                         encoding="utf-8"))["labels"]
    # 48단계가 남긴 기준선 오답 목록이 있으면 그걸 쓰고, 없으면 전체에서 고른다
    md = os.path.join(OUT_DIR, "39_holdout2_eval.md")
    targets = []
    if os.path.exists(md):
        for line in open(md, encoding="utf-8"):
            if line.startswith("| `") and ".jpg" in line:
                targets.append(line.split("`")[1])
    targets = [f for f in targets if f in lab][:a.limit]
    if not targets:
        targets = list(lab)[:a.limit]
    print(f"오답 {len(targets)}장에 그린다 (conf={a.conf})", flush=True)

    os.makedirs(VIS, exist_ok=True)
    ocr = ocrlib.build_ocr(det_side=640, threads=0)
    pipe = P.DatePipeline(ocr, long_side=640, rotations=(0, 90),
                          adaptive=True, keep=None)
    pipe.yolo = P.OnnxDateDetector(WEIGHTS)
    pipe.yolo_conf = a.conf
    pipe._yolo_on = True

    n_hit = 0
    for f in targets:
        img = P.imread(os.path.join(IMG_DIR, f))
        if img is None:
            continue
        small = P.downscale(img, 640)
        vis = small.copy()

        # 초록 — 기존 det
        res = ocr.ocr(small, cls=False)
        for ln in (res[0] if res and res[0] else []):
            q = np.asarray(ln[0], np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [q], True, (0, 200, 0), 1)

        # 빨강 — YOLO
        boxes = pipe._yolo_boxes(small)
        for q in boxes:
            p = np.asarray(q, np.int32)
            cv2.polylines(vis, [p.reshape(-1, 1, 2)], True, (0, 0, 255), 2)
            c = P.crop_quad(small, q)
            txt = ""
            if c is not None:
                try:
                    rr = pipe._rec([c])
                    rr = rr[0] if isinstance(rr, tuple) else rr
                    txt = "%s %.2f" % (rr[0][0], float(rr[0][1]))
                except Exception:
                    pass
            if txt:
                cv2.putText(vis, txt, (int(p[:, 0].min()), max(14, int(p[:, 1].min()) - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        if boxes:
            n_hit += 1
        cv2.putText(vis, "GT %s  YOLO %d box" % (lab[f], len(boxes)),
                    (6, vis.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 255), 2)
        imwrite(os.path.join(VIS, "[%s]_%s" % (lab[f], f)), vis)

    print(f"\n→ {VIS}")
    print(f"  YOLO 가 박스를 낸 오답 이미지 {n_hit}/{len(targets)}장")
    print("  초록=기존 det, 빨강=YOLO. 빨강이 날짜를 감싸고 있으면 성공이다.")


if __name__ == "__main__":
    main()
