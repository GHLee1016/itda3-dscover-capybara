"""
EDA 46단계 — YOLO 데이터셋 구축

45단계가 뽑은 박스(237장/238개)를 YOLOv8 형식으로 변환한다.

중요한 설계 결정
  1) 학습 이미지는 **640px 정사각 레터박스**로 통일한다. 추론 때와 같은 전처리라야
     분포가 맞는다.
  2) 회전 90° 에서 찾은 박스는 **회전된 이미지 자체를 학습 샘플로** 넣는다.
     추론에서도 0°/90° 를 둘 다 돌리므로 두 방향 모두 배워야 한다.
  3) 검증 분할은 **2차 홀드아웃 200장 기준으로 자른다.** 홀드아웃 이미지는
     학습에 넣지 않는다 — 그래야 최종 평가가 오염되지 않는다.
"""
import os
import sys
import json
import shutil
import argparse

import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import pipeline as P

IMG_DIR = os.path.join(ROOT, "images", "상품사진입니다")
OUT_DIR = os.path.join(ROOT, "eda", "out")
DS = os.path.join(ROOT, "yolo_ds")
SIDE = 640


def imwrite(path, img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if ok:
        buf.tofile(path)
    return ok


def letterbox(img, side=SIDE):
    """비율 유지 + 회색 패딩. 추론에서도 동일하게 쓴다."""
    h, w = img.shape[:2]
    s = side / max(h, w)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((side, side, 3), 114, np.uint8)
    y0, x0 = (side - nh) // 2, (side - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = r
    return canvas, s, x0, y0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pad", type=float, default=0.15, help="박스 여유 비율")
    a = ap.parse_args()

    data = json.load(open(os.path.join(OUT_DIR, "yolo_boxes.json"), encoding="utf-8"))
    hold2 = set(json.load(open(os.path.join(OUT_DIR, "truth_holdout2.json"),
                               encoding="utf-8"))["labels"])

    if os.path.isdir(DS):
        shutil.rmtree(DS)
    for sp in ("train", "val"):
        os.makedirs(os.path.join(DS, "images", sp), exist_ok=True)
        os.makedirs(os.path.join(DS, "labels", sp), exist_ok=True)

    n = {"train": 0, "val": 0}
    for rec in data:
        f = rec["file"]
        # 홀드아웃2 는 검증으로만 (학습 오염 방지)
        split = "val" if f in hold2 else "train"
        img0 = P.imread(os.path.join(IMG_DIR, f))
        if img0 is None:
            continue
        for i, b in enumerate(rec["boxes"]):
            # 박스가 잡힌 그 조건(해상도·회전)을 그대로 재현
            small = P.downscale(img0, b["side"])
            view = small if b["deg"] == 0 else cv2.rotate(small, cv2.ROTATE_90_CLOCKWISE)
            if view.shape[1] != b["w"] or view.shape[0] != b["h"]:
                continue
            canvas, s, x0, y0 = letterbox(view)

            q = np.asarray(b["quad"], dtype=float)
            xs, ys = q[:, 0] * s + x0, q[:, 1] * s + y0
            cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
            bw, bh = xs.max() - xs.min(), ys.max() - ys.min()
            bw *= (1 + a.pad)
            bh *= (1 + a.pad * 2)      # 세로로 잘리는 사고가 잦아 더 넉넉히
            if bw < 4 or bh < 4:
                continue
            line = "0 %.6f %.6f %.6f %.6f" % (cx / SIDE, cy / SIDE,
                                              min(bw, SIDE) / SIDE,
                                              min(bh, SIDE) / SIDE)
            stem = "%s_%d" % (os.path.splitext(f)[0], i)
            imwrite(os.path.join(DS, "images", split, stem + ".jpg"), canvas)
            with open(os.path.join(DS, "labels", split, stem + ".txt"), "w") as fo:
                fo.write(line + "\n")
            n[split] += 1

    yaml = (
        "# 소비기한 날짜 영역 검출\n"
        "# 학습 데이터는 기존 PP-OCR det 가 성공한 박스에서 약지도로 얻었다.\n"
        "# 홀드아웃2 200장은 val 로만 쓴다 (최종 평가 오염 방지).\n"
        f"path: {DS}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: date\n"
    )
    with open(os.path.join(DS, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(yaml)

    print(f"YOLO 데이터셋 생성 → {DS}")
    print(f"  train {n['train']}장 / val {n['val']}장")
    print(f"  이미지 크기 {SIDE}x{SIDE} 레터박스, 클래스 1개(date)")
    if n["train"] < 120:
        print("\n⚠️ 학습 표본이 매우 적다. 과적합을 각오해야 한다.")


if __name__ == "__main__":
    main()
