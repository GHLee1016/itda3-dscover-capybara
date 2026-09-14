"""
라벨링용 몽타주 생성기.

이미지를 한 장씩 보면 턴이 너무 많이 든다. 2x2 로 묶되, 각 칸을 정사각으로
letterbox 해서 비율을 유지하고 파일명을 크게 박는다.

    python label/montage.py --start 0 --count 4
"""
import os
import sys
import json
import argparse

import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "label", "imgs")
OUT = r"C:\Users\Public\itda_label"      # ASCII 경로 (cv2 호환)
os.makedirs(OUT, exist_ok=True)


def imread(p):
    return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)


def imwrite(p, img, q=94):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    if ok:
        buf.tofile(p)
    return ok


def panel(path, side, tag):
    img = imread(path)
    if img is None:
        return np.full((side, side, 3), 40, np.uint8)
    h, w = img.shape[:2]
    s = side / max(h, w)
    img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))),
                     interpolation=cv2.INTER_AREA)
    canvas = np.full((side, side, 3), 20, np.uint8)
    y0 = (side - img.shape[0]) // 2
    x0 = (side - img.shape[1]) // 2
    canvas[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
    cv2.rectangle(canvas, (0, 0), (side - 1, side - 1), (0, 200, 255), 3)
    cv2.rectangle(canvas, (0, 0), (250, 40), (0, 0, 0), -1)
    cv2.putText(canvas, tag, (8, 29), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0, 220, 255), 2)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", required=True, help="쉼표구분 파일명")
    ap.add_argument("--side", type=int, default=780)
    ap.add_argument("--out", default="montage.jpg")
    a = ap.parse_args()

    files = [f.strip() for f in a.files.split(",") if f.strip()]
    panels = []
    for i, f in enumerate(files):
        stem = os.path.splitext(f)[0]
        p = os.path.join(SRC, stem + ".jpg")
        panels.append(panel(p, a.side, stem))
    while len(panels) < 4:
        panels.append(np.full((a.side, a.side, 3), 20, np.uint8))
    top = np.hstack(panels[:2])
    bot = np.hstack(panels[2:4])
    grid = np.vstack([top, bot])
    dst = os.path.join(OUT, a.out)
    imwrite(dst, grid)
    print(dst, grid.shape)


if __name__ == "__main__":
    main()
