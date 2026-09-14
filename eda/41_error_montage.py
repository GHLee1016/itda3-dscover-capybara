"""
EDA 41단계 — 오답 이미지를 유형별 몽타주로

holdout2_errors.csv 의 오답을 유형별로 묶어 한 장씩 그린다.
각 칸에 정답과 출력을 함께 박아 무엇을 어떻게 틀렸는지 바로 보이게 한다.
"""
import os
import sys
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "eda", "out")
SRC = os.path.join(ROOT, "label", "imgs")
IMG_DIR = os.path.join(ROOT, "images", "상품사진입니다")
DST = r"C:\Users\Public\itda_err"


def imread(p):
    return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)


def imwrite(p, img, q=92):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    if ok:
        buf.tofile(p)
    return ok


def panel(fname, side, truth, pred):
    stem = os.path.splitext(fname)[0]
    p = os.path.join(SRC, stem + ".jpg")
    img = imread(p) if os.path.exists(p) else imread(os.path.join(IMG_DIR, fname))
    canvas = np.full((side, side, 3), 18, np.uint8)
    if img is not None:
        h, w = img.shape[:2]
        s = (side - 56) / max(h, w)
        img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))),
                         interpolation=cv2.INTER_AREA)
        y0 = 52 + (side - 52 - img.shape[0]) // 2
        x0 = (side - img.shape[1]) // 2
        canvas[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
    cv2.rectangle(canvas, (0, 0), (side - 1, side - 1), (0, 190, 255), 3)
    cv2.rectangle(canvas, (0, 0), (side, 50), (0, 0, 0), -1)
    cv2.putText(canvas, stem, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (0, 220, 255), 2)
    cv2.putText(canvas, "GT %s" % truth, (8, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (120, 255, 120), 1)
    cv2.putText(canvas, "-> %s" % pred, (side // 2 + 10, 43),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (120, 160, 255), 1)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", type=int, default=760)
    ap.add_argument("--per", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(DST, exist_ok=True)

    df = pd.read_csv(os.path.join(OUT_DIR, "holdout2_errors.csv"))
    tag = {"미검출-못읽음": "unread", "미검출-읽긴읽음": "parse",
           "완전히 다름": "wrongpick", "연도만 오독": "yearerr",
           "월/일 오독": "mderr"}
    groups = defaultdict(list)
    for _, r in df.iterrows():
        groups[r["kind"]].append(r)

    made = []
    for kind, rows in groups.items():
        t = tag.get(kind, "etc")
        for i in range(0, len(rows), a.per):
            chunk = rows[i:i + a.per]
            panels = [panel(r["file"], a.side, r["truth"], r["pred"]) for r in chunk]
            while len(panels) < a.per:
                panels.append(np.full((a.side, a.side, 3), 18, np.uint8))
            grid = np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:4])])
            name = "%s_%02d.jpg" % (t, i // a.per + 1)
            imwrite(os.path.join(DST, name), grid)
            made.append((kind, name, [r["file"] for r in chunk]))

    print("유형별 오답 몽타주")
    for kind, rows in groups.items():
        print("  %-16s %d장 -> %s_*.jpg" % (kind, len(rows), tag.get(kind, "etc")))
    print("\n생성 위치:", DST)
    for kind, name, files in made:
        print("  %-22s %s" % (name, ", ".join(os.path.splitext(f)[0] for f in files)))


if __name__ == "__main__":
    main()
