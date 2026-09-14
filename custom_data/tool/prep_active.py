"""
액티브 러닝(EDA 25)이 선발한 이미지들의 라벨링용 축소본을 만든다.

  label_eval_random.json  무작위 45장  → 정확도 측정용 (대표성)
  label_active.json       액티브 60장  → 규칙 개선용 (어려운 것 위주)
"""
import os
import sys
import json

import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(ROOT, "images", "상품사진입니다")
OUT_DIR = os.path.join(ROOT, "eda", "out")
DST = os.path.join(ROOT, "label", "imgs")
LONG = 1500


def save(path, img, q=88):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    if ok:
        buf.tofile(path)          # 한글 경로 대응
    return ok


def main():
    os.makedirs(DST, exist_ok=True)
    made = skipped = 0
    allitems = []
    for fn in ("label_eval_random.json", "label_active.json"):
        items = json.load(open(os.path.join(OUT_DIR, fn), encoding="utf-8"))
        allitems += items
        for it in items:
            dst = os.path.join(DST, os.path.splitext(it["file"])[0] + ".jpg")
            if os.path.exists(dst):
                skipped += 1
                continue
            src = os.path.join(IMG_DIR, it["file"])
            img = cv2.imdecode(np.fromfile(src, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]
            s = LONG / max(h, w)
            if s < 1.0:
                img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            if save(dst, img):
                made += 1
    print(f"축소본 생성 {made}장 (이미 있던 것 {skipped}장), 총 대상 {len(allitems)}장")

    # 라벨링 순서 목록 (콘솔에서 바로 보기 좋게)
    out = []
    for it in allitems:
        out.append({
            "file": it["file"], "kind": it["kind"], "cluster": it["cluster"],
            "size": it["size"], "cat": it["cat"],
            "votes": it["votes"], "cands": it["cands"], "anchors": it["anchors"],
        })
    json.dump(out, open(os.path.join(OUT_DIR, "label_todo.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"→ eda/out/label_todo.json ({len(out)}장)")


if __name__ == "__main__":
    main()
