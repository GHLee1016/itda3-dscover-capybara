"""
EDA 94단계 — 홀드아웃4 표본 뽑기

왜 또 만드나
  홀드아웃3(197장)까지 포함해 라벨이 있는 580장을 이번 라운드에서 **전부**
  오답 목록으로 들여다보며 고쳤다(88~91단계). 그래서 94.1% 는 그 표본에 맞춘
  정도가 섞인 값이고 본선 보고서에 쓸 수 없다.
  한 번도 쓰지 않은 2,772장에서 새로 뽑아야 깨끗한 추정이 나온다.

이번에는 표적 표본을 두지 않는다
  홀드아웃3 는 "2자리 연도가 뒤집힐 수 있는 것" 50장을 따로 모았다. 특정 수정의
  효과를 보려던 것인데, 이번에 재 보니 그 수정으로 얻은 것은 크지 않았고
  난이도가 편향된 표본을 관리하는 부담만 남았다. 이번 목적은 **전반 성능의
  정직한 추정** 하나뿐이므로 무작위 표집만 한다.

    python eda/94_make_holdout4.py [--n 150]
"""
import os
import csv
import sys
import json
import random
import argparse

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(ROOT, "images", "상품사진입니다")
OUT = os.path.join(ROOT, "eda", "out")
DEST = os.path.join(ROOT, "custom_data", "tool", "imgs4")

SEED = 20261003          # 80단계(20260925)·87단계(20260926)와 다른 값
LONG_SIDE = 1400         # 내가 읽을 이미지. 원본은 2MB 가 넘어 다루기 무겁다


def used_files():
    """이미 라벨이 있는 것 = 이번 라운드에서 오염된 것."""
    used = set()
    p = os.path.join(ROOT, "custom_data", "labels.csv")
    with open(p, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            used.add(r["source_file"])
    h3 = json.load(open(os.path.join(OUT, "truth_holdout3.json"),
                        encoding="utf-8"))
    used |= set(h3["labels"])
    used |= set(h3.get("_제외", {}))      # 판독 불가로 뺀 것도 이미 봤다
    return used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    a = ap.parse_args()

    allf = sorted(n for n in os.listdir(IMG_DIR)
                  if os.path.splitext(n)[1].lower() in (".jpg", ".jpeg", ".png"))
    used = used_files()
    pool = [f for f in allf if f not in used]
    print(f"전체 {len(allf)}장 · 이미 쓴 것 {len(used)}장 · 남은 풀 {len(pool)}장")

    rng = random.Random(SEED)
    pick = sorted(rng.sample(pool, min(a.n, len(pool))))
    rng.shuffle(pick)                       # 보는 순서는 섞는다

    os.makedirs(DEST, exist_ok=True)
    import cv2
    import numpy as np
    n_copy = 0
    for f in pick:
        src = os.path.join(IMG_DIR, f)
        dst = os.path.join(DEST, f)
        img = cv2.imdecode(np.fromfile(src, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  못 읽음: {f}")
            continue
        h, w = img.shape[:2]
        m = max(h, w)
        if m > LONG_SIDE:
            s = LONG_SIDE / m
            img = cv2.resize(img, (int(w * s), int(h * s)),
                             interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if ok:
            buf.tofile(dst)
            n_copy += 1

    json.dump({"seed": SEED, "n": len(pick), "files": pick},
              open(os.path.join(OUT, "94_holdout4_split.json"), "w",
                   encoding="utf-8"), ensure_ascii=False, indent=1)

    size = sum(os.path.getsize(os.path.join(DEST, f))
               for f in os.listdir(DEST)) / 1024 / 1024
    print(f"{n_copy}장 준비 ({size:.0f}MB) → {DEST}")
    print(f"→ eda/out/94_holdout4_split.json")
    print()
    print("앞 12장:", " ".join(pick[:12]))


if __name__ == "__main__":
    main()
