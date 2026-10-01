"""
EDA 103단계 — 홀드아웃5 표본 뽑기

인식기 칸(rec 한국어 · rec 영문)을 채택할지 판정할 표본이다.

왜 홀드아웃4 를 쓰지 않는가
  홀드아웃4 는 이미 두 번 썼다. 한 번은 최종 판정(91.9%), 한 번은 1600px 칸을
  기각하는 데 썼다. 세 번째로 쓰면 91.9% 라는 수치 자체의 신뢰도가 떨어진다.
  새 칸의 채택 여부는 새 표본에서 가른다.

  ⚠️ 1600px 칸은 580장에서 +0.6%p 로 보였다가 홀드아웃4 에서 **0장**이었다.
     오염된 표본에서 고른 것이 새 데이터에서 사라지는 일을 이미 겪었으므로,
     이번에도 같은 절차를 지킨다.

    python eda/103_make_holdout5.py [--n 100]
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
DEST = os.path.join(ROOT, "custom_data", "tool", "imgs5")

SEED = 20261010          # 80(20260925)·87(20260926)·94(20261003) 와 다른 값
LONG_SIDE = 1400


def used_files():
    used = set()
    with open(os.path.join(ROOT, "custom_data", "labels.csv"),
              encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            used.add(r["source_file"])
    # 판독 불가로 뺀 것도 이미 본 이미지다
    for nm in ("truth_holdout3.json", "truth_holdout4.json"):
        p = os.path.join(OUT, nm)
        if os.path.exists(p):
            d = json.load(open(p, encoding="utf-8"))
            used |= set(d.get("_제외", {}))
    return used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()

    allf = sorted(n for n in os.listdir(IMG_DIR)
                  if os.path.splitext(n)[1].lower() in (".jpg", ".jpeg", ".png"))
    used = used_files()
    pool = [f for f in allf if f not in used]
    print(f"전체 {len(allf)}장 · 이미 쓴 것 {len(used)}장 · 남은 풀 {len(pool)}장")

    rng = random.Random(SEED)
    pick = sorted(rng.sample(pool, min(a.n, len(pool))))
    rng.shuffle(pick)

    os.makedirs(DEST, exist_ok=True)
    import cv2
    import numpy as np
    n = 0
    for f in pick:
        img = cv2.imdecode(np.fromfile(os.path.join(IMG_DIR, f), dtype=np.uint8),
                           cv2.IMREAD_COLOR)
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
            buf.tofile(os.path.join(DEST, f))
            n += 1

    json.dump({"seed": SEED, "n": len(pick), "files": pick},
              open(os.path.join(OUT, "103_holdout5_split.json"), "w",
                   encoding="utf-8"), ensure_ascii=False, indent=1)

    size = sum(os.path.getsize(os.path.join(DEST, f))
               for f in os.listdir(DEST)) / 1024 / 1024
    print(f"{n}장 준비 ({size:.0f}MB) → {DEST}")
    print("앞 12장:", " ".join(pick[:12]))


if __name__ == "__main__":
    main()
