"""
EDA 97단계 — 홀드아웃3·4 라벨을 custom_data/labels.csv 에 합친다

`labels.csv` 는 제출물이고 "직접 라벨링한 정답"을 보이는 파일이다. 그런데
383장(튜닝·홀드아웃1·2)만 들어 있고, 본선 라운드에서 만든 홀드아웃3 197장과
홀드아웃4 148장이 빠져 있다. 실제 분량은 728장이다.

합치면서 지키는 것
  · `image_id` 는 확장자를 뺀 이름, `source_file` 은 **실제 파일명**
    (확장자가 어긋나면 이미지와 대조가 안 된다 — 77·93단계에서 두 번 겪었다)
  · `split` 에 holdout3 / holdout4 를 더한다
  · 판독 불가로 뺀 장은 넣지 않는다. 억지로 채우면 그게 오염이다
  · 기존 383행은 건드리지 않는다. 원본은 `.bak` 으로 남긴다

    python eda/97_merge_labels.py
"""
import os
import csv
import sys
import json
import shutil
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(ROOT, "images", "상품사진입니다")
OUT = os.path.join(ROOT, "eda", "out")
CSV = os.path.join(ROOT, "custom_data", "labels.csv")

ADD = [("truth_holdout3.json", "holdout3"),
       ("truth_holdout4.json", "holdout4"),
       ("truth_holdout5.json", "holdout5")]


def main():
    with open(CSV, encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        cols = r.fieldnames
        rows = list(r)
    have = {x["source_file"] for x in rows}
    print(f"기존 {len(rows)}행")

    added = 0
    for fname, split in ADD:
        p = os.path.join(OUT, fname)
        if not os.path.exists(p):
            print(f"  없음, 건너뜀: {fname}")
            continue
        lab = json.load(open(p, encoding="utf-8"))["labels"]
        n = 0
        for src, date in sorted(lab.items()):
            if src in have:
                continue
            if not os.path.exists(os.path.join(IMG_DIR, src)):
                print(f"  원본 없음, 건너뜀: {src}")
                continue
            y, m, d = date.split("-")
            rows.append({
                "image_id": os.path.splitext(src)[0],
                "year": y, "month": m, "day": d,
                "final_date": "NONE" if date == "NONE-NONE-NONE" else date,
                "split": split,
                "source_file": src,
            })
            have.add(src)
            n += 1
        added += n
        print(f"  {split}: {n}행 추가 (라벨 {len(lab)}장)")

    if not added:
        print("추가할 것 없음.")
        return

    shutil.copy2(CSV, CSV + ".bak")
    rows.sort(key=lambda x: x["image_id"])
    with open(CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    print(f"\n합계 {len(rows)}행")
    print("  split:", dict(Counter(x["split"] for x in rows)))
    part = [x for x in rows if "NONE" in x["final_date"]]
    print(f"  부분 라벨 {len(part)}행")
    gone = [x["source_file"] for x in rows
            if not os.path.exists(os.path.join(IMG_DIR, x["source_file"]))]
    print(f"  원본 폴더에 없는 파일: {len(gone)}행 {gone[:5]}")
    print(f"\n원본은 {os.path.basename(CSV)}.bak")


if __name__ == "__main__":
    main()
