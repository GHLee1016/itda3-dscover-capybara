"""
EDA 93단계 — custom_data/labels.csv 의 확장자 16건 바로잡기

77단계에서 `truth_holdout2.json` 의 같은 문제를 고치면서 `labels.csv` 는
빠뜨렸다. `source_file` 이 `.jpg` 인데 실제 파일은 `.jpeg` 또는 `.png` 인
것이 16건 있다.

왜 고쳐야 하나
  · `labels.csv` 는 제출물이다. 채점자가 우리 라벨을 배포 이미지와 대조할 때
    16장이 어긋난다.
  · 이 파일을 읽어 측정하는 코드가 imread 에서 None 을 받으면 그 장을
    통째로 미검출로 세어, 성능이 실제보다 나쁘게 나온다. 77단계에서 겪은 일이다.

`image_id` 는 확장자를 뺀 이름이라 손댈 필요가 없다. `source_file` 만 고친다.
고치기 전 원본은 `.bak` 으로 남긴다.

    python eda/93_fix_labels_csv.py
"""
import os
import csv
import sys
import shutil

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(ROOT, "images", "상품사진입니다")
CSV = os.path.join(ROOT, "custom_data", "labels.csv")


def main():
    have = {}
    for n in os.listdir(IMG_DIR):
        have[os.path.splitext(n)[0]] = n

    with open(CSV, encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        cols = r.fieldnames
        rows = list(r)

    fixed = []
    for row in rows:
        src = row["source_file"]
        if os.path.exists(os.path.join(IMG_DIR, src)):
            continue
        stem = os.path.splitext(src)[0]
        real = have.get(stem)
        if real:
            fixed.append((src, real))
            row["source_file"] = real

    if not fixed:
        print("고칠 것 없음.")
        return

    shutil.copy2(CSV, CSV + ".bak")
    with open(CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    print(f"{len(fixed)}건 수정 (원본은 labels.csv.bak)")
    for a, b in fixed:
        print(f"  {a} → {b}")

    missing = [r["source_file"] for r in rows
               if not os.path.exists(os.path.join(IMG_DIR, r["source_file"]))]
    print(f"\n남은 불일치: {len(missing)}건 {missing[:5]}")


if __name__ == "__main__":
    main()
