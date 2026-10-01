"""
EDA 95단계 — 홀드아웃4 라벨을 평가용 파일로 변환

94단계가 뽑은 150장을 이미지를 직접 보고 라벨링한 결과를 평가가 읽는
형식으로 바꾼다. 84단계(홀드아웃3)와 같은 검증을 한다.

  · 94단계가 뽑은 150장과 키가 정확히 일치하는가
  · 원본 폴더에 그 이름이 실제로 있는가
      ⚠️ 확장자 불일치(.jpg ↔ .jpeg)로 16장이 조용히 버려진 적이 두 번 있다.
         imread 가 None 을 내면 그 장이 통째로 미검출로 세어져 수치가 나빠진다.
  · 날짜 형식이 YYYY-MM-DD 또는 필드별 NONE 인가

SKIP 은 제외한다. 중의적이거나 판독 불가인 것을 억지로 채우면 그게 오염이다.

    python eda/95_build_truth4.py
"""
import os
import re
import sys
import json

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "eda", "out")
IMG_DIR = os.path.join(ROOT, "images", "상품사진입니다")

SRC = os.path.join(OUT, "holdout4_labels.jsonl")
DST = os.path.join(OUT, "truth_holdout4.json")
SPLIT = os.path.join(OUT, "94_holdout4_split.json")

RX = re.compile(r"^(\d{4}|NONE)-(\d{2}|NONE)-(\d{2}|NONE)$")


def main():
    rows = []
    with open(SRC, encoding="utf-8-sig") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))

    want = json.load(open(SPLIT, encoding="utf-8"))["files"]
    got = [r["f"] for r in rows]

    bad = []
    if len(got) != len(set(got)):
        from collections import Counter
        bad.append(f"중복 라벨: {[k for k, v in Counter(got).items() if v > 1]}")
    miss = [f for f in want if f not in set(got)]
    extra = [f for f in got if f not in set(want)]
    if miss:
        bad.append(f"라벨이 없는 과제 {len(miss)}장: {miss[:10]}")
    if extra:
        bad.append(f"과제에 없는 라벨 {len(extra)}장: {extra[:10]}")

    labels, skipped, notes = {}, {}, {}
    for r in rows:
        f, d = r["f"], r["d"]
        if r.get("note"):
            notes[f] = r["note"]
        if d == "SKIP":
            skipped[f] = r.get("why", "")
            continue
        if not RX.match(d):
            bad.append(f"형식 오류 {f}: {d!r}")
            continue
        labels[f] = d

    gone = [f for f in labels if not os.path.exists(os.path.join(IMG_DIR, f))]
    if gone:
        bad.append(f"원본 폴더에 없는 파일 {len(gone)}장: {gone[:10]}")

    if bad:
        print("검증 실패:")
        for b in bad:
            print("  -", b)
        return

    json.dump({
        "_출처": "이미지를 직접 보고 판독. 소비기한/제조일자 병기 시 늦은 쪽, "
                "일부만 판독되면 그 필드만 채우고 나머지는 NONE, 중의적이면 제외.",
        "_제외": skipped,
        "_판단근거": notes,
        "labels": labels,
    }, open(DST, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    part = [f for f, v in labels.items() if "NONE" in v]
    print(f"라벨 {len(labels)}장 (제외 {len(skipped)}장)")
    print(f"  부분 라벨 {len(part)}장: {part}")
    print(f"  판단 근거를 적어 둔 것 {len(notes)}장")
    print(f"→ {DST}")


if __name__ == "__main__":
    main()
