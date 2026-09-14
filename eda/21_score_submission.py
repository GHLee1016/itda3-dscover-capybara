"""
submission.csv 를 사람 정답과 대조해 채점한다.

규정에 정확도 채점 세부(완전일치 vs 필드별 부분점수)가 공개되지 않았으므로
두 가지를 모두 계산한다.
"""
import os
import sys
import json
import argparse

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "eda", "out")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", required=True)
    ap.add_argument("--truth", default=os.path.join(OUT_DIR, "truth_v2.json"))
    a = ap.parse_args()

    truth = json.load(open(a.truth, encoding="utf-8"))
    # image_id 는 문자열로 읽어야 한다 ('000001' 이 1 로 바뀌면 안 된다)
    df = pd.read_csv(a.sub, dtype=str).fillna("NONE")
    gt = {os.path.splitext(f)[0]: v for f, v in truth.items()}

    rows = df[df["image_id"].isin(gt)]
    n = len(rows)
    exact = fy = fm = fd = 0
    wrong, nones = [], []
    for _, r in rows.iterrows():
        g = gt[r["image_id"]]
        gy, gm, gd = g.split("-")
        exact += r["final_date"] == g
        fy += r["year"] == gy
        fm += r["month"] == gm
        fd += r["day"] == gd
        if r["final_date"] == "NONE":
            nones.append(r["image_id"])
        elif r["final_date"] != g:
            wrong.append((r["image_id"], g, r["final_date"]))

    print(f"정답과 대조 가능한 행: {n} / 제출 {len(df)}")
    print()
    print("  [운영진 확정] year/month/day 가 각각 독립 채점된다.")
    print(f"  ★ 부분점수 (필드 평균)   {(fy+fm+fd)/(3*n)*100:.1f}%   ← 실제 점수 기준")
    print(f"    · year   {fy}/{n} = {fy/n*100:.1f}%")
    print(f"    · month  {fm}/{n} = {fm/n*100:.1f}%")
    print(f"    · day    {fd}/{n} = {fd/n*100:.1f}%")
    print(f"  (참고) 완전일치          {exact}/{n} = {exact/n*100:.1f}%")
    print()
    print(f"  NONE 출력  {len(nones)}장")
    print(f"  틀린 날짜  {len(wrong)}장")
    for i, g, p in wrong:
        print(f"    {i}: 정답 {g} → 출력 {p}")

    # 형식 검증
    print()
    bad = []
    if df["image_id"].duplicated().any():
        bad.append("image_id 중복")
    for c in ("month", "day"):
        if not df[c].map(lambda v: v == "NONE" or len(v) == 2).all():
            bad.append(f"{c} 자릿수")
    if list(df.columns) != ["image_id", "year", "month", "day", "final_date"]:
        bad.append(f"컬럼 불일치: {list(df.columns)}")
    print("형식 검증:", "통과" if not bad else "실패 — " + ", ".join(bad))


if __name__ == "__main__":
    main()
