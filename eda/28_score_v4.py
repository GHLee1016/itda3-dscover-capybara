"""
EDA 28단계 — truth_v3 기준 채점 (부분점수)

운영진 확정: year / month / day 가 **각각 독립 채점**된다.
따라서 완전일치율이 아니라 **필드 평균**이 실제 점수에 가깝다.

truth_v3 는 부분 정답을 지원한다 ('2023-05-NONE', 'NONE').
"""
import os
import sys
import json
import argparse

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "eda", "out")


def split3(v):
    """final_date 문자열 → (year, month, day) 각 'NONE' 가능."""
    if v is None or v == "NONE":
        return ("NONE", "NONE", "NONE")
    parts = v.split("-")
    if len(parts) != 3:
        return ("NONE", "NONE", "NONE")
    return tuple(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", required=True)
    ap.add_argument("--truth", default=os.path.join(OUT_DIR, "truth_v4.json"))
    ap.add_argument("--subset", default="all", choices=["all", "v2", "al_eval"])
    a = ap.parse_args()

    t = json.load(open(a.truth, encoding="utf-8"))
    labels = t["labels"]
    # 액티브러닝 무작위 측정셋만 따로 보고 싶을 때
    AL_EVAL = {"002367.jpg", "002654.jpg", "002165.jpg", "000451.jpg", "001548.jpg",
               "002826.jpg", "002311.jpg", "000509.jpg", "002630.jpg", "001495.jpg"}
    if a.subset == "al_eval":
        labels = {k: v for k, v in labels.items() if k in AL_EVAL}
    elif a.subset == "v2":
        labels = {k: v for k, v in labels.items() if k not in AL_EVAL}

    df = pd.read_csv(a.sub, dtype=str).fillna("NONE")
    gt = {os.path.splitext(k)[0]: v for k, v in labels.items()}
    rows = df[df["image_id"].isin(gt)]
    n = len(rows)
    if not n:
        print("대조 가능한 행이 없습니다."); return

    fy = fm = fd = exact = 0
    wrong = []
    for _, r in rows.iterrows():
        gy, gm, gd = split3(gt[r["image_id"]])
        fy += r["year"] == gy
        fm += r["month"] == gm
        fd += r["day"] == gd
        got = (r["year"], r["month"], r["day"])
        if got == (gy, gm, gd):
            exact += 1
        else:
            wrong.append((r["image_id"], f"{gy}-{gm}-{gd}", r["final_date"]))

    print(f"정답 집합: {a.subset}   대조 {n}장 / 제출 {len(df)}행")
    print()
    print(f"  ★ 부분점수 (필드 평균)  {(fy+fm+fd)/(3*n)*100:.1f}%   ← 실제 채점 기준")
    print(f"      year   {fy}/{n} = {fy/n*100:.1f}%")
    print(f"      month  {fm}/{n} = {fm/n*100:.1f}%")
    print(f"      day    {fd}/{n} = {fd/n*100:.1f}%")
    print(f"  (참고) 세 필드 모두 일치  {exact}/{n} = {exact/n*100:.1f}%")
    print()
    if wrong:
        print(f"  틀린 {len(wrong)}장:")
        for i, g, p in wrong:
            print(f"    {i}: 정답 {g:<16} 출력 {p}")

    bad = []
    if df["image_id"].duplicated().any():
        bad.append("image_id 중복")
    for c in ("month", "day"):
        if not df[c].map(lambda v: v == "NONE" or len(v) == 2).all():
            bad.append(f"{c} 자릿수")
    if list(df.columns) != ["image_id", "year", "month", "day", "final_date"]:
        bad.append("컬럼 불일치")
    # final_date 일관성
    inc = df.apply(lambda r: r["final_date"] != (
        "NONE" if r["year"] == r["month"] == r["day"] == "NONE"
        else f"{r['year']}-{r['month']}-{r['day']}"), axis=1).sum()
    if inc:
        bad.append(f"final_date 불일치 {inc}행")
    print("\n형식 검증:", "통과" if not bad else "실패 — " + ", ".join(bad))


if __name__ == "__main__":
    main()

