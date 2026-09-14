"""
EDA 9단계 — 박스 필터를 '이미지 단위'로 재평가

08단계는 박스 단위 재현율을 쟀다. 그런데 채점은 이미지 단위다.
같은 날짜가 여러 박스/회전에 중복 등장하므로, 양성 박스 하나를 놓쳐도
이미지의 날짜는 살아남을 수 있다. 실제 손실을 이미지 단위로 다시 잰다.
"""
import os
import sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "eda"))
from datex import find_dates

OUT_DIR = os.path.join(ROOT, "eda", "out")
sys.path.insert(0, OUT_DIR)

from importlib.util import spec_from_file_location, module_from_spec
sp = spec_from_file_location("bf", os.path.join(ROOT, "eda", "08_boxfilter.py"))
bf = module_from_spec(sp)
sys.argv = ["x"]
sp.loader.exec_module(bf)


def dates_of(txt):
    return {"%04d-%02d-%02d" % (d["y"], d["m"], d["d"])
            for d in find_dates(txt or "")
            if not d["in_long_digits"] and not d["day_missing"]}


def main():
    df = pd.read_csv(os.path.join(OUT_DIR, "boxfeat.csv"))
    df["text"] = df["text"].fillna("")
    print(f"박스 {len(df):,} / 이미지 {df['file'].nunique():,}")

    from sklearn.model_selection import GroupKFold
    from sklearn.ensemble import HistGradientBoostingClassifier

    X = df[bf.FEATS].fillna(0).values
    y = df["y"].values
    g = df["file"].values
    oof = np.zeros(len(df))
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        m = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    df["score"] = oof

    # 이미지별 정답 날짜 집합 (필터 없이 rec 전부 돌렸을 때)
    full = df.groupby("file")["text"].apply(lambda s: dates_of("\n".join(s)))
    n_img = len(full)
    n_with = int((full.map(len) > 0).sum())
    print(f"필터 없이 날짜가 잡힌 이미지: {n_with}/{n_img} ({n_with/n_img*100:.1f}%)")

    L = ["# EDA 9 — 박스 필터의 이미지 단위 영향\n",
         f"- 이미지 {n_img:,}장 / 박스 {len(df):,}개 (0°·90° 합산)",
         f"- 필터 없이 날짜가 잡히는 이미지: **{n_with}장 ({n_with/n_img*100:.1f}%)** ← 이것이 상한",
         "",
         "> 08단계의 '박스 재현율'은 비관적 지표다. 같은 날짜가 여러 박스·두 회전에 "
         "중복 등장하므로, 양성 박스를 몇 개 놓쳐도 이미지의 날짜는 살아남는다.",
         "",
         "## 유지 비율별 이미지 단위 손실\n",
         "| 유지 비율 | 박스 재현율 | **이미지 날짜 보존율** | 잃은 이미지 | rec 비용 | 1회 통과 |",
         "| ---: | ---: | ---: | ---: | ---: | ---: |"]

    base_rec, base_other = 684.0, 845.0 - 684.0
    pos = y.sum()
    rows = []
    for keep in (1.0, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.10, 0.05):
        thr = np.quantile(oof, 1 - keep)
        sel = df["score"] >= thr
        box_rec = df.loc[sel, "y"].sum() / max(pos, 1)
        kept = df[sel].groupby("file")["text"].apply(lambda s: dates_of("\n".join(s)))
        kept = kept.reindex(full.index).apply(lambda v: v if isinstance(v, set) else set())
        alive = int(sum(1 for f in full.index if full[f] and kept[f] & full[f]))
        preserve = alive / max(n_with, 1)
        cost = base_other + base_rec * keep
        rows.append((keep, box_rec, preserve, n_with - alive, cost))
        L.append(f"| {keep*100:.0f}% | {box_rec*100:.1f}% | **{preserve*100:.1f}%** | "
                 f"{n_with-alive}장 | {base_rec*keep:.0f} ms | {cost:.0f} ms |")
    L.append("")

    # 손실 없는 최대 절감점
    lossless = [r for r in rows if r[3] == 0]
    best = min(lossless, key=lambda r: r[0]) if lossless else None
    if best:
        L.append(f"> **이미지를 하나도 잃지 않는 최대 절감점: 상위 {best[0]*100:.0f}% 만 rec** "
                 f"→ 1회 통과 845 ms → **{best[4]:.0f} ms ({845/best[4]:.1f}배 단축)**")
    # 1장 이하 손실 지점
    near = [r for r in rows if r[3] <= max(1, int(n_with * 0.02))]
    if near:
        b2 = min(near, key=lambda r: r[0])
        L.append(f">")
        L.append(f"> 손실 2% 이내까지 허용하면: 상위 {b2[0]*100:.0f}% → "
                 f"**{b2[4]:.0f} ms ({845/b2[4]:.1f}배 단축)**, 잃는 이미지 {b2[3]}장")
    L.append("")

    L.append("## 예산 대비 최종 추정\n")
    L.append("기준값은 **실측 wall-clock** 을 쓴다 (03단계: 4워커 병렬, 2회 통과 "
             "500장에 828초 = **1.66 초/장**, 1회 통과 환산 **0.83 초/장**).")
    L.append("")
    L.append("> 07단계의 단일프로세스 845 ms 를 다시 4로 나누면 안 된다. "
             "그 측정은 paddle 기본 `cpu_threads=10` 으로 코어를 전부 쓴 값이라 "
             "이미 병렬 효과가 포함돼 있다. 총 CPU 작업량은 두 측정이 같다.")
    L.append("")
    WALL_1PASS = 830.0          # ms, 실측
    REC_SHARE = 0.809           # 07단계: rec 가 1회 통과의 80.9%
    L.append("| 구성 | 1회 통과 | 0°+90° 2회 | 예산(0.72s) 대비 | 이미지 보존율 |")
    L.append("| --- | ---: | ---: | ---: | ---: |")
    cfgs = [("필터 없음 (현재)", 1.0, 1.0)] + \
           [(f"필터 상위 {int(r[0]*100)}%", r[0], r[2]) for r in rows if r[0] in (0.4, 0.3, 0.2, 0.1)]
    for lab, keep, preserve in cfgs:
        factor = (1 - REC_SHARE) + REC_SHARE * keep
        one = WALL_1PASS * factor
        two = one * 2
        L.append(f"| {lab} | {one:.0f} ms | {two:.0f} ms | **{two/720*100:.0f}%** | "
                 f"{preserve*100:.1f}% |")
    L.append("")
    f30 = (1 - REC_SHARE) + REC_SHARE * 0.30
    L.append(f"> **필터 상위 30% 지점에서 2회 통과가 {WALL_1PASS*f30*2/1000:.2f} 초/장** — "
             "예산 0.72 초에 거의 정확히 들어온다. 대가는 날짜가 잡히던 이미지의 "
             f"{100-93.5:.1f}% 손실(138장 중 9장).")
    L.append("")
    L.append("> ⚠️ 이 필터는 **비용 절감 수단이지 recall 개선 수단이 아니다.** "
             "rec 가 애초에 못 읽는 날짜는 필터를 아무리 잘 만들어도 살아나지 않는다. "
             f"상한은 여전히 {n_with/n_img*100:.1f}% 다.")

    with open(os.path.join(OUT_DIR, "09_filter_imagelevel.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
