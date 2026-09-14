"""
EDA 18단계 — 박스 필터를 '파일 없이 배포 가능한' 형태로

08단계는 HistGradientBoosting 을 썼다. 성능은 좋지만 제출 시 모델 파일을
따로 배포해야 하고 sklearn 런타임 의존이 생긴다.

로지스틱 회귀면 계수 15개뿐이라 노트북에 **숫자로 박아 넣을 수 있고**,
추론은 numpy 내적 한 번이면 된다. 성능 손실이 얼마인지 재고,
허용 범위면 그쪽을 택한다.
"""
import os
import sys
import json

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "eda"))
from datex import find_dates

OUT_DIR = os.path.join(ROOT, "eda", "out")

FEATS = ["aspect", "h_norm", "fill", "n_cc", "cc_per_w", "cc_ar_mean", "cc_ar_std",
         "cc_fill_mean", "cc_h_ratio", "cc_h_std", "gap_ratio", "stroke", "edge_dens",
         "sat_mean", "val_std"]


def dates_of(txt):
    return {"%04d-%02d-%02d" % (d["y"], d["m"], d["d"])
            for d in find_dates(txt or "")
            if not d["in_long_digits"] and not d["day_missing"]}


def image_preservation(df, score, keeps):
    """유지비율별 '이미지의 날짜가 살아남는 비율'."""
    full = df.groupby("file")["text"].apply(lambda s: dates_of("\n".join(s)))
    have = [f for f in full.index if full[f]]
    out = []
    for k in keeps:
        thr = np.quantile(score, 1 - k)
        sel = score >= thr
        kept = df[sel].groupby("file")["text"].apply(lambda s: dates_of("\n".join(s)))
        alive = sum(1 for f in have if f in kept.index and kept[f] & full[f])
        out.append(alive / max(len(have), 1))
    return out, len(have)


def main():
    df = pd.read_csv(os.path.join(OUT_DIR, "boxfeat.csv"))
    df["text"] = df["text"].fillna("")
    X = df[FEATS].fillna(0).values.astype(np.float64)
    y = df["y"].values
    g = df["file"].values
    print(f"박스 {len(df):,} / 이미지 {df['file'].nunique()} / 양성 {y.sum()}")

    from sklearn.model_selection import GroupKFold
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    # 꼬리가 긴 특징은 로그 변환 (로지스틱은 선형이라 도움이 된다)
    LOG = [FEATS.index(c) for c in ("aspect", "n_cc", "cc_per_w", "gap_ratio", "stroke")]
    Xl = X.copy()
    Xl[:, LOG] = np.log1p(np.clip(Xl[:, LOG], 0, None))

    oof = {"HGB": np.zeros(len(df)), "LR": np.zeros(len(df))}
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        h = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(X[tr], y[tr])
        oof["HGB"][te] = h.predict_proba(X[te])[:, 1]
        l = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000, C=1.0,
                                             class_weight="balanced")).fit(Xl[tr], y[tr])
        oof["LR"][te] = l.predict_proba(Xl[te])[:, 1]

    keeps = [0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1]
    L = ["# EDA 18 — 박스 필터: 트리 vs 로지스틱\n",
         f"- 박스 {len(df):,}개 / 이미지 {df['file'].nunique()}장 / 양성 {int(y.sum())}개",
         "- 이미지 단위 GroupKFold 5-fold out-of-fold 예측으로 비교",
         "",
         "> 로지스틱이면 계수 15개를 노트북에 숫자로 박아 넣을 수 있다 — "
         "모델 파일 배포도, sklearn 런타임 의존도 없어진다.",
         "",
         "## 유지비율별 이미지 날짜 보존율\n",
         "| 유지 비율 | HistGradientBoosting | 로지스틱 회귀 | 차이 |",
         "| ---: | ---: | ---: | ---: |"]
    res = {}
    for name in ("HGB", "LR"):
        res[name], nhave = image_preservation(df, oof[name], keeps)
    for i, k in enumerate(keeps):
        d = (res["LR"][i] - res["HGB"][i]) * 100
        L.append(f"| {k*100:.0f}% | {res['HGB'][i]*100:.1f}% | {res['LR'][i]*100:.1f}% | {d:+.1f}%p |")
    L.append("")
    L.append(f"- 기준: 필터 없이 날짜가 잡히는 이미지 {nhave}장")
    L.append("")

    i30 = keeps.index(0.3)
    gap = (res["HGB"][i30] - res["LR"][i30]) * 100
    if gap <= 2.0:
        L.append(f"> ✅ 유지비율 30% 지점에서 차이가 **{gap:+.1f}%p** 에 불과하다. "
                 "로지스틱을 택해 모델 파일 배포를 없앤다.")
    else:
        L.append(f"> ⚠️ 30% 지점에서 로지스틱이 {gap:.1f}%p 열세다. "
                 "트리 모델을 파일로 배포하는 편이 낫다.")
    L.append("")

    # 전체 데이터로 최종 학습 → 계수 추출
    final = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000, C=1.0,
                                             class_weight="balanced")).fit(Xl, y)
    sc, lr = final.steps[0][1], final.steps[1][1]
    # 표준화를 계수에 흡수: z = (x-mu)/sd → w·z + b = (w/sd)·x + (b - w·mu/sd)
    w = lr.coef_[0] / sc.scale_
    b = float(lr.intercept_[0] - np.dot(lr.coef_[0], sc.mean_ / sc.scale_))
    coef = {"feats": FEATS, "log_idx": LOG, "w": [float(v) for v in w], "b": b}

    # 임계값도 함께 저장 (학습 데이터 분위수)
    z = Xl @ w + b
    coef["thresholds"] = {str(k): float(np.quantile(z, 1 - k)) for k in keeps}
    with open(os.path.join(OUT_DIR, "boxfilter_lr.json"), "w", encoding="utf-8") as f:
        json.dump(coef, f, ensure_ascii=False, indent=1)

    L.append("## 추출된 계수 (노트북에 그대로 삽입)\n")
    L.append("| 특징 | 가중치 |")
    L.append("| --- | ---: |")
    order = np.argsort(-np.abs(w))
    for i in order:
        L.append(f"| `{FEATS[i]}`{' (log)' if i in LOG else ''} | {w[i]:+.5f} |")
    L.append(f"| _절편_ | {b:+.5f} |")
    L.append("")
    L.append(f"- 유지비율 30% 임계값: `{coef['thresholds']['0.3']:.4f}` (로짓 기준)")
    L.append(f"- 저장: `eda/out/boxfilter_lr.json`")

    with open(os.path.join(OUT_DIR, "18_filter_lr.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
