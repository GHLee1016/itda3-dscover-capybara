"""
EDA 12단계 — 전처리 결과의 통계적 유의성 검정

10단계는 n=100 이라 coverage 차이 9%p 가 우연일 수 있다.
같은 이미지에 두 설정을 모두 적용한 **대응표본**이므로 McNemar 검정이 맞다.
(독립표본 비율검정을 쓰면 검정력을 버리게 된다.)

또한 어떤 이미지가 새로 잡히고 어떤 이미지를 잃었는지 목록으로 남겨
다음 실험의 표적을 만든다.
"""
import os
import sys
import json
import argparse
from collections import defaultdict

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "eda", "out")


def mcnemar(b, c):
    """b = A만 성공, c = B만 성공. 이항검정(양측)."""
    n = b + c
    if n == 0:
        return 1.0
    return float(stats.binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(OUT_DIR, "preproc_raw.json"))
    ap.add_argument("--base", default="none|both")
    ap.add_argument("--out", default="12_preproc_signif.md")
    a = ap.parse_args()

    data = json.load(open(a.raw, encoding="utf-8"))
    truth = json.load(open(os.path.join(OUT_DIR, "truth.json"), encoding="utf-8"))
    keys = [k for k in data[0] if k != "file"]
    evalset = [r for r in data if r["file"] not in truth]
    n = len(evalset)

    base = a.base
    rows = []
    for k in keys:
        if k == base:
            continue
        b = c = both = neither = 0
        gained, lost = [], []
        for r in evalset:
            if k not in r or base not in r:
                continue
            A = bool(r[base]["dates"])
            B = bool(r[k]["dates"])
            if A and not B:
                b += 1; lost.append(r["file"])
            elif B and not A:
                c += 1; gained.append(r["file"])
            elif A and B:
                both += 1
            else:
                neither += 1
        p = mcnemar(b, c)
        rows.append({"key": k, "gain": c, "loss": b, "net": c - b, "p": p,
                     "gained": gained, "lost": lost,
                     "cov": (both + c) / max(n, 1)})
    rows.sort(key=lambda r: -r["net"])

    base_cov = sum(1 for r in evalset if r[base]["dates"]) / max(n, 1)
    L = ["# EDA 12 — 전처리 결과의 유의성 검정\n",
         f"- 평가 표본 **{n}장** (육안 정답셋 제외), 기준선 `{base}` coverage **{base_cov*100:.1f}%**",
         "- 같은 이미지에 두 설정을 모두 적용한 **대응표본** → McNemar (이항) 검정",
         "",
         "| 설정 | coverage | 새로 잡음 | 잃음 | 순증 | p |",
         "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        sig = "**" if r["p"] < 0.05 else ""
        L.append(f"| `{r['key']}` | {r['cov']*100:.1f}% | +{r['gain']} | −{r['loss']} | "
                 f"{sig}{r['net']:+d}{sig} | {r['p']:.3f} |")
    L.append("")

    sig = [r for r in rows if r["p"] < 0.05 and r["net"] > 0]
    if sig:
        L.append(f"> **유의미하게 개선된 설정: {len(sig)}개** (p < 0.05)")
        for r in sig:
            L.append(f"> - `{r['key']}` : 순증 {r['net']:+d}장, p={r['p']:.3f}")
    else:
        L.append("> ⚠️ **p < 0.05 를 넘긴 설정이 없다.** n=100 에서는 "
                 "관측된 차이가 우연과 구분되지 않는다. 표본을 늘리기 전에는 "
                 "어떤 전처리도 '효과 있음'으로 채택해선 안 된다.")
        best = rows[0]
        L.append(">")
        L.append(f"> 가장 유망한 후보는 `{best['key']}` (순증 {best['net']:+d}장, "
                 f"p={best['p']:.3f}) — 표본을 늘려 재검정할 가치가 있다.")
    L.append("")

    top = rows[0]
    L.append(f"## `{top['key']}` 상세 (순증 {top['net']:+d})\n")
    L.append(f"- 새로 잡은 이미지 {len(top['gained'])}장: "
             f"`{', '.join(top['gained'][:20])}`")
    L.append(f"- 잃은 이미지 {len(top['lost'])}장: "
             f"`{', '.join(top['lost'][:20]) or '없음'}`")
    L.append("")
    L.append("> 잃은 이미지가 있다는 것은 전처리가 **양날**이라는 뜻이다. "
             "두 설정을 모두 돌려 합집합을 쓰면 손실 없이 이득만 취할 수 있지만 "
             "비용이 2배가 된다 — CPU 예산과 상충한다.")
    L.append("")

    # 합집합 상한
    L.append("## 참고: 전 설정 합집합 상한\n")
    uni = sum(1 for r in evalset if any(r[k]["dates"] for k in keys if k in r))
    L.append(f"- 모든 전처리 설정의 합집합 coverage: **{uni/n*100:.1f}%** "
             f"(기준선 {base_cov*100:.1f}%)")
    L.append(f"- 즉 전처리를 아무리 잘 골라도 이 표본에서 상한은 {uni/n*100:.1f}% 다. "
             "나머지는 인식 모델 자체가 못 읽는 것이다.")

    with open(os.path.join(OUT_DIR, a.out), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
