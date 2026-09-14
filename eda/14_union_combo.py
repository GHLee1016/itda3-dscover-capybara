"""
EDA 14단계 — 전처리 조합(합집합) 분석

12단계에서 드러난 사실: 전 설정 합집합 coverage 가 63.0% 인데
단일 최고 설정은 40.0% 다. 즉 **전처리마다 잡는 이미지가 다르다.**
det 가 입력의 작은 변화에 민감하다는 뜻이고, 조합으로 얻을 여지가 크다.

여기서는 이미 수집한 결과로 '어떤 2~3개를 조합하면 가장 이득인가'를
비용과 함께 계산한다. 새 OCR 실행 없음.
"""
import os
import sys
import json
import itertools
import argparse

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "eda", "out")

# 실측 소요시간 (초/장, 0°+90° 2회, 단일스레드).
# 13단계 400장 확정 실행값을 우선 사용하고, 나머지는 10단계 100장 값.
SEC_CONFIRM = {
    "none|both": 7.15, "adapt_bin|det_only": 4.80, "adapt_bin|both": 5.34,
    "illum+clahe|det_only": 5.49, "illum+clahe|both": 5.89,
}
SEC = {
    "none|both": 8.23, "clahe|both": 7.10, "clahe|det_only": 6.50,
    "illum_norm|both": 7.17, "illum_norm|det_only": 6.82,
    "illum+clahe|both": 6.54, "illum+clahe|det_only": 6.32,
    "dot_close3|both": 6.05, "dot_close3|det_only": 5.69,
    "blackhat|both": 5.12, "blackhat|det_only": 4.71,
    "adapt_bin|both": 5.62, "adapt_bin|det_only": 5.11,
}
SEC.update(SEC_CONFIRM)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(OUT_DIR, "preproc_raw.json"))
    ap.add_argument("--out", default="14_union_combo.md")
    a = ap.parse_args()

    data = json.load(open(a.raw, encoding="utf-8"))
    truth = json.load(open(os.path.join(OUT_DIR, "truth.json"), encoding="utf-8"))
    ev = [r for r in data if r["file"] not in truth]
    n = len(ev)
    keys = [k for k in data[0] if k != "file"]

    def cov(ks):
        c = 0
        for r in ev:
            if any(r[k]["dates"] for k in ks if k in r):
                c += 1
        return c / n

    def plaus(ks):
        ok = tot = 0
        for r in ev:
            s = set()
            for k in ks:
                if k in r:
                    s |= set(r[k]["dates"])
            for d in s:
                tot += 1
                ok += 2015 <= int(d[:4]) <= 2035
        return ok / max(tot, 1), tot / n

    base = "none|both"
    bc = cov([base])
    L = ["# EDA 14 — 전처리 조합(합집합) 분석\n",
         f"- 평가 표본 {n}장, 기준선 `{base}` coverage **{bc*100:.1f}%**",
         "- 새 OCR 실행 없음 (10단계 수집 결과 재사용)",
         "",
         "> 12단계에서 전 설정 합집합이 63.0% 였다. 단일 최고가 40.0% 인데 합집합이 그보다 훨씬 높다는 것은 "
         "**전처리마다 잡아내는 이미지가 다르다**는 뜻이다. det 가 입력의 작은 변화에 민감하다.",
         "",
         "## 단일 설정\n",
         "| 설정 | coverage | plausible | 후보/장 | 초/장 |",
         "| --- | ---: | ---: | ---: | ---: |"]
    singles = sorted(keys, key=lambda k: -cov([k]))
    for k in singles:
        p, cpi = plaus([k])
        L.append(f"| `{k}` | {cov([k])*100:.1f}% | {p*100:.1f}% | {cpi:.2f} | {SEC.get(k, float('nan')):.2f} |")
    L.append("")

    L.append("## 2개 조합 — 상위 12\n")
    L.append("| 조합 | coverage | Δ(기준선) | plausible | 초/장 | 시간당 이득 |")
    L.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    combos = []
    for c in itertools.combinations(keys, 2):
        cv = cov(c)
        sec = sum(SEC.get(k, 8.0) for k in c)
        combos.append((c, cv, sec))
    combos.sort(key=lambda t: -t[1])
    for c, cv, sec in combos[:12]:
        p, _ = plaus(c)
        eff = (cv - bc) * 100 / sec
        L.append(f"| `{c[0]}` + `{c[1]}` | **{cv*100:.1f}%** | {(cv-bc)*100:+.1f}%p | "
                 f"{p*100:.1f}% | {sec:.2f} | {eff:.2f} %p/초 |")
    L.append("")

    # 비용 대비 효율 1위
    eff_sorted = sorted(combos, key=lambda t: -((t[1] - bc) * 100 / t[2]))
    c, cv, sec = eff_sorted[0]
    L.append(f"> 비용 대비 효율 1위 조합: `{c[0]}` + `{c[1]}` — "
             f"coverage {cv*100:.1f}% ({(cv-bc)*100:+.1f}%p), {sec:.2f}초/장")
    L.append("")

    L.append("## 3개 조합 — 상위 6\n")
    L.append("| 조합 | coverage | Δ | 초/장 |")
    L.append("| --- | ---: | ---: | ---: |")
    tri = []
    for c in itertools.combinations(keys, 3):
        tri.append((c, cov(c), sum(SEC.get(k, 8.0) for k in c)))
    tri.sort(key=lambda t: -t[1])
    for c, cv, sec in tri[:6]:
        L.append(f"| {' + '.join('`'+k+'`' for k in c)} | **{cv*100:.1f}%** | "
                 f"{(cv-bc)*100:+.1f}%p | {sec:.2f} |")
    L.append("")

    allc = cov(keys)
    L.append(f"- 전 설정({len(keys)}개) 합집합 상한: **{allc*100:.1f}%**, "
             f"비용 {sum(SEC.get(k,8.0) for k in keys):.1f}초/장 (비현실적)")
    L.append("")
    L.append("## 해석\n")
    L.append("조합은 coverage 를 크게 올리지만 비용이 선형으로 늘어난다. "
             "CPU 예산(0.72초/장)을 감안하면 **2개까지가 현실적 상한**이고, "
             "그것도 박스 필터로 rec 비용을 3배 줄인다는 전제에서만 가능하다.")
    L.append("")
    L.append("> ⚠️ 더 근본적인 문제: 합집합이 이렇게 크게 오른다는 것은 "
             "**det 가 불안정하다**는 신호다. 같은 날짜가 전처리에 따라 잡히기도 하고 "
             "안 잡히기도 한다. 안정적인 검출기를 쓰는 것이 조합을 늘리는 것보다 낫다.")

    with open(os.path.join(OUT_DIR, a.out), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
