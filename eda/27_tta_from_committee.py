"""
EDA 27단계 — 위원회 데이터로 다중해상도 TTA 효과 계산

액티브 러닝(EDA 25)에서 900장을 640/512/448/640-0°단독 네 설정으로 이미 돌렸다.
그 결과를 재활용해 "여러 해상도를 합치면 커버리지가 얼마나 오르는가"를 잰다.
**새 OCR 실행 없음.**

운영진 확정으로 장당 예산이 0.72초 → 4.8초로 늘었으므로, 이전에 비용 때문에
포기했던 TTA 가 이제 현실적인 선택지가 됐다.
"""
import os
import sys
import json
import itertools
from collections import Counter

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "eda", "out")

# EDA 23/24 실측 (단일 프로세스, 0°+90° 적응형)
SEC = {"s640": 0.992, "s512": 0.641, "s448": 0.499, "s640_r0": 0.62}


def main():
    df = pd.read_csv(os.path.join(OUT_DIR, "active_pool.csv"))
    names = [c for c in ("s640", "s512", "s448", "s640_r0") if c in df.columns]
    n = len(df)

    def cov(cols):
        """여러 설정 중 하나라도 날짜를 냈으면 커버."""
        return df[list(cols)].notna().any(axis=1).mean()

    def agree(cols):
        """커버된 것 중 설정들이 같은 날짜로 합의한 비율 (정밀도 대용)."""
        sub = df[list(cols)]
        m = sub.notna().any(axis=1)
        if not m.any():
            return 0.0
        same = sub[m].apply(lambda r: len(set(v for v in r if pd.notna(v))) == 1, axis=1)
        return same.mean()

    L = ["# EDA 27 — 다중해상도 TTA (위원회 데이터 재활용)\n",
         f"- 대상 **{n}장** (EDA 25 액티브 풀). 새 OCR 실행 없음",
         "- 커버리지 = 날짜를 하나라도 낸 비율 / 합의율 = 설정들이 같은 날짜를 낸 비율",
         "",
         "> 운영진 확정으로 장당 예산이 0.72초 → **4.8초** 가 됐다.",
         "> 비용 때문에 포기했던 TTA 가 현실적인 선택지가 됐으므로 다시 계산한다.",
         "",
         "## 단일 설정\n",
         "| 설정 | 커버리지 | 초/장 |",
         "| --- | ---: | ---: |"]
    for c in names:
        L.append(f"| `{c}` | {cov([c])*100:.1f}% | {SEC.get(c, float('nan')):.2f} |")
    L.append("")

    L.append("## 조합 (합집합)\n")
    L.append("| 조합 | 커버리지 | Δ(640 대비) | 합의율 | 초/장 | 500장 |")
    L.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    base = cov(["s640"])
    rows = []
    for k in (2, 3, 4):
        for combo in itertools.combinations(names, k):
            c = cov(combo)
            sec = sum(SEC.get(x, 1.0) for x in combo)
            rows.append((combo, c, sec, agree(combo)))
    rows.sort(key=lambda t: -t[1])
    for combo, c, sec, ag in rows[:10]:
        tot = sec * 500
        mark = "✅" if tot < 2400 else "❌"
        L.append(f"| {' + '.join('`'+x+'`' for x in combo)} | **{c*100:.1f}%** | "
                 f"{(c-base)*100:+.1f}%p | {ag*100:.0f}% | {sec:.2f} | {tot:.0f}초 {mark} |")
    L.append("")

    best = rows[0]
    L.append(f"> 최고 커버리지 조합: {' + '.join(best[0])} → **{best[1]*100:.1f}%** "
             f"({(best[1]-base)*100:+.1f}%p), 500장 **{best[2]*500:.0f}초** "
             f"(제한 2,400초의 {best[2]*500/2400*100:.0f}%)")
    L.append("")
    L.append("> ⚠️ 커버리지가 오른다고 정확도가 오르는 건 아니다. EDA 19 에서 "
             "7자리 느슨 패턴이 커버리지를 +12%p 올리고 정확도는 0%p 올린 전례가 있다. "
             "**합의율이 낮은 조합은 서로 다른 날짜를 내놓는다는 뜻**이므로, "
             "후보가 늘어난 만큼 선택 규칙이 틀릴 여지도 커진다.")
    L.append("")

    # 불일치 시 어느 설정이 맞는지 — 사람 정답으로 확인
    truth = json.load(open(os.path.join(OUT_DIR, "truth_v2.json"), encoding="utf-8"))
    tdf = df[df["file"].isin(truth)]
    if len(tdf):
        L.append(f"## 사람 정답과 대조 ({len(tdf)}장)\n")
        L.append("| 설정 | 정확도 |")
        L.append("| --- | ---: |")
        for c in names:
            hit = sum(1 for _, r in tdf.iterrows() if r[c] == truth[r["file"]])
            L.append(f"| `{c}` | {hit}/{len(tdf)} ({hit/len(tdf)*100:.0f}%) |")
        L.append("")
    else:
        L.append("> 액티브 풀에서 사람 정답 이미지는 제외되어 직접 대조는 불가하다. "
                 "TTA 채택 여부는 라벨링 확대 후 판단해야 한다.")

    with open(os.path.join(OUT_DIR, "27_tta_from_committee.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
