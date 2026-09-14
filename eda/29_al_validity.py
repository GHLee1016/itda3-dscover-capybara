"""
EDA 29단계 — 액티브 러닝이 실제로 작동하는가

주장: 위원회(해상도 4종) 불일치가 크면 정보량이 크다 → 그걸 먼저 라벨링하면
      같은 라벨 예산으로 더 많이 배운다.

검증: 라벨링한 이미지에 대해 **위원회 유형 × 파이프라인 정답 여부** 교차표를 만든다.
      불일치 유형(A)에서 오답률이 높아야 주장이 성립한다.
      일치 유형(D)에서 오답이 많이 나오면 신호가 무의미하다는 뜻이다.
"""
import os
import sys
import json
from collections import Counter, defaultdict

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "eda", "out")


def split3(v):
    if v is None or v == "NONE":
        return ("NONE", "NONE", "NONE")
    p = v.split("-")
    return tuple(p) if len(p) == 3 else ("NONE", "NONE", "NONE")


def main():
    t = json.load(open(os.path.join(OUT_DIR, "truth_v3.json"), encoding="utf-8"))
    labels = t["labels"]

    ev = {x["file"]: x for x in json.load(
        open(os.path.join(OUT_DIR, "label_eval_random.json"), encoding="utf-8"))}
    ac = {x["file"]: x for x in json.load(
        open(os.path.join(OUT_DIR, "label_active.json"), encoding="utf-8"))}

    sub = pd.read_csv(r"C:\Users\Public\itda_eval\submission_v3.csv", dtype=str).fillna("NONE")
    pred = {r["image_id"]: (r["year"], r["month"], r["day"]) for _, r in sub.iterrows()}

    L = ["# EDA 29 — 액티브 러닝 유효성 검증\n"]

    # 1. 내가 라벨링한 것이 어느 집합에서 왔나
    from_ev = [f for f in labels if f in ev]
    from_ac = [f for f in labels if f in ac]
    other = [f for f in labels if f not in ev and f not in ac]
    L += ["## 1. 라벨링한 이미지의 출처\n",
          "| 집합 | 선발 | 라벨링 완료 |",
          "| --- | ---: | ---: |",
          f"| 무작위 측정셋 | {len(ev)} | **{len(from_ev)}** |",
          f"| 액티브셋 (불일치 상위) | {len(ac)} | **{len(from_ac)}** |",
          f"| 그 이전 라벨 (v2) | — | {len(other)} |",
          ""]
    if not from_ac:
        L += ["> ⚠️ **액티브셋은 한 장도 라벨링되지 않았다.**",
              "> 지금까지 라벨링한 것은 전부 무작위 측정셋과 그 이전 라벨이다.",
              "> 즉 **액티브 선발의 효과는 아직 검증되지 않았다.**", ""]

    # 2. 위원회 유형 × 정답 여부
    L += ["## 2. 위원회 유형 × 파이프라인 정답 여부\n",
          "액티브 러닝의 전제는 '불일치가 크면 틀릴 가능성이 높다' 이다. "
          "그게 맞는지 라벨이 있는 이미지로 확인한다.\n"]
    tab = defaultdict(lambda: [0, 0])   # cat -> [정답, 오답]
    detail = []
    for f in from_ev + from_ac:
        meta = ev.get(f) or ac.get(f)
        iid = os.path.splitext(f)[0]
        if iid not in pred:
            continue
        g = split3(labels[f])
        ok = pred[iid] == g
        tab[meta["cat"]][0 if ok else 1] += 1
        detail.append((f, meta["cat"], round(meta["disagree"], 2),
                       labels[f], "-".join(pred[iid]), ok))
    if tab:
        L += ["| 위원회 유형 | 정답 | 오답 | 오답률 |",
              "| --- | ---: | ---: | ---: |"]
        for cat in sorted(tab):
            o, x = tab[cat]
            L.append(f"| {cat} | {o} | {x} | {x/(o+x)*100:.0f}% |")
        L.append("")
        L += ["| 파일 | 유형 | 불일치 | 정답 | 출력 | |",
              "| --- | --- | ---: | --- | --- | :-: |"]
        for f, cat, dis, g, p, ok in sorted(detail, key=lambda r: r[1]):
            L.append(f"| `{f}` | {cat} | {dis} | {g} | {p} | {'O' if ok else 'X'} |")
        L.append("")

        # 핵심 판정
        a_rate = (tab["A_날짜불일치"][1] / sum(tab["A_날짜불일치"])
                  if sum(tab["A_날짜불일치"]) else None)
        d_rate = (tab["D_전원일치"][1] / sum(tab["D_전원일치"])
                  if sum(tab["D_전원일치"]) else None)
        L.append("### 판정\n")
        if a_rate is not None and d_rate is not None:
            if a_rate > d_rate:
                L.append(f"> 불일치(A) 오답률 {a_rate*100:.0f}% > 일치(D) {d_rate*100:.0f}% "
                         "— 신호가 **의도대로 작동**한다.")
            else:
                L.append(f"> ⚠️ 불일치(A) 오답률 **{a_rate*100:.0f}%** ≤ 일치(D) "
                         f"**{d_rate*100:.0f}%** — **신호가 작동하지 않는다.**")
                L.append(">")
                L.append("> 위원회가 전원 일치해도 틀리는 경우가 있다는 뜻이다. "
                         "실제로 `002165`(20.10.30부터 / 21.01.29까지)는 4명 전원이 "
                         "'부터' 날짜로 일치하고도 오답이었다. 해상도만 바꾼 위원회는 "
                         "**같은 종류의 실수를 함께 저지르므로** 불일치가 오답의 좋은 "
                         "예측자가 되지 못한다.")
        L.append("")
        L.append(f"> 표본이 {len(detail)}장뿐이라 이 판정 자체도 확정적이지 않다.")
    else:
        L.append("> 대조 가능한 라벨이 없다.")
    L.append("")

    # 3. 그래도 건진 것
    L += ["## 3. 실제로 건진 것\n",
          "선발 방식과 무관하게, 새로 라벨링한 10장에서 규칙 구멍 3개를 찾았다.",
          "",
          "| 사례 | 발견 | 출처 |",
          "| --- | --- | --- |",
          "| `002367` `01-10-2020` | 연도 후행형 미지원 | 무작위 |",
          "| `002311` `제조2020.08.17` | 제조일만 있는데 소비기한으로 출력 | 무작위 |",
          "| `002630` `05.2023` | 연·월만 있는 경우 버림 | 무작위 |",
          "",
          "> 셋 다 **무작위 표본**에서 나왔다. 액티브 선발이 아니어도 "
          "새 이미지를 보는 것 자체로 규칙 구멍이 드러났다는 뜻이다.",
          "> 640×640 웹 수집 클러스터가 미검증이었던 점을 감안하면, "
          "**클러스터 커버리지**가 불일치 점수보다 더 중요한 선발 기준일 수 있다."]

    with open(os.path.join(OUT_DIR, "29_al_validity.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
