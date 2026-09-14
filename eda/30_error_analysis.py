"""
EDA 30단계 — 오답 유형 분해 (정답 137장)

부분점수 체계에서는 '완전히 틀린 것'과 '연도 한 자리만 틀린 것'의 손실이 다르다.
어디를 고쳐야 점수가 오르는지 보려면 유형별로 갈라야 한다.
"""
import os
import sys
import json
import argparse
from collections import Counter, defaultdict

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "eda", "out")


def split3(v):
    if v is None or v == "NONE":
        return ("NONE", "NONE", "NONE")
    p = str(v).split("-")
    return tuple(p) if len(p) == 3 else ("NONE", "NONE", "NONE")


def classify(g, p):
    """정답 g=(y,m,d), 출력 p=(y,m,d) → 오답 유형"""
    if g == p:
        return "정답"
    gy, gm, gd = g
    py, pm, pd_ = p
    g_none = g == ("NONE", "NONE", "NONE")
    p_none = p == ("NONE", "NONE", "NONE")
    if g_none and not p_none:
        return "F. 정답NONE인데 날짜출력"
    if p_none and not g_none:
        return "A. 미검출 (NONE 출력)"
    # 부분 정답 케이스
    if "NONE" in g and g != p:
        return "E. 부분정답 케이스 불일치"
    same_md = (pm, pd_) == (gm, gd)
    same_y = py == gy
    if same_md and not same_y:
        return "B. 연도만 틀림"
    if same_y and not same_md:
        return "C. 월/일만 틀림"
    return "D. 완전히 다른 날짜"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default=r"C:\Users\Public\itda_eval\submission_v4.csv")
    ap.add_argument("--truth", default=os.path.join(OUT_DIR, "truth_v4.json"))
    a = ap.parse_args()

    t = json.load(open(a.truth, encoding="utf-8"))
    labels = t["labels"]
    mem = t["membership"]
    setof = {}
    for k, v in mem.items():
        for f in v:
            setof[f] = k

    meta = pd.read_csv(os.path.join(OUT_DIR, "metadata.csv")).set_index("file")

    def cluster(f):
        if f not in meta.index:
            return "?"
        r = meta.loc[f]
        if r["w"] == 640 and r["h"] == 640:
            return "sq640"
        return "hires" if r["mp"] > 5 else "mid"

    sub = pd.read_csv(a.sub, dtype=str).fillna("NONE")
    pred = {r["image_id"]: (r["year"], r["month"], r["day"]) for _, r in sub.iterrows()}

    rows = []
    for f, gv in labels.items():
        iid = os.path.splitext(f)[0]
        if iid not in pred:
            continue
        g, p = split3(gv), pred[iid]
        rows.append({"file": f, "set": setof.get(f, "?"), "cluster": cluster(f),
                     "truth": gv, "pred": "-".join(p), "cat": classify(g, p),
                     "y_ok": g[0] == p[0], "m_ok": g[1] == p[1], "d_ok": g[2] == p[2]})
    df = pd.DataFrame(rows)
    n = len(df)
    df.to_csv(os.path.join(OUT_DIR, "error_analysis.csv"), index=False, encoding="utf-8-sig")

    L = ["# EDA 30 — 오답 유형 분해\n",
         f"- 정답 **{n}장** 대조. 부분점수 = year/month/day 각각 독립 채점",
         "",
         "## 1. 유형별 분포\n",
         "| 유형 | 장수 | 비율 | 잃는 필드 | 고칠 여지 |",
         "| --- | ---: | ---: | ---: | --- |"]
    lost = defaultdict(int)
    for _, r in df.iterrows():
        lost[r["cat"]] += (not r["y_ok"]) + (not r["m_ok"]) + (not r["d_ok"])
    fix = {
        "정답": "—",
        "A. 미검출 (NONE 출력)": "OCR recall — 가장 크고 가장 어렵다",
        "B. 연도만 틀림": "**연도 오독. 2자리 연도 규칙으로 고칠 여지 큼**",
        "C. 월/일만 틀림": "인식 오류",
        "D. 완전히 다른 날짜": "선택 규칙 (제조일 선택 등)",
        "E. 부분정답 케이스 불일치": "연·월만 / 월·일만 처리",
        "F. 정답NONE인데 날짜출력": "오탐 — 제조일만 있는 경우 등",
    }
    for cat, c in df["cat"].value_counts().items():
        L.append(f"| {cat} | {c} | {c/n*100:.1f}% | {lost[cat]} | {fix.get(cat,'')} |")
    tot_lost = sum(lost.values())
    L.append(f"| **합계** | {n} | 100% | **{tot_lost}** / {n*3} | |")
    L.append("")
    L.append(f"> 필드 단위 점수: **{(n*3-tot_lost)/(n*3)*100:.1f}%**")
    L.append("")

    L.append("## 2. 표본 집합별\n")
    L.append("| 집합 | 장수 | 부분점수 | 정답률 |")
    L.append("| --- | ---: | ---: | ---: |")
    for s, g in df.groupby("set"):
        f3 = (g["y_ok"].sum() + g["m_ok"].sum() + g["d_ok"].sum()) / (len(g) * 3)
        L.append(f"| {s} | {len(g)} | **{f3*100:.1f}%** | {(g['cat']=='정답').mean()*100:.1f}% |")
    L.append("")
    L.append("> `eval_random` 이 대표성 있는 수치다. `active` 는 어려운 것만 모아놓아 "
             "실제보다 나쁘게 나오고, `v2_initial` 은 고해상도 편향이라 좋게 나온다.")
    L.append("")

    L.append("## 3. 클러스터별\n")
    L.append("| 클러스터 | 장수 | 부분점수 | 미검출률 |")
    L.append("| --- | ---: | ---: | ---: |")
    for c, g in df.groupby("cluster"):
        f3 = (g["y_ok"].sum() + g["m_ok"].sum() + g["d_ok"].sum()) / (len(g) * 3)
        miss = (g["cat"] == "A. 미검출 (NONE 출력)").mean()
        L.append(f"| {c} | {len(g)} | **{f3*100:.1f}%** | {miss*100:.1f}% |")
    L.append("")

    L.append("## 4. 연도만 틀린 사례 (가장 고치기 쉬운 유형)\n")
    b = df[df["cat"] == "B. 연도만 틀림"]
    if len(b):
        L.append("| 파일 | 정답 | 출력 | 연도 차이 |")
        L.append("| --- | --- | --- | ---: |")
        for _, r in b.iterrows():
            gy, py = int(split3(r["truth"])[0]), int(r["pred"].split("-")[0])
            L.append(f"| `{r['file']}` | {r['truth']} | {r['pred']} | {py-gy:+d} |")
        diffs = Counter(int(r["pred"].split("-")[0]) - int(split3(r["truth"])[0])
                        for _, r in b.iterrows())
        L.append("")
        L.append(f"- 연도 차이 분포: {dict(sorted(diffs.items()))}")
        L.append("")
        L.append("> 차이가 −1, −5 처럼 특정 값에 몰리면 2자리 연도를 잘못 읽은 것이다 "
                 "(예: `27` 을 `21` 로). 4자리 연도 우선 규칙을 더 강하게 걸 여지가 있다.")
    L.append("")

    L.append("## 5. 완전히 다른 날짜 (선택 규칙 실패)\n")
    d = df[df["cat"] == "D. 완전히 다른 날짜"]
    if len(d):
        L.append("| 파일 | 정답 | 출력 |")
        L.append("| --- | --- | --- |")
        for _, r in d.head(15).iterrows():
            L.append(f"| `{r['file']}` | {r['truth']} | {r['pred']} |")
    L.append("")

    L.append("## 6. 오탐 (정답 NONE인데 날짜 출력)\n")
    f6 = df[df["cat"] == "F. 정답NONE인데 날짜출력"]
    if len(f6):
        for _, r in f6.iterrows():
            L.append(f"- `{r['file']}` → {r['pred']}")
    else:
        L.append("없음")

    with open(os.path.join(OUT_DIR, "30_error_analysis.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
