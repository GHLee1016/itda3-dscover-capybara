"""
EDA 33단계 — (1) NONE 비율 점검  (2) 오답 심층 분해

질문: 정답률이 그럴듯해 보이는 게 사실은 'NONE 남발' 덕 아닌가?
      → NONE 을 많이 내면 정답이 NONE 인 것만 맞고 나머지는 다 틀린다.
        부분점수 체계에서 NONE 은 3필드를 통째로 버리는 선택이므로
        오히려 손해다. 실제 비율을 센다.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default=r"C:\Users\Public\itda_eval\submission_v5.csv")
    ap.add_argument("--wrong", default=os.path.join(OUT_DIR, "quick_pairON.csv"))
    a = ap.parse_args()

    t = json.load(open(os.path.join(OUT_DIR, "truth_v4.json"), encoding="utf-8"))
    lab = t["labels"]
    sub = pd.read_csv(a.sub, dtype=str).fillna("NONE")
    meta = pd.read_csv(os.path.join(OUT_DIR, "metadata.csv")).set_index("file")

    L = ["# EDA 33 — NONE 비율 점검 및 오답 심층 분해\n"]

    # ---------------- 1. NONE 비율 ----------------
    allnone = (sub["final_date"] == "NONE").sum()
    L += ["## 1. NONE 남발인가\n",
          "부분점수 체계에서 NONE 은 **3필드를 통째로 버리는 선택**이다. "
          "많이 낼수록 손해이므로 비율을 확인한다.\n",
          "| 대상 | 전체 | NONE 출력 | 비율 |",
          "| --- | ---: | ---: | ---: |",
          f"| 제출 전체 | {len(sub)} | {allnone} | {allnone/len(sub)*100:.1f}% |"]

    labeled = sub[sub["image_id"].isin({os.path.splitext(k)[0] for k in lab})]
    lab_none = (labeled["final_date"] == "NONE").sum()
    truth_none = sum(1 for v in lab.values() if v == "NONE")
    L.append(f"| 정답 있는 이미지 | {len(labeled)} | {lab_none} | {lab_none/len(labeled)*100:.1f}% |")
    L.append(f"| 그 중 **정답이 NONE** 인 것 | {len(labeled)} | {truth_none} | "
             f"{truth_none/len(labeled)*100:.1f}% |")
    L.append("")
    L.append(f"> 정답이 있는 {len(labeled)}장에서 NONE 출력은 **{lab_none}장({lab_none/len(labeled)*100:.1f}%)** 이고, "
             f"그 중 실제로 정답이 NONE 인 것은 {truth_none}장뿐이다. "
             "**NONE 남발이 아니라 못 읽어서 나온 NONE 이다.**")
    L.append("")
    L.append(f"> ⚠️ 다만 제출 전체 {len(sub)}장 기준으로는 NONE 이 **{allnone/len(sub)*100:.1f}%** 다. "
             "라벨이 없는 나머지가 더 어려운지, 아니면 파이프라인이 더 많이 놓치는지는 "
             "라벨이 없어 알 수 없다. 라벨 표본의 미검출률이 하한이라고 보면 된다.")
    L.append("")

    # 부분 출력이 실제로 쓰이고 있는가
    part = sub[(sub["final_date"] != "NONE") &
               (sub[["year", "month", "day"]] == "NONE").any(axis=1)]
    L.append(f"- 부분 출력(연·월만 / 월·일만) 사용: **{len(part)}장** "
             f"({len(part)/len(sub)*100:.1f}%)")
    L.append("")

    # ---------------- 2. 오답 분해 ----------------
    w = pd.read_csv(a.wrong)
    L += ["## 2. 오답 심층 분해\n",
          f"틀린 **{len(w)}장** 을 구조별로 가른다.\n"]

    def cluster(f):
        if f not in meta.index:
            return "?"
        r = meta.loc[f]
        if r["w"] == 640 and r["h"] == 640:
            return "sq640"
        return "hires" if r["mp"] > 5 else "mid"

    rows = []
    for _, r in w.iterrows():
        g = split3(r["truth"])
        p = split3(r["pred"])
        anc = set(str(r["anchors"]).split("|")) if pd.notna(r["anchors"]) else set()
        nc = int(r["n_cands"])
        if p == ("NONE",) * 3:
            kind = "미검출"
        elif nc >= 2:
            kind = "후보 여러개 중 오선택"
        elif g[1:] == p[1:]:
            kind = "연도만 오독"
        elif g[0] == p[0]:
            kind = "월/일 오독"
        else:
            kind = "단일 후보가 오답"
        rows.append({**r.to_dict(), "kind": kind, "cluster": cluster(r["file"]),
                     "pair": ("부터" in anc and "까지" in anc),
                     "mfg": any(x in anc for x in ("제조일자", "생산일자", "포장일자", "MFG"))})
    d = pd.DataFrame(rows)

    L += ["| 구조 | 장수 | 비율 |", "| --- | ---: | ---: |"]
    for k, c in d["kind"].value_counts().items():
        L.append(f"| {k} | {c} | {c/len(d)*100:.1f}% |")
    L.append("")

    L += ["### 클러스터 × 구조\n"]
    ct = pd.crosstab(d["cluster"], d["kind"])
    L.append("| 클러스터 | " + " | ".join(ct.columns) + " |")
    L.append("| --- | " + " | ".join(["---:"] * len(ct.columns)) + " |")
    for c, row in ct.iterrows():
        L.append(f"| {c} | " + " | ".join(str(v) for v in row) + " |")
    L.append("")

    L += ["### 후보가 하나뿐인데 틀린 경우 (재시도로도 두 번째를 못 찾음)\n",
          "| 파일 | 정답 | 출력 | 부터/까지 | 제조앵커 |",
          "| --- | --- | --- | :-: | :-: |"]
    for _, r in d[(d["n_cands"] == 1)].iterrows():
        L.append(f"| `{r['file']}` | {r['truth']} | {r['pred']} | "
                 f"{'O' if r['pair'] else ''} | {'O' if r['mfg'] else ''} |")
    L.append("")

    L += ["### 후보가 2개 이상인데 틀린 경우 (선택 규칙이 진짜 틀림)\n",
          "| 파일 | 정답 | 출력 | 후보수 | 앵커 |",
          "| --- | --- | --- | ---: | --- |"]
    ms = d[d["n_cands"] >= 2]
    for _, r in ms.iterrows():
        L.append(f"| `{r['file']}` | {r['truth']} | {r['pred']} | {r['n_cands']} | "
                 f"{str(r['anchors'])[:40]} |")
    L.append("")
    L.append(f"> 선택 규칙이 실제로 틀린 것은 **{len(ms)}장** 뿐이다. "
             "나머지는 후보를 하나밖에 못 만든 것이라 결국 **인식 문제**다.")

    d.to_csv(os.path.join(OUT_DIR, "error_deep.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(OUT_DIR, "33_none_and_errors.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
