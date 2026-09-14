"""
EDA 19단계 — 사람 정답 기준 '진짜 정확도' 측정

지금까지의 모든 실험은 coverage(날짜가 잡혔는가)와 plausible(연도가 그럴듯한가)이라는
**대용 지표**로만 측정했다. 잡힌 날짜가 *맞는* 날짜인지는 재본 적이 없다.

여기서 처음으로 사람이 눈으로 확인한 정답과 대조한다. 세 가지를 나눠 본다.

  coverage  후보를 하나라도 냈는가          (= 지금까지 재온 것)
  recall@k  정답이 후보 집합 안에 있는가     (선택 규칙의 상한)
  accuracy  최종 1개를 골라 그게 맞는가      (= 실제 점수)

coverage 와 accuracy 의 격차가 곧 '선택 규칙이 해야 할 일'의 크기다.
"""
import os
import sys
import json
import argparse
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "eda"))
from datex import (find_dates, find_anchors, merge_lines,
                   POSITIVE, NEGATIVE)

OUT_DIR = os.path.join(ROOT, "eda", "out")

PLAUSIBLE = range(2015, 2036)


def parse(texts, loose):
    out = {}
    for t in texts:
        for d in find_dates(t, loose=loose):
            if d["in_long_digits"] or d["day_missing"]:
                continue
            k = "%04d-%02d-%02d" % (d["y"], d["m"], d["d"])
            out.setdefault(k, d["kind"])
    return out


def select(cands, anchors):
    """
    최종 1개 선택 규칙 (METHOD.md 의 근거를 코드로).
      - 4자리 연도 우선
      - 타당 연도(2015~2035) 우선
      - 제조/소비 쌍으로 보이면 더 늦은 날짜
    """
    if not cands:
        return None
    def score(item):
        d, kind = item
        y = int(d[:4])
        s = 0.0
        s += 3.0 if kind.startswith("Y4") else 0.0      # 4자리 연도 정밀도 99.2%
        s += 2.0 if y in PLAUSIBLE else -5.0
        return s
    ranked = sorted(cands.items(), key=lambda it: (-score(it), it[0]))
    top = score(ranked[0])
    tied = [d for d, k in ranked if score((d, k)) == top]
    # 동점이면 더 늦은 날짜 (제조일 vs 소비기한)
    return max(tied)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", default=os.path.join(OUT_DIR, "truth_v2.json"))
    ap.add_argument("--jsonl", default=os.path.join(OUT_DIR, "ocr_sample.jsonl"))
    ap.add_argument("--out", default="19_real_accuracy.md")
    a = ap.parse_args()

    truth = json.load(open(a.truth, encoding="utf-8"))
    recs = {r["file"]: r for r in (json.loads(l) for l in open(a.jsonl, encoding="utf-8"))
            if r.get("ok")}
    common = [f for f in truth if f in recs]
    print(f"정답 {len(truth)}장 중 OCR 표본과 겹치는 것 {len(common)}장")

    variants = [
        ("A. 박스별 엄격 파싱 (기존)", False, False),
        ("B. + 인접 박스 병합", True, False),
        ("C. + 7자리 느슨 패턴", False, True),
        ("D. 병합 + 느슨", True, True),
    ]

    rows = []
    detail = {}
    for name, do_merge, loose in variants:
        cov = hit_in = hit_top = 0
        wrong_pick = []
        miss = []
        for f in common:
            r = recs[f]
            texts = []
            for key in ("lines_0", "lines_90"):
                ls = r.get(key) or []
                texts += [l["t"] for l in ls]
                if do_merge:
                    texts += merge_lines(ls)
            cands = parse(texts, loose)
            anc = find_anchors("\n".join(texts))
            gt = truth[f]
            if cands:
                cov += 1
            if gt in cands:
                hit_in += 1
            else:
                miss.append(f)
            pick = select(cands, anc)
            if pick == gt:
                hit_top += 1
            elif gt in cands:
                wrong_pick.append((f, gt, pick, sorted(cands)))
            detail.setdefault(f, {})[name] = (sorted(cands), pick)
        n = len(common)
        rows.append({"name": name, "cov": cov / n, "recall": hit_in / n,
                     "acc": hit_top / n, "n": n,
                     "wrong_pick": wrong_pick, "miss": miss})

    L = ["# EDA 19 — 사람 정답 기준 실제 정확도\n",
         f"- 사람이 눈으로 확인한 정답 **{len(truth)}장** 중, 03단계 OCR 표본과 겹치는 "
         f"**{len(common)}장**으로 측정",
         "",
         "| 지표 | 뜻 |",
         "| --- | --- |",
         "| coverage | 후보를 하나라도 냈는가 (**지금까지 재온 지표**) |",
         "| recall@k | 정답이 후보 집합 안에 있는가 (선택 규칙의 상한) |",
         "| **accuracy** | 최종 1개를 골라 그게 맞는가 (**= 실제 점수**) |",
         "",
         "## 결과\n",
         "| 방식 | coverage | recall@k | **accuracy** |",
         "| --- | ---: | ---: | ---: |"]
    for r in rows:
        L.append(f"| {r['name']} | {r['cov']*100:.1f}% | {r['recall']*100:.1f}% | "
                 f"**{r['acc']*100:.1f}%** |")
    L.append("")

    best = max(rows, key=lambda r: r["acc"])
    gap = (best["cov"] - best["recall"]) * 100
    L.append(f"> **coverage {best['cov']*100:.1f}% 인데 accuracy 는 {best['acc']*100:.1f}%** "
             f"(`{best['name']}` 기준).")
    L.append(">")
    L.append(f"> 격차의 정체: 후보를 냈지만 그 안에 정답이 없는 경우가 **{gap:.1f}%p** — "
             "즉 OCR 이 날짜를 *읽긴 읽었는데 틀리게* 읽은 것이다. "
             "`plausible` 같은 대용 지표로는 절대 잡히지 않는 유형이다.")
    L.append("")

    L.append("## 선택 규칙이 틀린 사례 (정답은 후보에 있었는데 다른 걸 고름)\n")
    if best["wrong_pick"]:
        L.append("| 파일 | 정답 | 고른 값 | 후보 |")
        L.append("| --- | --- | --- | --- |")
        for f, gt, pick, cands in best["wrong_pick"]:
            L.append(f"| `{f}` | {gt} | **{pick}** | {cands} |")
    else:
        L.append("없음 — 정답이 후보에 있으면 항상 골라냈다.")
    L.append("")

    L.append("## 정답이 후보에 없던 이미지\n")
    L.append(f"{len(best['miss'])}장: `{', '.join(best['miss'])}`")
    L.append("")

    # 오독 사례 분석
    L.append("## 오독(그럴듯하지만 틀린 날짜) 사례\n")
    L.append("| 파일 | 정답 | OCR 후보 |")
    L.append("| --- | --- | --- |")
    nmis = 0
    for f in best["miss"]:
        c, _ = detail[f][best["name"]]
        if c:
            L.append(f"| `{f}` | {truth[f]} | {c} |")
            nmis += 1
    if nmis == 0:
        L.append("| — | — | 후보 자체를 못 낸 경우만 있었다 |")
    L.append("")
    L.append(f"> 후보는 냈으나 전부 틀린 이미지 **{nmis}장**. "
             "이런 경우는 `NONE` 을 내는 것보다 나을 게 없고, "
             "채점 방식이 오답 감점이라면 오히려 손해다.")
    L.append("")

    L.append("## 한계\n")
    L.append(f"- 표본 {len(common)}장은 작다. 95% 신뢰구간이 대략 ±{1.96*np.sqrt(0.25/len(common))*100:.0f}%p 다.")
    L.append("- 라벨링 250장을 마치면 이 표가 신뢰할 수 있는 수치가 된다.")

    with open(os.path.join(OUT_DIR, a.out), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
