"""
EDA 4단계 — OCR 결과 내용 분석

03 이 만든 ocr_sample.jsonl 을 읽어
 - 90° 회전 필요 비율 / 회전이 인식률에 주는 영향
 - CPU 속도 (해상도별) → 채점 예산 대비
 - 앵커 키워드 출현 빈도 (소비기한 vs 유통기한 vs 제조)
 - 날짜 후보 개수 분포 · 표기 형식 분포
 - 오탐원(품목보고번호/바코드/전화번호) 빈도
 - 자동 라벨링 난이도 등급
을 04_report.md 로 정리한다.
"""
import os
import sys
import json
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datex import find_dates, find_anchors, count_distractors, POSITIVE, NEGATIVE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "eda", "out")


def joined(lines):
    return "\n".join(l["t"] for l in lines)


def main():
    path = os.path.join(OUT_DIR, "ocr_sample.jsonl")
    recs = [json.loads(l) for l in open(path, encoding="utf-8")]
    recs = [r for r in recs if r.get("ok")]
    n = len(recs)

    rows = []
    for r in recs:
        best = r["best"]
        # 방향을 '고르지' 않고 0°/90° 결과를 모두 합친다.
        # (실측: 한쪽만 고르면 정답 날짜를 통째로 잃는 경우가 있다 — §2 참조)
        lines = r["lines_0"] + r["lines_90"]
        text = joined(lines)
        anc = find_anchors(text)
        dts = find_dates(text)
        clean = [d for d in dts if not d["in_long_digits"] and not d["day_missing"]]
        # 같은 날짜가 두 방향에서 중복 검출되면 1건으로
        seen, uniq = set(), []
        for d in clean:
            k = (d["y"], d["m"], d["d"])
            if k not in seen:
                seen.add(k)
                uniq.append(d)
        clean = uniq
        dis = count_distractors(text)
        rows.append({
            "file": r["file"],
            "image_id": r["image_id"],
            "orig_w": r.get("orig_w"), "orig_h": r.get("orig_h"),
            "proc_w": r.get("proc_w"), "proc_h": r.get("proc_h"),
            "mp": (r.get("orig_w", 0) * r.get("orig_h", 0)) / 1e6,
            "t_0": r["t_0"], "t_90": r["t_90"], "t_best": r["t_90"] if best == 90 else r["t_0"],
            "score_0": r["score_0"], "score_90": r["score_90"], "best": best,
            "n_lines": len(lines),
            "n_lines_0": r["n_0"], "n_lines_90": r["n_90"],
            "mean_conf": float(np.mean([l["c"] for l in lines])) if lines else 0.0,
            "text": text,
            "anchors": sorted(anc),
            "has_pos": bool(anc & set(POSITIVE)),
            "has_neg": bool(anc & set(NEGATIVE)),
            "has_soft": "소비기한" in anc,
            "has_yut": "유통기한" in anc,
            "has_bydo": "별도표기" in anc,
            "n_dates_all": len(dts),
            "n_dates": len(clean),
            "kinds": [d["kind"] for d in clean],
            "dates": [f"{d['y']}-{d['m']:02d}-{d['d']:02d}" for d in clean],
            "n_in_long": sum(d["in_long_digits"] for d in dts),
            **{f"dis_{k}": v for k, v in dis.items()},
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "ocr_analysis.csv"), index=False, encoding="utf-8-sig")

    L = []
    A = L.append
    A("# EDA 4 — OCR 내용 분석\n")
    A(f"- 표본: **{n:,}장** (층화 추출), OCR: PaddleOCR PP-OCRv3 det + korean PP-OCRv4 rec, CPU")
    A(f"- 처리 해상도: 긴 변 {int(df['proc_w'].combine(df['proc_h'], max).max())}px 로 축소\n")

    # ---------------- 1. 속도 ----------------
    A("## 1. CPU 추론 속도 (채점 예산 검증)\n")
    A("> 규정: 4-Core vCPU, `--ExecutePreprocessor.timeout=2400`. "
      "배포 데이터와 같은 3,352장이면 **장당 0.72초**가 상한.\n")
    A("| 지표 | 값 |")
    A("| --- | ---: |")
    A(f"| 1회 OCR 소요 (0°) 중앙값 | {df['t_0'].median():.2f} s |")
    A(f"| 1회 OCR 소요 (0°) 평균 | {df['t_0'].mean():.2f} s |")
    A(f"| 1회 OCR 소요 p90 | {df['t_0'].quantile(.9):.2f} s |")
    A(f"| 1회 OCR 소요 최대 | {df['t_0'].max():.2f} s |")
    A(f"| 0°+90° 2회 합계 중앙값 | {(df['t_0']+df['t_90']).median():.2f} s |")
    A("")
    A("해상도 구간별 1회 OCR 시간:\n")
    A("| 원본 megapixel | 장수 | 중앙값 | p90 |")
    A("| --- | ---: | ---: | ---: |")
    bins = [0, 0.5, 1, 2, 5, 9, 100]
    labs = ["<0.5MP", "0.5-1MP", "1-2MP", "2-5MP", "5-9MP", ">9MP"]
    df["mpbin"] = pd.cut(df["mp"], bins, labels=labs)
    for lab, g in df.groupby("mpbin", observed=True):
        A(f"| {lab} | {len(g):,} | {g['t_0'].median():.2f} s | {g['t_0'].quantile(.9):.2f} s |")
    A("")
    med1 = df["t_0"].median()
    A(f"> ⚠️ 단일 프로세스 기준 중앙값 **{med1:.2f}초/장**. "
      f"4코어 병렬로 나누면 이론상 **{med1/4:.2f}초/장**이지만, "
      f"0°+90° 2회를 돌리면 **{(df['t_0']+df['t_90']).median()/4:.2f}초/장**으로 "
      f"예산(0.72초)의 {(df['t_0']+df['t_90']).median()/4/0.72*100:.0f}% 를 쓴다.")
    A("")

    # ---------------- 2. 회전 ----------------
    A("## 2. 90° 회전 필요성\n")
    r90 = (df["best"] == 90)
    A(f"- 90° 회전본이 유의미하게 우세한 이미지: **{int(r90.sum()):,}장 "
      f"({r90.mean()*100:.1f}%)** (점수 15% 이상 우세 기준)")
    A(f"- 0° 우세: {int((~r90).sum()):,}장 ({(~r90).mean()*100:.1f}%)")
    A("")
    A("| 방향 | 인식 라인수 중앙값 |")
    A("| --- | ---: |")
    A(f"| 0° 고정 | {df['n_lines_0'].median():.0f} |")
    A(f"| 90° 고정 | {df['n_lines_90'].median():.0f} |")
    A("")
    # 회전 선택이 날짜 검출에 미치는 영향
    d0 = [len([x for x in find_dates(joined(r["lines_0"]))
               if not x["in_long_digits"] and not x["day_missing"]]) for r in recs]
    d90 = [len([x for x in find_dates(joined(r["lines_90"]))
                if not x["in_long_digits"] and not x["day_missing"]]) for r in recs]
    df["nd_0"], df["nd_90"] = d0, d90
    only90 = int(((df["nd_0"] == 0) & (df["nd_90"] > 0)).sum())
    only0 = int(((df["nd_0"] > 0) & (df["nd_90"] == 0)).sum())
    A(f"- 0°에서는 날짜를 **하나도 못 찾고** 90°에서 찾은 이미지: **{only90:,}장 "
      f"({only90/n*100:.1f}%)**")
    A(f"- 반대(90°만 실패): {only0:,}장 ({only0/n*100:.1f}%)")
    A("")
    # 방향을 '고르는' 정책 vs '합치는' 정책 비교
    nd_best = np.where(df["best"] == 90, df["nd_90"], df["nd_0"])
    cov_0 = (df["nd_0"] > 0).mean()
    cov_best = (nd_best > 0).mean()
    cov_union = (df["n_dates"] > 0).mean()
    A("| 방향 정책 | 날짜 검출 커버리지 | OCR 호출 |")
    A("| --- | ---: | ---: |")
    A(f"| 0° 만 | {cov_0*100:.1f}% | 1회 |")
    A(f"| 점수로 한쪽 선택 | {cov_best*100:.1f}% | 2회 |")
    A(f"| **0°+90° 합집합** | **{cov_union*100:.1f}%** | 2회 |")
    A("")
    A(f"> ⚠️ **방향을 '고르면' 손해다.** 2회 돌려놓고 한쪽만 채택하면 {cov_best*100:.1f}% 인데, "
      f"두 결과를 합치면 {cov_union*100:.1f}% 로 **+{(cov_union-cov_best)*100:.1f}%p** 오른다. "
      "비용은 똑같다 — 어차피 두 번 돌렸기 때문. 방향 판별기를 만들 이유가 없다.")
    A("")

    # ---------------- 3. 앵커 ----------------
    A("## 3. 앵커 키워드 출현 빈도\n")
    cnt = Counter()
    for a in df["anchors"]:
        cnt.update(a)
    A("| 앵커 | 장수 | 비율 | 성격 |")
    A("| --- | ---: | ---: | --- |")
    role = {**{k: "✅ 정답 앵커" for k in POSITIVE},
            **{k: "❌ 반대 앵커" for k in NEGATIVE},
            "까지": "보조", "별도표기": "⚠️ 함정"}
    for k, c in cnt.most_common():
        A(f"| {k} | {c:,} | {c/n*100:.1f}% | {role.get(k, '')} |")
    A("")
    A(f"- 정답 앵커(소비기한/유통기한/Best Before 등) 보유: "
      f"**{int(df['has_pos'].sum()):,}장 ({df['has_pos'].mean()*100:.1f}%)**")
    A(f"- **앵커가 전혀 없는 이미지: {int((~df['has_pos'] & ~df['has_neg']).sum()):,}장 "
      f"({(~df['has_pos'] & ~df['has_neg']).mean()*100:.1f}%)** → 위치·형식만으로 판단해야 함")
    both = df["has_pos"] & df["has_neg"]
    A(f"- **정답 앵커와 제조 앵커가 동시에 있는 이미지: {int(both.sum()):,}장 "
      f"({both.mean()*100:.1f}%)** → 둘을 구분 못 하면 오답")
    A(f"- '별도표기' 함정(앵커 옆에 날짜가 없음): {int(df['has_bydo'].sum()):,}장 "
      f"({df['has_bydo'].mean()*100:.1f}%)")
    A("")

    # ---------------- 4. 날짜 후보 ----------------
    A("## 4. 날짜 후보\n")
    A("| 이미지당 날짜 후보 수 | 장수 | 비율 |")
    A("| ---: | ---: | ---: |")
    vc = df["n_dates"].clip(upper=6).value_counts().sort_index()
    for k, c in vc.items():
        lab = f"{int(k)}개" + ("+" if k == 6 else "")
        A(f"| {lab} | {c:,} | {c/n*100:.1f}% |")
    A("")
    A(f"- 날짜를 **하나도 못 찾은** 이미지: **{int((df['n_dates']==0).sum()):,}장 "
      f"({(df['n_dates']==0).mean()*100:.1f}%)** → 이대로면 `NONE` 처리")
    A(f"- 후보가 정확히 1개(고민 없음): {int((df['n_dates']==1).sum()):,}장 "
      f"({(df['n_dates']==1).mean()*100:.1f}%)")
    A(f"- **후보가 2개 이상(선택 규칙 필요): {int((df['n_dates']>=2).sum()):,}장 "
      f"({(df['n_dates']>=2).mean()*100:.1f}%)**")
    A("")
    A("### 날짜 표기 형식 분포\n")
    kc = Counter()
    for ks in df["kinds"]:
        kc.update(ks)
    tot = sum(kc.values()) or 1
    kind_desc = {
        "Y4_sep": "2026.05.29 / 2026-05-29 / 2026년 5월 29일 (4자리 연도 + 구분자)",
        "Y4_compact": "20260529 (구분자 없는 8자리)",
        "Y2_sep": "26.05.29 (2자리 연도)",
        "D_MON_Y": "29/SEP/2022 (일-영문월-연)",
        "MON_D_Y": "SEP/29/2022 (영문월-일-연)",
    }
    A("| 형식 | 건수 | 비율 | 예시 |")
    A("| --- | ---: | ---: | --- |")
    for k, c in kc.most_common():
        A(f"| `{k}` | {c:,} | {c/tot*100:.1f}% | {kind_desc.get(k,'')} |")
    A("")
    A(f"> **2자리 연도(`Y2_sep`)가 {kc.get('Y2_sep',0)/tot*100:.1f}%** — "
      "`YY` → `20YY` 확장 규칙이 없으면 그만큼 통째로 틀린다.")
    A("")
    A("### 추출된 연도 분포\n")
    ys = Counter()
    for ds in df["dates"]:
        for s in ds:
            ys[s[:4]] += 1
    A("| 연도 | 건수 |")
    A("| ---: | ---: |")
    for y, c in sorted(ys.items()):
        A(f"| {y} | {c:,} |")
    A("")

    # ---------------- 4b. 패턴별 정밀도 ----------------
    A("### 패턴별 정밀도 (연도 타당성으로 근사)\n")
    A("> 소비기한이라면 연도는 대략 2015~2035 범위여야 한다. "
      "이 범위를 벗어난 매치는 사실상 오탐(LOT 코드·일련번호)으로 본다.\n")
    kind_year = defaultdict(list)
    for r in recs:
        txt = joined(r["lines_0"] + r["lines_90"])
        for d in find_dates(txt):
            if not d["in_long_digits"] and not d["day_missing"]:
                kind_year[d["kind"]].append(d["y"])
    A("| 패턴 | 총 건수 | 타당(2015-2035) | 비타당 | 정밀도 근사 |")
    A("| --- | ---: | ---: | ---: | ---: |")
    for k, ys in sorted(kind_year.items(), key=lambda kv: -len(kv[1])):
        g = sum(2015 <= y <= 2035 for y in ys)
        A(f"| `{k}` | {len(ys):,} | {g:,} | {len(ys)-g:,} | {g/len(ys)*100:.1f}% |")
    A("")
    y2 = kind_year.get("Y2_sep", [])
    y4 = kind_year.get("Y4_sep", [])
    if y2 and y4:
        p2 = sum(2015 <= y <= 2035 for y in y2) / len(y2)
        p4 = sum(2015 <= y <= 2035 for y in y4) / len(y4)
        A(f"> ⚠️ **2자리 연도 패턴이 정밀도의 구멍이다.** "
          f"4자리 연도는 {p4*100:.1f}% 가 타당한 반면 2자리는 {p2*100:.1f}% 에 그친다. "
          f"`080723`, `411011`, `000703` 같은 LOT 코드가 그대로 날짜로 파싱된다.")
        A(">")
        A("> → **2자리 연도 후보는 (a) 앵커가 근처에 있거나 (b) 연도가 타당 범위일 때만 채택**하는 "
          "이중 조건을 걸어야 한다. 4자리 연도 후보에는 이 제약이 거의 손해를 주지 않는다 "
          f"(비타당 {len(y4)-sum(2015 <= y <= 2035 for y in y4)}건뿐).")
    A("")

    # ---------------- 5. 오탐원 ----------------
    A("## 5. 날짜로 오인되기 쉬운 숫자 (오탐원)\n")
    A("| 오탐원 | 출현 이미지 | 비율 | 총 건수 |")
    A("| --- | ---: | ---: | ---: |")
    for c in [c for c in df.columns if c.startswith("dis_")]:
        s = df[c]
        A(f"| {c[4:]} | {int((s>0).sum()):,} | {(s>0).mean()*100:.1f}% | {int(s.sum()):,} |")
    A("")
    A(f"- 14자리 이상 숫자(품목보고번호류)를 가진 이미지: "
      f"{int((df['dis_품목보고번호(14자리+)']>0).sum()):,}장 / "
      f"13자리(바코드): {int((df['dis_바코드(13자리)']>0).sum()):,}장")
    A("")
    A("> `품목보고번호 20130628332176` 의 앞 8자리는 `2013-06-28` 로 **완벽하게 날짜 형태**다. "
      "이런 것을 막는 방법은 두 가지인데, 실측 결과 **정규식 경계(lookaround)로 막는 쪽이 확실**했다.")
    A(">")
    A("> - `(?<!\\d)` / `(?!\\d)` 로 앞뒤에 숫자가 붙은 매치를 원천 차단 → "
      f"이 표본에서 긴 숫자열 내부 오탐 **{int(df['n_in_long'].sum())}건** (사후 필터가 필요 없었음)")
    A("> - 사후에 '9자리 이상 숫자 안에 있으면 버린다'로 거르는 방식은, "
      "경계 조건이 이미 막아주므로 보조 수단이다.")
    A("")

    # ---------------- 6. 난이도 등급 ----------------
    A("## 6. 자동 라벨링 난이도 등급\n")

    def grade(r):
        if r["n_dates"] == 0:
            return "E. 날짜 미검출"
        if r["n_dates"] == 1 and r["has_pos"] and not r["has_neg"]:
            return "A. 후보1개+정답앵커"
        if r["n_dates"] == 1:
            return "B. 후보1개, 앵커 불충분"
        if r["has_pos"] and not r["has_neg"]:
            return "C. 후보다수+정답앵커만"
        return "D. 후보다수+제조앵커 혼재"

    df["grade"] = df.apply(grade, axis=1)
    A("| 등급 | 장수 | 비율 | 대응 |")
    A("| --- | ---: | ---: | --- |")
    plan = {
        "A. 후보1개+정답앵커": "룰만으로 자동 라벨 신뢰 가능",
        "B. 후보1개, 앵커 불충분": "그 1개를 채택하되 검증 표본에 포함",
        "C. 후보다수+정답앵커만": "앵커 근접도로 선택",
        "D. 후보다수+제조앵커 혼재": "**수동 검수 최우선 대상**",
        "E. 날짜 미검출": "전처리/해상도 개선 또는 NONE",
    }
    for g, c in df["grade"].value_counts().sort_index().items():
        A(f"| {g} | {c:,} | {c/n*100:.1f}% | {plan.get(g,'')} |")
    A("")
    auto = df["grade"].isin(["A. 후보1개+정답앵커", "B. 후보1개, 앵커 불충분"]).mean()
    A(f"> 후보가 1개뿐이라 사실상 고민이 없는 이미지가 **{auto*100:.1f}%**. "
      f"나머지 {100-auto*100:.1f}% 가 실제 승부처이며, 그 중 "
      f"**D등급 {df['grade'].eq('D. 후보다수+제조앵커 혼재').mean()*100:.1f}% 가 최우선 수동 검수 대상**이다.")
    A("")

    # ---------------- 7. 표본 케이스 ----------------
    A("## 7. 케이스 표본\n")
    A("### D등급 (제조 + 소비기한 혼재) 예시\n")
    for _, r in df[df["grade"].str.startswith("D")].head(6).iterrows():
        A(f"- `{r['file']}` — 앵커 {r['anchors']} / 후보 {r['dates']}")
    A("")
    A("### E등급 (날짜 미검출) 예시\n")
    for _, r in df[df["grade"].str.startswith("E")].head(10).iterrows():
        A(f"- `{r['file']}` ({r['orig_w']}x{r['orig_h']}, 라인 {r['n_lines']}개, "
          f"신뢰도 {r['mean_conf']:.2f}) 앵커 {r['anchors']}")
    A("")

    with open(os.path.join(OUT_DIR, "04_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("wrote 04_report.md / ocr_analysis.csv")


if __name__ == "__main__":
    main()
