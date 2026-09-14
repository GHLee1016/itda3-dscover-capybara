"""
EDA 11단계 — 인접 박스 병합 + 느슨한 자릿수 패턴

10단계 오라클 실험에서 드러난 것:
  002401  det 가 '2022/11/04' 를 '20221' + '104' 두 박스로 쪼갰다 → 붙이면 정답
  001201  '2022.07.16' 을 '2022716' 으로 읽었다 (07 의 0 탈락) → 7자리 허용하면 정답
  000801  주황 카톤 위 점(dot) 인쇄 — 어떤 전처리로도 판독 불가

앞의 둘은 **추가 OCR 없이 후처리만으로** 복구된다. 비용 0.
이미 저장된 03단계 OCR 결과(ocr_sample.jsonl)로 검증한다.
"""
import os
import sys
import json
import argparse
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "eda"))
from datex import find_dates, PATTERNS
import re

OUT_DIR = os.path.join(ROOT, "eda", "out")


# ---------------------------------------------------------------- 박스 병합
def box_geom(b):
    q = np.array(b, dtype=float).reshape(4, 2)
    cx, cy = q[:, 0].mean(), q[:, 1].mean()
    w = max(np.linalg.norm(q[0] - q[1]), np.linalg.norm(q[2] - q[3]))
    h = max(np.linalg.norm(q[0] - q[3]), np.linalg.norm(q[1] - q[2]))
    return cx, cy, w, h, q[:, 0].min(), q[:, 0].max()


def merge_lines(lines, y_tol=0.6, gap_tol=2.0):
    """같은 행에 있고 가로로 인접한 박스들의 텍스트를 이어 붙인다."""
    items = []
    for l in lines:
        if "box" not in l or not l["box"]:
            continue
        cx, cy, w, h, x0, x1 = box_geom(l["box"])
        if h < 1:
            continue
        items.append({"t": l["t"], "cy": cy, "h": h, "x0": x0, "x1": x1})
    items.sort(key=lambda d: (d["cy"], d["x0"]))

    rows, used = [], [False] * len(items)
    for i, a in enumerate(items):
        if used[i]:
            continue
        row = [a]
        used[i] = True
        for j in range(i + 1, len(items)):
            if used[j]:
                continue
            b = items[j]
            # 같은 행: 세로 중심 차이가 글자 높이의 y_tol 배 이내, 높이도 비슷
            if abs(b["cy"] - a["cy"]) > y_tol * max(a["h"], b["h"]):
                continue
            if not (0.5 <= b["h"] / a["h"] <= 2.0):
                continue
            row.append(b)
            used[j] = True
        row.sort(key=lambda d: d["x0"])
        rows.append(row)

    merged = []
    for row in rows:
        if len(row) == 1:
            continue
        # 가로 간격이 글자 높이의 gap_tol 배 이내인 것만 이어 붙인다
        cur = [row[0]]
        for prev, nxt in zip(row, row[1:]):
            gap = nxt["x0"] - prev["x1"]
            if gap <= gap_tol * max(prev["h"], nxt["h"]):
                cur.append(nxt)
            else:
                if len(cur) > 1:
                    merged.append("".join(d["t"] for d in cur))
                cur = [nxt]
        if len(cur) > 1:
            merged.append("".join(d["t"] for d in cur))
    return merged


# ------------------------------------------------- 느슨한 자릿수 패턴 (7자리)
# 2022.07.16 → '2022716' 처럼 앞자리 0 이 탈락한 경우
P_Y4_M1_D2 = re.compile(r"(?<![\d])(?P<y>20[0-4]\d)(?P<m>[1-9])(?P<d>0[1-9]|[12]\d|3[01])(?![\d])")
P_Y4_M2_D1 = re.compile(r"(?<![\d])(?P<y>20[0-4]\d)(?P<m>0[1-9]|1[0-2])(?P<d>[1-9])(?![\d])")


def loose_dates(txt):
    """7자리 압축형(연4 + 월1 + 일2 / 연4 + 월2 + 일1)을 추가로 뽑는다."""
    out = set()
    for rx in (P_Y4_M1_D2, P_Y4_M2_D1):
        for m in rx.finditer(txt or ""):
            y, mm, d = int(m["y"]), int(m["m"]), int(m["d"])
            if 1 <= mm <= 12 and 1 <= d <= 31:
                out.add("%04d-%02d-%02d" % (y, mm, d))
    return out


def strict_dates(txt):
    return {"%04d-%02d-%02d" % (d["y"], d["m"], d["d"])
            for d in find_dates(txt or "")
            if not d["in_long_digits"] and not d["day_missing"]}


def plausible(s):
    return 2015 <= int(s[:4]) <= 2035


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default=os.path.join(OUT_DIR, "ocr_sample.jsonl"))
    ap.add_argument("--out", default="11_linemerge.md")
    a = ap.parse_args()

    recs = [json.loads(l) for l in open(a.jsonl, encoding="utf-8")]
    recs = [r for r in recs if r.get("ok")]
    truth = json.load(open(os.path.join(OUT_DIR, "truth.json"), encoding="utf-8"))
    n = len(recs)

    variants = {
        "A. 현재 (박스별 엄격 파싱)":        lambda per, mg: strict_dates("\n".join(per)),
        "B. + 인접 박스 병합":               lambda per, mg: strict_dates("\n".join(per + mg)),
        "C. + 7자리 느슨 패턴":              lambda per, mg: strict_dates("\n".join(per)) | loose_dates("\n".join(per)),
        "D. 병합 + 느슨 (둘 다)":            lambda per, mg: (strict_dates("\n".join(per + mg))
                                                              | loose_dates("\n".join(per + mg))),
    }

    stats = {k: {"cov": 0, "pl_ok": 0, "pl_tot": 0, "n_cand": 0} for k in variants}
    per_img = defaultdict(dict)
    for r in recs:
        per_txt, merged = [], []
        for key in ("lines_0", "lines_90"):
            ls = r.get(key) or []
            per_txt += [l["t"] for l in ls]
            merged += merge_lines(ls)
        for k, fn in variants.items():
            ds = fn(per_txt, merged)
            per_img[r["file"]][k] = ds
            s = stats[k]
            if ds:
                s["cov"] += 1
            s["n_cand"] += len(ds)
            for d in ds:
                s["pl_tot"] += 1
                s["pl_ok"] += plausible(d)

    L = ["# EDA 11 — 인접 박스 병합 + 느슨한 자릿수 패턴\n",
         f"- 대상: 03단계에서 이미 저장한 OCR 결과 **{n:,}장** (0°·90°)",
         "- **추가 OCR 없음 → 추론 비용 증가 0**",
         "",
         "10단계 오라클 실험에서 확인한 두 가지 손실을 후처리로 되돌린다.",
         "",
         "| 사례 | OCR 출력 | 원인 |",
         "| --- | --- | --- |",
         "| `002401` | `'20221'` + `'104'` | det 가 날짜 한 줄을 두 박스로 분할 |",
         "| `001201` | `'2022716'` | `2022.07.16` 에서 `07` 의 0 이 탈락 |",
         "",
         "## 결과\n",
         "| 방식 | coverage | Δ | plausible | 이미지당 후보 |",
         "| --- | ---: | ---: | ---: | ---: |"]
    base = stats["A. 현재 (박스별 엄격 파싱)"]
    for k, s in stats.items():
        d = (s["cov"] - base["cov"]) / n * 100
        L.append(f"| {k} | **{s['cov']/n*100:.1f}%** | {d:+.1f}%p | "
                 f"{s['pl_ok']/max(s['pl_tot'],1)*100:.1f}% | {s['n_cand']/n:.2f} |")
    L.append("")

    # 정답 10장 검증
    L.append("## 육안 정답 검증\n")
    L.append("| 파일 | 정답 | A 현재 | B 병합 | C 느슨 | D 둘다 |")
    L.append("| --- | --- | :-: | :-: | :-: | :-: |")
    hits = defaultdict(int)
    seen = 0
    for f, gt in truth.items():
        if f not in per_img:
            continue
        seen += 1
        cells = []
        for k in variants:
            ok = gt in per_img[f][k]
            hits[k] += ok
            cells.append("O" if ok else "·")
        L.append(f"| `{f}` | {gt} | " + " | ".join(cells) + " |")
    if seen:
        L.append(f"| **합계** | | " +
                 " | ".join(f"**{hits[k]}/{seen}**" for k in variants) + " |")
    L.append("")
    L.append(f"> 정답셋 {seen}장만 표본에 포함되어 있다 (500장 층화표본과 겹치는 만큼).")
    L.append("")

    best = max(stats.items(), key=lambda kv: kv[1]["cov"])
    dcov = (best[1]["cov"] - base["cov"]) / n * 100
    L.append(f"> **{best[0]}** 가 coverage **{best[1]['cov']/n*100:.1f}%** 로 최고 "
             f"({dcov:+.1f}%p). plausible 은 {base['pl_ok']/max(base['pl_tot'],1)*100:.1f}% → "
             f"{best[1]['pl_ok']/max(best[1]['pl_tot'],1)*100:.1f}%.")
    L.append(">")
    L.append("> plausible 이 크게 떨어지지 않는다면 순수 이득이다 — **추론 비용이 0**이므로 "
             "CPU 예산에 영향을 주지 않는다.")

    with open(os.path.join(OUT_DIR, a.out), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
