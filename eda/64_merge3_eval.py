"""
EDA 64단계 — 세 파이프라인 비교와 합치기 판정

63단계 결과가 판을 바꿨다.
  우리     66.8%  (2.19초/장)   PP-OCRv3 det + 한국어 PP-OCRv4 rec + YOLOv8n
  팀원1    30.1%  (1.67초/장)   classical CV + EasyOCR
  팀원2  **79.3%** (0.36초/장)  RapidOCR (PP-OCRv6 det/rec, ONNX)

팀원2 가 정확도 +12.5%p, 속도 6배다. 그냥 갈아타야 하는 것처럼 보이지만,
결정 전에 확인할 것이 있다.

  1) 우리 66.8% 는 홀드아웃2 를 설정 선택에 네 번 쓴 뒤의 값이라 **낙관 편향**이 있다.
     팀원2 의 79.3% 는 이 표본을 한 번도 안 쓴 깨끗한 값이다. 실제 격차는 더 클 수 있다.
  2) 두 파이프라인이 서로 다른 이미지를 맞히면 합쳐서 더 올릴 수 있다.
  3) 우리 쪽이 더 나은 이미지가 있다면 그 원인을 봐야 한다 (규칙? YOLO? 해상도 사다리?).
"""
import os
import sys
import json
import argparse
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "eda", "out")


def split3(v):
    if not v or v == "NONE":
        return ("NONE",) * 3
    p = str(v).split("-")
    return tuple(p) if len(p) == 3 else ("NONE",) * 3


def is_none(t):
    return t == ("NONE",) * 3


def mcnemar(pa, pb):
    up = sum(1 for k in pa if pb.get(k, 0) > pa[k])
    dn = sum(1 for k in pa if pb.get(k, 0) < pa[k])
    if up + dn == 0:
        return up, dn, 1.0
    from math import comb
    n, k = up + dn, min(up, dn)
    return up, dn, min(1.0, sum(comb(n, i) for i in range(k + 1)) / 2 ** n * 2)


def main():
    argparse.ArgumentParser().parse_args()

    lab = json.load(open(os.path.join(OUT_DIR, "truth_holdout2.json"),
                         encoding="utf-8"))["labels"]
    ours = json.load(open(os.path.join(OUT_DIR, "57_env_snapshot.json"),
                          encoding="utf-8"))["pred"]
    t1 = json.load(open(os.path.join(OUT_DIR, "58_team_pred.json"),
                        encoding="utf-8"))["pred"]
    t2 = json.load(open(os.path.join(OUT_DIR, "63_team2_pred.json"),
                        encoding="utf-8"))["pred"]
    files = [f for f in lab if f in ours and f in t1 and f in t2]

    A = {f: split3(ours[f]) for f in files}
    B = {f: split3(t1[f]) for f in files}
    C = {f: split3(t2[f]) for f in files}

    def per(pred):
        return {f: sum(split3(lab[f])[i] == pred[f][i] for i in range(3)) for f in files}

    def sc(pred):
        p = per(pred)
        hits = sum(p.values())
        return {"score": hits / (3 * len(files)),
                "fill": sum(not is_none(pred[f]) for f in files) / len(files),
                "exact": sum(split3(lab[f]) == pred[f] for f in files) / len(files),
                "per": p}

    # 합치기: 팀원2 를 주로 쓰고, 팀원2 가 NONE 인 이미지만 우리 것으로 메운다
    M1 = {f: (A[f] if is_none(C[f]) else C[f]) for f in files}
    # 필드 단위로 팀원2 의 NONE 칸만 우리 값으로 채운다
    M2 = {f: tuple(A[f][i] if C[f][i] == "NONE" else C[f][i] for i in range(3))
          for f in files}
    # 반대 방향 — 우리를 주로 쓰고 우리가 NONE 일 때만 팀원2
    M3 = {f: (C[f] if is_none(A[f]) else A[f]) for f in files}
    # 셋 중 둘 이상이 같은 값이면 채택 (다수결), 아니면 팀원2
    M4 = {}
    for f in files:
        vote = Counter([A[f], B[f], C[f]])
        top, n = vote.most_common(1)[0]
        M4[f] = top if n >= 2 else C[f]
    # 오라클 — 우리/팀원2 중 더 나은 쪽 (도달 불가)
    OR = {}
    for f in files:
        g = split3(lab[f])
        OR[f] = A[f] if sum(g[i] == A[f][i] for i in range(3)) >= \
            sum(g[i] == C[f][i] for i in range(3)) else C[f]

    rows = [("우리 (PP-OCRv3+v4 + YOLO)", A),
            ("팀원1 (classical CV + EasyOCR)", B),
            ("팀원2 (RapidOCR / PP-OCRv6)", C),
            ("합치기 A — 팀원2 주, 실패 시 우리", M1),
            ("합치기 B — 팀원2 주, NONE 칸만 우리", M2),
            ("합치기 C — 우리 주, 실패 시 팀원2", M3),
            ("합치기 D — 셋 중 다수결", M4),
            ("오라클 (우리/팀원2 중 좋은 쪽)", OR)]

    S = {}
    L = ["# EDA 64 — 세 파이프라인 비교와 합치기 판정\n",
         "홀드아웃2 198장. 우리 66.8% 는 이 표본을 설정 선택에 네 번 쓴 뒤의 값이라",
         "**낙관 편향**이 있고, 팀원2 의 값은 이 표본을 한 번도 안 쓴 깨끗한 값이다.",
         "즉 아래 격차는 하한으로 봐야 한다.",
         "",
         "| 구성 | **부분점수** | 값이 채워진 행 | 완전일치 |",
         "| --- | ---: | ---: | ---: |"]
    for name, pred in rows:
        s = sc(pred)
        S[name] = s
        L.append(f"| {name} | **{s['score']*100:.1f}%** | {s['fill']*100:.1f}% | "
                 f"{s['exact']*100:.1f}% |")
    L.append("")

    base = S["팀원2 (RapidOCR / PP-OCRv6)"]
    ourss = S["우리 (PP-OCRv3+v4 + YOLO)"]
    up, dn, pv = mcnemar(ourss["per"], base["per"])
    L.append(f"우리 → 팀원2: **{(base['score']-ourss['score'])*100:+.1f}%p**, "
             f"개선 {up}장 / 악화 {dn}장, McNemar p={pv:.4f}")
    L.append("")
    for name in [n for n, _ in rows if n.startswith("합치기")] + ["오라클 (우리/팀원2 중 좋은 쪽)"]:
        d = (S[name]["score"] - base["score"]) * 100
        u, dd, p = mcnemar(base["per"], S[name]["per"])
        L.append(f"- **{name}** — 팀원2 대비 {d:+.1f}%p (개선 {u} / 악화 {dd}, p={p:.3f})")
    L.append("")

    # 상호보완성
    c_none = [f for f in files if is_none(C[f])]
    rescued = [f for f in c_none if not is_none(A[f])]
    gain = 0
    for f in rescued:
        g = split3(lab[f])
        gain += sum(g[i] == A[f][i] for i in range(3))
    L.append("## 상호보완성\n")
    L.append(f"- 팀원2 가 전부 NONE 으로 낸 이미지: **{len(c_none)}장**")
    L.append(f"- 그중 우리가 값을 낸 것: **{len(rescued)}장**, 실제로 맞은 필드 **{gain}개**")
    a_none = [f for f in files if is_none(A[f])]
    rescued2 = [f for f in a_none if not is_none(C[f])]
    gain2 = 0
    for f in rescued2:
        g = split3(lab[f])
        gain2 += sum(g[i] == C[f][i] for i in range(3))
    L.append(f"- 우리가 전부 NONE 으로 낸 이미지: **{len(a_none)}장**")
    L.append(f"- 그중 팀원2 가 값을 낸 것: **{len(rescued2)}장**, 실제로 맞은 필드 **{gain2}개**")
    L.append("")

    better_ours = []
    for f in files:
        g = split3(lab[f])
        ha = sum(g[i] == A[f][i] for i in range(3))
        hc = sum(g[i] == C[f][i] for i in range(3))
        if ha > hc:
            better_ours.append((f, lab[f], "-".join(A[f]), "-".join(C[f]), ha, hc))
    L.append(f"## 우리 쪽이 더 나은 이미지 — {len(better_ours)}장\n")
    if better_ours:
        L.append("| 파일 | 정답 | 우리 | 팀원2 | 우리 | 팀원2 |")
        L.append("| --- | --- | --- | --- | ---: | ---: |")
        for f, g, pa, pc, ha, hc in better_ours[:30]:
            L.append(f"| `{f}` | {g} | {pa} | {pc} | {ha}/3 | {hc}/3 |")
    L.append("")

    with open(os.path.join(OUT_DIR, "64_merge3_eval.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L[:30]))
    print("\n→ eda/out/64_merge3_eval.md")


if __name__ == "__main__":
    main()
