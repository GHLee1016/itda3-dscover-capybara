"""
EDA 102단계 — 현재 계단에 인식기 칸을 덧붙이면 실제로 얼마나 오르는가

101단계는 **1칸만 쓸 때**를 쟀다. 거기서 rec 오라클 상한이 89.4% 였는데,
현재 파이프라인은 계단 4칸 덕분에 이미 93.9%(580장)다. 즉 rec 을 바꿔 얻는
것의 상당 부분을 det 계단이 이미 가져가고 있을 수 있다. **중복이 얼마나
되는지**가 채택 여부를 가른다.

99단계(det 변형 8종)와 101단계(rec 변형 4종)가 같은 580장에 대한 칸별 결과를
남겼으므로, OCR 을 다시 돌리지 않고 계단을 이어 붙여 시뮬레이션한다.

계단 규칙은 구현과 같다. 각 칸을 차례로 보며 **온전한 날짜**를 낸 첫 칸을
채택하고, 끝까지 없으면 부분 결과를 쓴다.

⚠️ 580장은 오염된 표본이다. 후보를 추리는 데만 쓰고, 채택 판정은 새 표본에서 한다.

    python eda/102_det_plus_rec.py
"""
import os
import sys
import json

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "eda", "out")


def s3(v):
    if not v or v == "NONE":
        return ("NONE",) * 3
    q = str(v).split("-")
    return tuple(q) if len(q) == 3 else ("NONE",) * 3


def full(v):
    return bool(v) and "NONE" not in str(v)


def main():
    det = json.load(open(os.path.join(OUT, "99_rungs.json"), encoding="utf-8"))
    rec = json.load(open(os.path.join(OUT, "101_recs.json"), encoding="utf-8"))
    lab = json.load(open(os.path.join(OUT, "88_all_pred.json"),
                         encoding="utf-8"))["label"]

    files = [f for f in det if f in rec and f in lab]
    n = len(files)
    print(f"두 측정에 공통으로 있는 {n}장으로 시뮬레이션한다")
    print(f"  (det 측정 {len(det)}장 · rec 측정 {len(rec)}장)\n")

    # 한 곳에 모은다. rec 쪽 '현재' 칸은 det 1칸과 같은 구성이라 뺀다.
    REC_ALIAS = {"rec_medium": "R medium", "rec_korean": "R korean",
                 "rec_en": "R en"}
    got, sec = {}, {}
    for f in files:
        g = dict(det[f]["got"])
        s = dict(det[f]["sec"])
        for k, v in rec[f]["got"].items():
            if k in REC_ALIAS:
                g[REC_ALIAS[k]] = v
                s[REC_ALIAS[k]] = rec[f]["sec"].get(k, 0)
        got[f], sec[f] = g, s

    def run(order):
        ok = exact = 0
        cost = 0.0
        used = {nm: 0 for nm in order}
        for f in files:
            pick, partial = None, None
            for nm in order:
                cost += sec[f].get(nm, 0)
                v = got[f].get(nm)
                if full(v):
                    pick = v
                    used[nm] += 1
                    break
                if partial is None and v:
                    partial = v
            if pick is None:
                pick = partial
            g, p = s3(lab[f]), s3(pick)
            ok += sum(g[i] == p[i] for i in range(3))
            exact += (g == p)
        return {"fa": ok / (n * 3) * 100, "ex": exact / n * 100,
                "sec": cost / n, "used": used}

    BASE = ["1 small@1024", "2 adaptive", "3 medium@1024", "4 CLAHE+med"]
    CANDS = [
        ("A 현재 계단 (det 4칸)", BASE),
        ("B + R medium", BASE + ["R medium"]),
        ("C + R korean", BASE + ["R korean"]),
        ("D + R en", BASE + ["R en"]),
        ("E + R korean + R en", BASE + ["R korean", "R en"]),
        ("F + R medium + R korean", BASE + ["R medium", "R korean"]),
        ("G + 셋 다", BASE + ["R medium", "R korean", "R en"]),
        ("H R korean 을 2번째로", ["1 small@1024", "R korean", "2 adaptive",
                                  "3 medium@1024", "4 CLAHE+med"]),
    ]

    rows = [(nm, run(o), o) for nm, o in CANDS]
    base = rows[0][1]

    L = ["# EDA 102 — 현재 계단에 인식기 칸을 덧붙이면\n",
         "101단계의 89.4% 는 **1칸만 쓸 때**의 상한이다. 현재 계단은 이미 그보다",
         "높으므로, 관건은 rec 이 주는 이득이 det 계단과 **얼마나 겹치는가**다.",
         "",
         f"- 공통 {n}장 (홀드아웃4 제외) · 99·101단계 결과를 이어 붙인 시뮬레이션",
         "",
         "| 구성 | 필드 정확도 | 완전일치 | 장당(병렬) | 현재 대비 |",
         "| --- | ---: | ---: | ---: | ---: |"]
    for nm, s, _ in rows:
        d = s["fa"] - base["fa"]
        mark = "기준" if nm == rows[0][0] else f"{d:+.1f}%p"
        L.append(f"| {nm} | **{s['fa']:.1f}%** | {s['ex']:.1f}% | "
                 f"{s['sec']:.2f}초 | {mark} |")
    L.append("")
    L.append("> 장당 시간은 4워커 병렬 기준이라 채점 조건(순차)보다 크다. 상대 비교용이다.")
    L.append("")

    # 오라클: det 4칸 + rec 3칸을 모두 돌렸을 때
    ALL = BASE + ["R medium", "R korean", "R en"]

    def oracle():
        ok = exact = 0
        for f in files:
            g = s3(lab[f])
            best, bh = None, -1
            for nm in ALL:
                h = sum(g[i] == s3(got[f].get(nm))[i] for i in range(3))
                if h > bh:
                    best, bh = got[f].get(nm), h
            p = s3(best)
            ok += sum(g[i] == p[i] for i in range(3))
            exact += (g == p)
        return ok / (n * 3) * 100, exact / n * 100
    ofa, oex = oracle()
    L.append(f"일곱 칸 오라클 상한: **{ofa:.1f}%** (완전일치 {oex:.1f}%)")
    L.append("")

    best = max(rows[1:], key=lambda x: x[1]["fa"])
    L.append("## 칸별 처리 분담 (가장 나은 구성)\n")
    s = best[1]
    part = " · ".join(f"{k} {v}" for k, v in s["used"].items() if v)
    L.append(f"- **{best[0]}**: {part}")
    L.append("")

    L.append("## 판단\n")
    d = best[1]["fa"] - base["fa"]
    r = best[1]["sec"] / base["sec"] if base["sec"] else 1
    if d < 0.3:
        L.append(f"가장 나은 것이 {d:+.1f}%p 다. **det 계단이 이미 가져간 것과 거의 겹친다.**")
        L.append("인식기 교체로 얻을 것이 없다. 이 방향도 접는다.")
    else:
        L.append(f"**{best[0]}** 가 {d:+.1f}%p, 비용은 {r:.1f}배다.")
        L.append("")
        L.append("오염된 표본에서 나온 값이므로 새 표본에서 확인해야 한다.")
        L.append(f"오라클 상한({ofa:.1f}%)까지는 {ofa - best[1]['fa']:.1f}%p 가 더 남아 있는데,")
        L.append("그 부분은 라벨 없이 고를 방법이 없어 가져오지 못한다.")

    with open(os.path.join(OUT, "102_det_plus_rec.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L[5:]))
    print("\n→ eda/out/102_det_plus_rec.md")


if __name__ == "__main__":
    main()
