"""
EDA 100단계 — 계단의 순서를 바꾸면 어떻게 되는가

99단계에서 두 가지가 드러났다.
  · 칸을 늘리는 방향은 막혔다. 여덟 칸 오라클 상한이 94.9% 로 현재(93.6%)보다
    1.3%p 뿐이고, 라벨 없이 고르는 다수결은 오히려 더 나빴다(93.4%).
  · 그런데 **medium@1024 단독이 89.1% 로 small@1024 의 86.8% 보다 높다.**
    지금은 small 을 1차로 두고 medium 을 재시도로 쓴다. 순서가 최선이 아닐 수 있다.

99단계가 칸별 결과를 전부 남겼으므로 OCR 을 다시 돌리지 않고 순서만 바꿔
시뮬레이션한다. 계단의 규칙은 실제 구현과 같다.
    각 칸을 차례로 보면서 **온전한 날짜**(NONE 이 없는 것)를 낸 첫 칸을 채택한다.
    끝까지 없으면 그때 부분 결과를 쓴다.

비용도 같은 자료로 잰다. 앞 칸이 성공하면 뒤 칸은 돌지 않으므로,
순서가 곧 비용이다.

⚠️ 580장은 오염된 표본이다. 여기서는 **후보를 추리는 데만** 쓰고,
   채택 판정은 홀드아웃4 에서 한 번만 한다.

    python eda/100_rung_order.py
"""
import os
import sys
import json
import itertools

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
    R = json.load(open(os.path.join(OUT, "99_rungs.json"), encoding="utf-8"))
    lab = json.load(open(os.path.join(OUT, "88_all_pred.json"),
                         encoding="utf-8"))["label"]
    files = [f for f in R if f in lab]
    n = len(files)
    print(f"{n}장으로 순서를 시뮬레이션한다\n")

    names = sorted({k for r in R.values() for k in r["got"]})
    SHORT = {nm: nm.split(" ", 1)[1] for nm in names}

    def run(order):
        """계단을 순서대로 오르며 채택 결과와 비용을 낸다."""
        ok = exact = 0
        cost = 0.0
        used = {nm: 0 for nm in order}
        for f in files:
            r = R[f]
            pick, partial = None, None
            for nm in order:
                cost += r["sec"].get(nm, 0)
                v = r["got"].get(nm)
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
    CANDS = {
        "A 현재 (small → adaptive → medium → CLAHE+med)": BASE,
        "B medium 먼저 (medium → small → adaptive → CLAHE+med)":
            ["3 medium@1024", "1 small@1024", "2 adaptive", "4 CLAHE+med"],
        "C medium → CLAHE+med → small":
            ["3 medium@1024", "4 CLAHE+med", "1 small@1024", "2 adaptive"],
        "D medium 단독 + CLAHE":
            ["3 medium@1024", "4 CLAHE+med"],
        "E small → medium (adaptive 제거)":
            ["1 small@1024", "3 medium@1024", "4 CLAHE+med"],
        "F 현재 + 1600 두 칸 추가":
            BASE + ["6 medium@1600", "5 small@1600"],
        "G medium 먼저 + 1600 추가":
            ["3 medium@1024", "1 small@1024", "2 adaptive",
             "4 CLAHE+med", "6 medium@1600"],
    }

    rows = []
    for nm, order in CANDS.items():
        rows.append((nm, run(order), order))

    base = rows[0][1]

    L = ["# EDA 100 — 계단의 순서를 바꾸면\n",
         "99단계에서 칸을 늘리는 방향은 막혔다(오라클 상한 94.9%, 다수결은 오히려 하락).",
         "그런데 **medium 단독 89.1% > small 단독 86.8%** 이다. 순서를 바꿔 본다.",
         "",
         "칸별 결과를 재사용한 시뮬레이션이라 OCR 을 다시 돌리지 않았다.",
         "비용은 앞 칸이 성공하면 뒤 칸이 돌지 않는 것을 반영한 값이다.",
         "",
         "| 구성 | 필드 정확도 | 완전일치 | 장당(병렬 기준) | 현재 대비 |",
         "| --- | ---: | ---: | ---: | ---: |"]
    for nm, s, _ in rows:
        d = s["fa"] - base["fa"]
        mark = f"{d:+.1f}%p" if nm != rows[0][0] else "기준"
        L.append(f"| {nm} | **{s['fa']:.1f}%** | {s['ex']:.1f}% | "
                 f"{s['sec']:.2f}초 | {mark} |")
    L.append("")
    L.append("> 장당 시간은 4워커 병렬에서 잰 값이라 채점 조건(순차)보다 크다.")
    L.append("> 구성 간 **상대 비교**로만 읽을 것. 단일 프로세스 실측은 따로 한다.")
    L.append("")

    L.append("## 칸별 처리 분담\n")
    for nm, s, order in rows[:4]:
        tot = sum(s["used"].values())
        part = " · ".join(f"{SHORT[k]} {v}" for k, v in s["used"].items() if v)
        L.append(f"- **{nm.split(' ')[0]}**: {part} (온전한 날짜 {tot}/{n})")
    L.append("")

    best = max(rows, key=lambda x: x[1]["fa"])
    L.append("## 판단\n")
    if best[0] == rows[0][0]:
        L.append("**현재 순서가 이미 최선이다.** 순서를 바꿔 얻을 것이 없다.")
    else:
        d = best[1]["fa"] - base["fa"]
        r = best[1]["sec"] / base["sec"]
        L.append(f"가장 나은 것은 **{best[0]}** 로 {d:+.1f}%p, 비용은 {r:.1f}배다.")
        L.append("")
        L.append("다만 이 수치는 오염된 580장에서 나온 것이므로, 채택하려면")
        L.append("홀드아웃4 에서 한 번 더 확인해야 한다.")

    with open(os.path.join(OUT, "100_rung_order.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L[6:]))
    print("\n→ eda/out/100_rung_order.md")


if __name__ == "__main__":
    main()
