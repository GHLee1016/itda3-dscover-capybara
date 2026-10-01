"""
EDA 78단계 — '값을 냈는데 틀린' 오답의 유형 분류

76단계 이후 손실 구성이 뒤집혔다.
    미검출 9장 × 3필드 = 27필드 =  4.5%p
    **값을 냈는데 틀림          =  8.3%p**   ← 이제 이쪽이 더 크다

지금까지 미검출만 쫓느라 오독은 한 번도 들여다보지 않았다. 69단계에서 실패를
원인별로 갈랐던 것과 같은 절차를 오답에 적용한다.

분류 축
    year_only      연·월·일 중 연도만 틀림      → 2자리 연도 해석, 20xx 오독
    md_only        월/일만 틀림                 → 자릿수 오독, 월일 순서
    mfg_picked     정답보다 이른 날짜를 냈다     → 제조일자를 고른 정황
    partial_short  정답은 완전한데 일부만 냈다   → 회수 부족
    partial_over   정답이 부분인데 더 채웠다     → 과잉 추정
    other          그 밖

라벨이 있어야 오답을 알 수 있으므로 홀드아웃2 를 쓴다. 다만 여기서 얻는 것은
**수정 방향**이고, 수정 후 검증은 별도 표본에서 해야 한다(같은 표본으로
고치고 재면 그 표본에 맞춘 것이 된다).

    python eda/78_wrong_value_diag.py
"""
import os
import sys
import json
import argparse
from collections import Counter
from datetime import date

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "eda", "out")


def s3(v):
    if not v or v == "NONE":
        return ("NONE",) * 3
    q = str(v).split("-")
    return tuple(q) if len(q) == 3 else ("NONE",) * 3


def as_date(t):
    try:
        return date(int(t[0]), int(t[1]), int(t[2]))
    except (ValueError, TypeError):
        return None


def classify(g, p):
    gn = g == ("NONE",) * 3
    pn = p == ("NONE",) * 3
    if pn and not gn:
        return "미검출"
    if gn and not pn:
        return "오탐(정답 NONE)"

    g_set = [i for i in range(3) if g[i] != "NONE"]
    p_set = [i for i in range(3) if p[i] != "NONE"]

    # 값이 있는 칸끼리 비교
    both = [i for i in g_set if i in p_set]
    wrong = [i for i in both if g[i] != p[i]]

    if len(p_set) < len(g_set) and not wrong:
        return "partial_short(덜 채움)"
    if len(p_set) > len(g_set) and not wrong:
        return "partial_over(더 채움)"

    if wrong == [0]:
        return "year_only(연도만 틀림)"
    if wrong and 0 not in wrong:
        return "md_only(월/일만 틀림)"
    if wrong:
        gd, pd = as_date(g), as_date(p)
        if gd and pd and pd < gd:
            return "mfg_picked(더 이른 날짜)"
        return "other(여러 필드)"
    return "other(여러 필드)"


def main():
    argparse.ArgumentParser().parse_args()

    lab = json.load(open(os.path.join(OUT_DIR, "truth_holdout2.json"),
                         encoding="utf-8"))["labels"]
    preds = json.load(open(os.path.join(OUT_DIR, "76_escalation.json"),
                           encoding="utf-8"))["escalation"]
    files = [f for f in lab if f in preds]

    rows = []
    ok = tot = 0
    for f in files:
        g, p = s3(lab[f]), s3(preds[f])
        hit = sum(g[i] == p[i] for i in range(3))
        ok += hit
        tot += 3
        if g != p:
            rows.append((f, lab[f], preds[f], classify(g, p), hit))

    n = len(files)
    cnt = Counter(r[3] for r in rows)
    lost = {k: sum(3 - r[4] for r in rows if r[3] == k) for k in cnt}
    total_lost = tot - ok

    L = ["# EDA 78 — '값을 냈는데 틀린' 오답의 유형 분류\n",
         "76단계 이후 손실 구성이 뒤집혔다. 미검출보다 오독이 더 크다.",
         "지금까지 미검출만 쫓느라 오독은 들여다보지 않았다.",
         "",
         f"- 홀드아웃2 {n}장, 필드 정확도 {ok/tot*100:.1f}%",
         f"- 틀린 이미지 {len(rows)}장, 잃은 필드 {total_lost}개 / 전체 {tot}개",
         "",
         "## 오답 유형별 손실\n",
         "| 유형 | 이미지 | 잃은 필드 | 전체 손실 중 | 전체 정확도에서 |",
         "| --- | ---: | ---: | ---: | ---: |"]
    for k, v in sorted(cnt.items(), key=lambda kv: -lost[kv[0]]):
        L.append(f"| {k} | {v} | {lost[k]} | {lost[k]/total_lost*100:.1f}% | "
                 f"{lost[k]/tot*100:.1f}%p |")
    L.append("")

    # 유형별 사례
    L.append("## 유형별 사례\n")
    for k, _ in sorted(cnt.items(), key=lambda kv: -lost[kv[0]]):
        sub = [r for r in rows if r[3] == k]
        L.append(f"### {k} — {len(sub)}장\n")
        L.append("| 파일 | 정답 | 출력 |")
        L.append("| --- | --- | --- |")
        for f, g, p, _, _h in sub[:12]:
            L.append(f"| `{f}` | {g} | {p} |")
        if len(sub) > 12:
            L.append(f"| … | ({len(sub)-12}장 더) | |")
        L.append("")

    # 연도 오독의 세부 — 몇 년 차이인지
    yr = [r for r in rows if r[3].startswith("year_only")]
    if yr:
        diffs = Counter()
        for f, g, p, _, _h in yr:
            gy, py = s3(g)[0], s3(p)[0]
            try:
                diffs[int(py) - int(gy)] += 1
            except ValueError:
                diffs["비수치"] += 1
        L.append("## 연도 오독의 차이\n")
        L.append("| 출력 − 정답 | 장수 |")
        L.append("| ---: | ---: |")
        for d, c in sorted(diffs.items(), key=lambda kv: str(kv[0])):
            L.append(f"| {d:+d}년 |" .replace("+비수치", "비수치") + f" {c} |"
                     if isinstance(d, int) else f"| {d} | {c} |")
        L.append("")
        L.append("`-1`/`+1` 이 많으면 숫자 오독이고, `-5` 이상이면 제조일자를")
        L.append("고른 정황이다. 대응이 다르다.")
        L.append("")

    L.append("## 다음 수정 방향\n")
    top = max(cnt, key=lambda k: lost[k])
    L.append(f"가장 큰 손실원은 **{top}** ({lost[top]}필드, 전체 정확도 {lost[top]/tot*100:.1f}%p).")
    L.append("")
    L.append("| 유형 | 접근 |")
    L.append("| --- | --- |")
    L.append("| year_only | 2자리 연도 해석 규칙, 검출 박스를 원본 해상도로 크롭해 재인식 |")
    L.append("| md_only | 자릿수 오독. 크롭 재인식이 직접적 |")
    L.append("| mfg_picked | 키워드 가중 조정. 규칙 수정으로 접근 |")
    L.append("| partial_short | 회수 부족. 검출 쪽 문제와 겹친다 |")
    L.append("| partial_over | 과잉 추정. 규칙을 더 보수적으로 |")
    L.append("")
    L.append("> ⚠️ 여기서 얻은 방향으로 고친 뒤 **같은 홀드아웃2 로 재면 안 된다.**")
    L.append("> 그 표본에 맞춘 정도가 성능으로 보인다. 새 표본이 필요하다.")

    with open(os.path.join(OUT_DIR, "78_wrong_value_diag.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L[3:20]))
    print("\n→ eda/out/78_wrong_value_diag.md")


if __name__ == "__main__":
    main()
