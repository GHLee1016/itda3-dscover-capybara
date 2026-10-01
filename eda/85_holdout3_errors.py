"""
EDA 85단계 — 홀드아웃3에서 남은 오답의 구성

83단계가 낸 수치(대표 147장 91.6%)의 나머지 8.4%가 무엇인지 본다.
여기서 정하는 것은 "다음에 무엇을 고칠까"이고, 그 판단은 이 표본이 아니라
**원인별 장수**로 한다. 표본에 맞춰 고치면 다시 오염된다.

같이 뽑는 것
  · 연도 규칙 변경(B→C)으로 **나빠진** 장. 한국식 기본이 틀리는 경우가
    어떤 모양인지 알아야 이 규칙을 더 밀지 말지 정할 수 있다.
  · 재시도(A→B)로도 못 살린 미검출 7장

    python eda/85_holdout3_errors.py
"""
import os
import sys
import json
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "eda", "out")


def s3(v):
    if not v or v == "NONE":
        return ("NONE",) * 3
    q = str(v).split("-")
    return tuple(q) if len(q) == 3 else ("NONE",) * 3


def kind(g, p):
    """오답 한 장의 모양을 한 마디로."""
    if p == ("NONE",) * 3:
        return "미검출"
    if g == ("NONE",) * 3:
        return "오탐"
    hit = [g[i] == p[i] for i in range(3)]
    if hit == [False, True, True]:
        return "연도만 틀림"
    if hit == [True, False, False] and g[1] == p[2] and g[2] == p[1]:
        return "월일 뒤바뀜"
    if hit == [False, False, False] and g[0] == p[2] and g[2] == p[0]:
        return "연일 뒤바뀜"
    if sum(hit) == 2:
        return "한 필드 오독"
    if "NONE" in p:
        return "일부만 인식"
    return "전부 다름"


def main():
    lab = json.load(open(os.path.join(OUT, "truth_holdout3.json"),
                        encoding="utf-8"))["labels"]
    pred = json.load(open(os.path.join(OUT, "83_holdout3.json"), encoding="utf-8"))
    split = json.load(open(os.path.join(OUT, "80_holdout3_split.json"),
                           encoding="utf-8"))
    rep = [f for f in split["representative"] if f in lab]
    tgt = [f for f in split["targeted"] if f in lab]

    A, B, C = pred["A_예선본"], pred["B_계단"], pred["C_최종"]

    L = ["# EDA 85 — 홀드아웃3에서 남은 오답의 구성\n",
         "83단계의 대표 147장 91.6% 에서 나머지가 무엇인지 본다.",
         "다음 수정 대상은 이 목록이 아니라 **원인별 장수**로 정한다.",
         "표본에 맞춰 고치면 홀드아웃3 도 홀드아웃2 처럼 오염된다.",
         ""]

    for title, subset in (("대표 147장", rep), ("표적 50장", tgt)):
        wrong = [f for f in subset if s3(lab[f]) != s3(C[f])]
        cnt = Counter(kind(s3(lab[f]), s3(C[f])) for f in wrong)
        L.append(f"## {title} — 오답 {len(wrong)}장\n")
        L.append("| 모양 | 장수 | 표본 대비 |")
        L.append("| --- | ---: | ---: |")
        for k, v in cnt.most_common():
            L.append(f"| {k} | {v} | {v/len(subset)*100:.1f}% |")
        L.append("")
        L.append("| 파일 | 정답 | 출력 | 모양 |")
        L.append("| --- | --- | --- | --- |")
        for f in sorted(wrong, key=lambda x: kind(s3(lab[x]), s3(C[x]))):
            L.append(f"| `{f}` | {lab[f]} | {C[f]} | {kind(s3(lab[f]), s3(C[f]))} |")
        L.append("")

    # 연도 규칙 때문에 나빠진 장 — 이 규칙을 더 밀지 말지의 근거
    def score(g, p):
        return sum(g[i] == p[i] for i in range(3))

    L.append("## 2자리 연도 한국식 기본이 **틀린** 장\n")
    L.append("B(유럽식) 가 맞았는데 C(한국식) 가 틀린 경우다. 이 규칙의 대가다.")
    L.append("")
    L.append("| 파일 | 집합 | 정답 | 유럽식 | 한국식 |")
    L.append("| --- | --- | --- | --- | --- |")
    worse = 0
    for f in rep + tgt:
        g = s3(lab[f])
        if score(g, s3(C[f])) < score(g, s3(B[f])):
            worse += 1
            grp = "대표" if f in set(rep) else "표적"
            L.append(f"| `{f}` | {grp} | {lab[f]} | {B[f]} | {C[f]} |")
    if worse == 0:
        L.append("| (없음) | | | | |")
    L.append("")
    better = sum(1 for f in rep + tgt
                 if score(s3(lab[f]), s3(C[f])) > score(s3(lab[f]), s3(B[f])))
    L.append(f"전체 197장에서 한국식 기본: 개선 {better}장 / 악화 {worse}장")
    L.append("")

    # 재시도로도 못 살린 미검출
    miss = [f for f in rep + tgt if s3(C[f]) == ("NONE",) * 3
            and s3(lab[f]) != ("NONE",) * 3]
    L.append(f"## 계단을 다 거치고도 못 읽은 {len(miss)}장\n")
    L.append("| 파일 | 정답 | 예선본 | 재시도 후 |")
    L.append("| --- | --- | --- | --- |")
    for f in miss:
        L.append(f"| `{f}` | {lab[f]} | {A[f]} | {C[f]} |")
    L.append("")

    with open(os.path.join(OUT, "85_holdout3_errors.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
