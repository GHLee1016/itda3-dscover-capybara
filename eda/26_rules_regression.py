"""
EDA 26단계 — 운영진 Q&A 반영 후 회귀 검증

새로 추가한 것
  - 연도 후행형 (21.02.2026 → 일/월/년)
  - 연도 없이 월·일만 (NONE / 08 / 25)
  - 부분 출력 (year/month/day 독립)

기존 케이스가 깨지지 않았는지 함께 확인한다.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import pipeline as P

# (설명, 텍스트, 기대 year/month/day)  None 은 NONE 을 뜻한다
CASES = [
    # ---- 기존 (깨지면 안 됨) ----
    ("표준", "소비기한 2027.06.26 까지  LOT NO 543123", (2027, 6, 26)),
    ("품목보고번호 함정", "품목보고번호 20130628332176 소비자상담실 1899-1494", None),
    ("시각 혼입", "소비기한: 2027.04.14 15:55 까지", (2027, 4, 14)),
    ("2자리연도", "유통기한 21.02.11 까지F1", (2021, 2, 11)),
    ("제조+소비 동시", "제조(년월일시) 25.12.08.08시  소비기한 25.12.09.22시까지", (2025, 12, 9)),
    ("영문 월", "Best Before  29/SEP/2022   0000582504 - 273  22:07", (2022, 9, 29)),
    ("일본어", "賞味期限(未開栓)  2022.07.16", (2022, 7, 16)),
    ("슬래시", "Best before end: 2022/11/04  8990800002675", (2022, 11, 4)),
    ("앵커없음 단일날짜", "2021.12.05   1577-7966  300ml  830kcal", (2021, 12, 5)),
    ("한글연월일", "소비기한 2026년 5월 29일까지", (2026, 5, 29)),
    ("바코드만", "8801052032368  8 801759 002848", None),
    ("LOT 오탐 차단", "유통기한 410414 까지", None),

    # ---- 신규: 연도 후행형 (운영진: 일/월/년) ----
    ("연도후행 DD/MM/YYYY", "Best before 21.02.2026", (2026, 2, 21)),
    ("연도후행 일>12", "THT 25/12/2026", (2026, 12, 25)),
    ("연도후행 월>12 불가", "EXP 03/25/2026", (2026, 3, 25)),

    # ---- 신규: 연도 없이 월·일만 ----
    ("월일만", "소비기한 08.25 까지", (None, 8, 25)),
    ("월일만 하이픈", "유통기한 10-14", (None, 10, 14)),
    ("시각은 월일 아님", "제조시각 10:14", None),
    ("완전날짜 우선", "소비기한 2026.05.29  포장 08.25", (2026, 5, 29)),

    # ---- 신규: 연·월만 (실측 002630) ----
    ("월연만 MM.YYYY", "05.2023  D9482", (2023, 5, None)),
    ("연월만 YYYY.MM", "賞味期限 2023.05", (2023, 5, None)),

    # ---- 신규: 제조일만 있으면 NONE (실측 002311) ----
    ("제조일만", "베이킹파우더 150g  제조2020.08.17", None),
    ("제조+소비 있으면 정상", "제조2020.08.17  소비기한 2021.08.16까지", (2021, 8, 16)),
    ("제조표시 없는 단일날짜", "2021.12.05   300ml", (2021, 12, 5)),
]


def run(text):
    cands = {}
    for d in P.find_dates(text):
        cands.setdefault("%04d-%02d-%02d" % (d["y"], d["m"], d["d"]), d["kind"])
    return P.select_date(cands, P.find_anchors(text),
                         P.find_month_day(text), P.find_year_month(text), text)


def main():
    ok = bad = 0
    print(f"{'':2} {'케이스':<22}{'기대':<16}{'실제':<16}")
    print("-" * 60)
    for name, text, exp in CASES:
        got = run(text)
        row = P.to_row("x", got)
        actual = (got.get("y"), got.get("m"), got.get("d")) if got else None
        hit = actual == exp
        ok += hit
        bad += not hit
        e = "NONE" if exp is None else "/".join(str(v) if v else "NONE" for v in exp)
        a = "NONE" if actual is None else "/".join(str(v) if v else "NONE" for v in actual)
        print(f"{'O' if hit else 'X'}  {name:<22}{e:<16}{a:<16}"
              f"{'' if hit else '  <- ' + text[:44]}")
        if hit and got:
            # 제출 형식도 확인
            assert row["month"] == "NONE" or len(row["month"]) == 2
            assert row["day"] == "NONE" or len(row["day"]) == 2
    print("-" * 60)
    print(f"통과 {ok}/{len(CASES)}   실패 {bad}")

    print("\n제출 형식 예시")
    for name, text, _ in CASES[:1] + CASES[-4:-2]:
        r = P.to_row("000001", run(text))
        print(f"  {name:<22} {r['year']} / {r['month']} / {r['day']} / {r['final_date']}")
    r = P.to_row("000002", None)
    print(f"  {'전부 미인식':<22} {r['year']} / {r['month']} / {r['day']} / {r['final_date']}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
