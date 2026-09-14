"""
EDA 60단계 — 이식한 두 규칙의 단위 검사

홀드아웃 실행은 400초가 걸린다. 그 전에 규칙 자체가 의도대로 도는지 본다.
합성 테스트는 **채택 근거가 될 수 없다**(TAIL_PATTERNS 는 합성 10/11 을 통과하고도
홀드아웃에서 떨어졌다). 여기서는 '코드가 의도대로 동작하는가'만 확인하고,
채택 여부는 61단계 홀드아웃 수치로만 판단한다.
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import pipeline as P

# 실제 EAN-13 (체크섬 유효)
KR = "8801043032001"     # 한국 880
DE = "4009900484169"     # 독일 400~440 → dmy
US = "0012000001291"     # 미국 0xx     → mdy

CASES_BARCODE = [
    (KR, None,  "한국(880) — 연-월-일 관례라 힌트 없음"),
    (DE, "dmy", "독일(400)"),
    (US, "mdy", "미국(0xx)"),
    ("1234567890123", None, "체크섬 불일치 — 바코드로 인정하지 않음"),
    ("20130628332176", None, "품목보고번호 14자리 — 오인식하면 안 된다"),
]

CASES_TIME = [
    ("20.12.14 22:54", True,  "콜론 — 제조 타임스탬프"),
    ("20.12.14 22,54", True,  "콜론이 콤마로 오인식"),
    ("20.12.14 22*54", True,  "콜론이 별표로 오인식"),
    ("20.12.14",       False, "시각 없음 — 살려야 한다"),
    ("20.12.14 2019",  False, "뒤 숫자가 시각 형태가 아님"),
]


def main():
    print("=== 1. 바코드 국가 판별")
    ok = 0
    for code, want, why in CASES_BARCODE:
        got = P.barcode_country("상품 " + code + " 표시")
        mark = "OK " if got == want else "!! "
        ok += got == want
        print(f"  {mark}{code:<16} → {str(got):<5} (기대 {want})  {why}")

    print("\n=== 2. 날짜 뒤 시각 제외 (2자리 연도에만 적용)")
    ok2 = 0
    for text, should_drop, why in CASES_TIME:
        ds = P.find_dates(text)
        dropped = not ds
        mark = "OK " if dropped == should_drop else "!! "
        ok2 += dropped == should_drop
        got = ds[0] if ds else None
        shown = f"{got['y']}-{got['m']:02d}-{got['d']:02d}" if got else "없음"
        print(f"  {mark}{text:<18} → {shown:<12} {why}")

    print("\n=== 3. 바코드 힌트가 2자리 연도 해석을 바꾸는가")
    # 22.10.21 은 2022-10-21(한국식) 로도 2021-10-22(유럽식) 로도 읽힌다
    for code, tag in ((KR, "한국 바코드"), (DE, "독일 바코드"), (US, "미국 바코드"), ("", "바코드 없음")):
        t = "소비기한 22.10.21 " + code
        ds = P.find_dates(t)
        d = ds[0] if ds else None
        shown = f"{d['y']}-{d['m']:02d}-{d['d']:02d}" if d else "없음"
        print(f"  {tag:<12} → {shown}")

    print("\n=== 4. 4자리 연도에는 시각 규칙이 걸리지 않아야 한다")
    # 실제 정답: 000826 은 '2026.06.22 17:34' 이고 정답이 2026-06-22 다
    ds = P.find_dates("소비기한 2026.06.22 17:34")
    d = ds[0] if ds else None
    shown = f"{d['y']}-{d['m']:02d}-{d['d']:02d}" if d else "없음"
    mark = "OK " if shown == "2026-06-22" else "!! "
    print(f"  {mark}2026.06.22 17:34 → {shown} (기대 2026-06-22)")

    print(f"\n바코드 {ok}/{len(CASES_BARCODE)}, 시각 {ok2}/{len(CASES_TIME)}")
    print("※ 이 결과는 '코드가 도는가'만 말해준다. 채택 판정은 61단계 홀드아웃으로 한다.")


if __name__ == "__main__":
    main()
