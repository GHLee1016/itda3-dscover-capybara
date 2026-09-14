"""
소비기한 추출 파이프라인.

설계 근거는 전부 실측이며 METHOD.md / eda/out/ 에 기록되어 있다.

  det 입력 960px 고정      3200px 까지 올려도 recall 불변, 비용만 증가 (EDA 5)
  0° → 실패 시 90°         방향 판별기는 무의미(+0.2%p), 합집합이 이득 (EDA 4)
  인접 박스 병합            det 가 날짜 한 줄을 쪼갠다. 정확도 48%→68% (EDA 19)
  7자리 느슨 패턴 미사용     coverage 는 오르지만 정확도는 그대로 (EDA 19)
  4자리 연도 우선           정밀도 99.2% vs 2자리 79.4% (EDA 4)
  제조/소비 쌍은 늦은 날짜   '부터/까지' 및 제조 앵커 사례에서 전부 성립 (EDA 19)
  미래성 필터 금지          정답에 2020~2022년 과거 날짜가 다수 (EDA 1)
"""
import os
import re
import time
import unicodedata
from datetime import date

import numpy as np
import cv2

# ----------------------------------------------------------------- 앵커
ANCHORS = {
    "소비기한": ["소비기한", "소비기 한", "소비기힌", "비기한"],
    "유통기한": ["유통기한", "유통기 한", "유통기힌", "통기한"],
    "품질유지기한": ["품질유지기한", "품질유지"],
    "제조일자": ["제조일자", "제조년월일", "제조일", "제조(년", "제조:"],
    "생산일자": ["생산일자", "생산일"],
    "포장일자": ["포장일자", "포장일", "포장년월일"],
    "賞味期限": ["賞味期限", "賞味期"],
    "消費期限": ["消費期限", "消費期"],
    "BEST_BEFORE": ["bestbefore", "bestbefor", "bestbe", "bbe", "bestbefend"],
    "EXP": ["exp.", "expiry", "expdate", "expirydate", "useby", "usebefore"],
    "MFG": ["mfg", "mfd", "manufactur", "proddate", "productiondate"],
    "까지": ["까지"],
    "부터": ["부터"],
    "별도표기": ["별도표기", "별도표시", "표기일까지", "하단표기", "상단표기",
                "제품별도", "별도기재", "우측표기", "전면표기", "표시일까지",
                "표시된일자", "아래에표시", "후면표시"],
}
POSITIVE = ("소비기한", "유통기한", "품질유지기한", "賞味期限", "消費期限", "BEST_BEFORE", "EXP")
NEGATIVE = ("제조일자", "생산일자", "포장일자", "MFG")

EN_MONTH = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
            "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12}
_MON_KEYS = {v: k for k, v in
             {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
              "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}.items()}
_CONFUSE = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s",
                          "6": "g", "8": "b", "9": "g"})


def norm(s):
    s = unicodedata.normalize("NFKC", s or "")
    return re.sub(r"[\s·:;,_()\[\]/\\|]+", "", s).lower()


def find_anchors(text):
    n = norm(text)
    hit = set()
    for key, pats in ANCHORS.items():
        for p in pats:
            if norm(p) and norm(p) in n:
                hit.add(key)
                break
    return hit


# ----------------------------------------------------------------- 날짜 패턴
# OCR 은 잉크젯 날짜의 구분자를 자주 통째로 날린다 ('21.02.11' → '210211').
S = r"[^\d\n]{0,2}"
PATTERNS = [
    ("Y4_sep", re.compile(
        r"(?<![\d])(?P<y>19\d{2}|20[0-4]\d)" + S +
        r"(?P<m>1[0-2]|0[1-9])" + S + r"(?P<d>3[01]|[12]\d|0[1-9])(?![\d])")),
    ("Y4_sep1", re.compile(
        r"(?<![\d])(?P<y>19\d{2}|20[0-4]\d)\s*[.\-/년]\s*(?P<m>1[0-2]|[1-9])\s*[.\-/월]\s*"
        r"(?P<d>3[01]|[12]\d|[1-9])\s*일?(?![\d])")),
    ("Y2_sep", re.compile(
        r"(?<![\d])(?P<y>[0-4]\d)" + S +
        r"(?P<m>1[0-2]|0[1-9])" + S + r"(?P<d>3[01]|[12]\d|0[1-9])(?![\d])")),
    ("D_MON_Y", re.compile(
        r"(?<![\d])(?P<d>3[01]|[12]\d|0?[1-9])[^\dA-Za-z]{0,2}(?P<mon>[A-Za-z0-9]{2,5})"
        r"[^\dA-Za-z]{0,2}(?P<y>19\d{2}|20[0-4]\d)(?![\d])")),
    ("MON_D_Y", re.compile(
        r"(?<![A-Za-z])(?P<mon>[A-Za-z]{3,4})[^\dA-Za-z]{0,2}(?P<d>3[01]|[12]\d|0?[1-9])"
        r"[^\dA-Za-z]{0,2}(?P<y>19\d{2}|20[0-4]\d)(?![\d])")),
    # 연도가 맨 뒤에 오는 형식 (21.02.2026). 운영진 답변: 일/월/년 으로 해석.
    # 앞 패턴들이 먼저 잡으므로 여기까지 오는 건 연도 후행형뿐이다.
    ("D_M_Y4", re.compile(
        r"(?<![\d])(?P<a>3[01]|[12]\d|0?[1-9])" + S +
        r"(?P<b>3[01]|[12]\d|0?[1-9])" + S + r"(?P<y>19\d{2}|20[0-4]\d)(?![\d])")),
]
LONG_DIGITS = re.compile(r"\d{9,}")

# 연도 없이 월·일만 찍힌 경우 (운영진 확정: NONE / 08 / 25 / NONE-08-25).
# 오탐이 쉬우므로 **구분자 필수 + 2자리 zero-pad** 로 좁히고,
# 완전한 날짜 후보가 하나도 없을 때만 사용한다.
M_D_ONLY = re.compile(
    r"(?<![\d])(?P<m>1[0-2]|0[1-9])\s*[.\-/]\s*(?P<d>3[01]|[12]\d|0[1-9])(?![\d.\-/:])")

# 일 없이 연·월만 찍힌 경우 (수입품에 흔하다: '05.2023', '2023.05').
# 실측 사례 002630 — 독일 제품에 '05.2023  D9482' 만 인쇄되어 있었다.
Y_M_ONLY = [
    re.compile(r"(?<![\d])(?P<m>1[0-2]|0[1-9])\s*[.\-/]\s*(?P<y>19\d{2}|20[0-4]\d)(?![\d])"),
    re.compile(r"(?<![\d])(?P<y>19\d{2}|20[0-4]\d)\s*[.\-/년]\s*(?P<m>1[0-2]|0[1-9])"
               r"\s*월?(?![\d.\-/])"),
]

# 날짜 바로 앞에 붙은 '제조' 계열 표시. 이게 붙은 날짜는 소비기한이 아니다.
# 실측 사례 002311 — '제조2020.08.17' 만 있고 소비기한이 없어 정답은 NONE.
MFG_PREFIX = re.compile(
    r"(제\s*조|생\s*산|포\s*장|mfg|mfd|prod|born)[^\d]{0,6}$", re.I)


def fuzzy_month(tok):
    t = re.sub(r"[^a-z0-9]", "", (tok or "").lower())
    if not t or t.isdigit():
        return None
    t2 = t.translate(_CONFUSE)
    for num, name in _MON_KEYS.items():
        if name in t or name in t2:
            return num
    for num, name in _MON_KEYS.items():
        if len(t2) >= 3 and sum(a == b for a, b in zip(t2[:3], name)) >= 2:
            return num
    return None


def _mk(y, m, d, kind, span):
    try:
        y, m, d = int(y), int(m), int(d)
    except (TypeError, ValueError):
        return None
    if y < 100:
        y = 2000 + y
    if not (1990 <= y <= 2049 and 1 <= m <= 12 and 1 <= d <= 31):
        return None
    try:
        date(y, m, d)
    except ValueError:
        return None
    return {"y": y, "m": m, "d": d, "kind": kind, "span": span}


# ---------------------------------------------------------------- 팀원 규칙 이식
# 팀원(classical CV + EasyOCR) 파이프라인에서 가져온 두 규칙.
# 엔진 자체를 합치는 것은 이득이 0%p 라 기각했고(EDA 59), 아래 두 규칙은
# 추론 비용이 0이고 torch 도 필요 없어 따로 검증했다.
#
# ⚠️ 둘 다 **기각**됐다 (홀드아웃2 198장, EDA 61·62).
#
#   바코드 국가 힌트: 부분점수 66.8% → 66.8%, 바뀐 이미지 0장.
#     판단에 영향을 줄 수 있는 이미지가 **0장**이었다. 체크섬이 유효한 EAN-13 을
#     읽어낸 것 자체가 1장(0.5%)뿐이다 — 바코드는 한 자리만 오인식해도 체크섬이
#     깨지는데 OCR 이 13자리를 다 맞히는 일이 거의 없다. 실전에서 죽은 코드다.
#     반면 잘못 걸리면 한국 날짜를 유럽식으로 뒤집어 연·일을 한꺼번에 틀린다.
#
#   날짜 뒤 시각 제외: 66.8% → 66.8%. 2장에서 발동했고 결과를 바꾸지 않았다.
#     무해함은 확인됐지만 n=2 로는 이득의 증거가 못 된다.
#
# 검증된 구성을 이득 0 인 변경으로 건드리지 않는다. 코드와 근거는 남긴다.
ENABLE_BARCODE_HINT = False   # EAN-13 국가 접두어로 2자리 연도 순서 판별
ENABLE_TIME_EXCLUDE = False   # 날짜 뒤 '시:분' 이 붙으면 제조 시각으로 보고 제외

_DIGIT_RUN_ANY = re.compile(r"\d+")

# 날짜 바로 뒤의 시각 표기. 콜론이 콤마·별표로 오인식되는 경우가 잦아 함께 인정한다.
# 마침표(.)는 진짜 날짜 뒤에 다른 숫자가 오는 경우와 구분되지 않아 제외한다.
_TIME_AFTER = re.compile(r"\s*\d{1,2}\s*[:,;*]\s*\d{2}(?!\d)")


def _ean13_ok(code):
    """EAN-13 체크섬. 전화번호·품목보고번호 같은 임의의 13자리를 걸러낸다."""
    if len(code) != 13 or not code.isdigit():
        return False
    dg = [int(c) for c in code]
    total = sum(d if i % 2 == 0 else d * 3 for i, d in enumerate(dg[:-1]))
    return (10 - total % 10) % 10 == dg[-1]


def barcode_country(text):
    """
    제품에 인쇄된 EAN-13 바코드의 GS1 국가 접두어로 **날짜 표기 순서**만 추정한다.

    2자리 연도(`22.10.21`)는 한국식 YY.MM.DD 로도, 유럽식 DD.MM.YY 로도 읽힌다.
    지금은 한글/라틴 문자 비율(latin_dominant)로 판단하는데, 포장지에 한글
    수입표시가 붙어 있으면 신호가 흐려진다. 바코드는 그보다 직접적인 근거다.

    반환: "mdy"(미국·캐나다) / "dmy"(유럽 등) / None(불명 또는 연-월-일 관례 국가)

    ⚠️ 숫자열 길이가 **정확히 13** 일 때만 본다. 더 긴 숫자열 안을 슬라이딩하면
       품목보고번호(20130628332176, 14자리)의 부분열이 우연히 체크섬을 통과한다
       (실측 — EDA 60에서 `2013062833217` 이 prefix 201 → "dmy" 로 잡혔다).
       품목보고번호는 국내 상품에 거의 항상 있고, 잘못 잡히면 한국 날짜를
       유럽식으로 뒤집는 사고가 난다. 진짜 바코드 몇 개를 놓치더라도 정밀도를 택한다.
       (팀원 코드는 8자리 이상 숫자열을 슬라이딩해서 이 문제가 남아 있다)
    """
    if not text:
        return None
    for run in _DIGIT_RUN_ANY.findall(text):
        if len(run) != 13 or not _ean13_ok(run):
            continue
        p = int(run[:3])
        if p <= 139 or 754 <= p <= 755:
            return "mdy"
        if p == 880 or 450 <= p <= 459 or 490 <= p <= 499 or 690 <= p <= 699:
            return None          # 한국·일본·중국 — 연-월-일이라 참고할 필요가 없다
        if 200 <= p <= 899:
            return "dmy"
    return None


def latin_dominant(text):
    """
    한글이 거의 없는 텍스트인가 (수입품 판별).

    2자리 연도 날짜의 순서가 나라마다 다르다.
      한국    YY.MM.DD   (25.12.09)
      유럽    DD.MM.YY   (09.12.25)
    둘 다 타당할 때 어느 쪽으로 읽을지 이 신호로 정한다.
    """
    if not text:
        return False
    han = sum("가" <= c <= "힣" for c in text)
    lat = sum(("a" <= c.lower() <= "z") for c in text)
    return lat > han * 2


# 날짜 바로 뒤에 시각·코드가 숫자로 붙어버린 경우를 위한 보조 패턴.
#   '202106081가'  '25102511:0F2'  '202104:190'
# 엄격 패턴은 뒤 경계 (?!\d) 때문에 이런 걸 통째로 놓친다. 그렇다고 경계를 없애면
# 품목보고번호(20130628332176) 앞 8자리가 날짜로 잡힌다.
# → **엄격 파싱이 아무것도 못 찾았을 때만** 쓰고, 숫자열 길이 상한으로 막는다.
TAIL_PATTERNS = [
    ("Y4_tail", re.compile(
        r"(?<![\d])(?P<y>20[0-4]\d)" + S + r"(?P<m>1[0-2]|0[1-9])" + S +
        r"(?P<d>3[01]|[12]\d|0[1-9])")),
    ("Y2_tail", re.compile(
        r"(?<![\d])(?P<y>[0-4]\d)" + S + r"(?P<m>1[0-2]|0[1-9])" + S +
        r"(?P<d>3[01]|[12]\d|0[1-9])")),
]
# 이 길이를 넘는 연속 숫자열 안에서는 보조 패턴을 쓰지 않는다.
# 바코드 13자리, 품목보고번호 14자리 이상이 여기서 걸러진다.
TAIL_MAX_RUN = 10
DIGIT_RUN = re.compile(r"\d+")

# ⚠️ 보조 패스는 **기각**됐다 (홀드아웃 200장 실측).
#      미검출 55 → 51장으로 4장 줄었으나 부분점수는 64.5% → 63.8% 로 떨어졌다.
#      새로 잡은 후보 중 틀린 날짜가 섞여 맞던 것을 밀어냈기 때문이다.
#      합성 테스트(10/11)와 회귀(24/24)는 통과했는데 실제 데이터에서는 손해였다.
#      '못 읽던 걸 읽게 만드는 것'이 곧 '맞게 읽는 것'은 아니다.
#      코드는 근거와 함께 남겨두되 기본은 끈다.
ENABLE_TAIL_PASS = False


def find_dates(text):
    """텍스트에서 날짜 후보를 뽑는다. 겹치는 매치는 앞선 패턴 우선."""
    if not text:
        return []
    long_spans = [mm.span() for mm in LONG_DIGITS.finditer(text)]
    latin = latin_dominant(text)
    bucket = barcode_country(text) if ENABLE_BARCODE_HINT else None
    out, taken = [], []
    for kind, rx in PATTERNS:
        for m in rx.finditer(text):
            s, e = m.span()
            if any(s < te and ts < e for ts, te in taken):
                continue
            gd = m.groupdict()
            if gd.get("mon"):
                mm = EN_MONTH.get(gd["mon"].lower().rstrip(".")) or fuzzy_month(gd["mon"])
                if mm is None:
                    continue
                rec = _mk(gd["y"], mm, gd["d"], kind, (s, e))
            elif kind == "D_M_Y4":
                # 연도 후행형. 두 수 중 12 를 넘는 쪽이 '일' 로 확정된다.
                # 둘 다 12 이하면 운영진 안내대로 일/월 순서로 읽는다.
                a, b = int(gd["a"]), int(gd["b"])
                if a > 12 and b <= 12:
                    dd, mm = a, b
                elif b > 12 and a <= 12:
                    mm, dd = a, b
                elif a <= 12 and b <= 12:
                    dd, mm = a, b          # 기본: 일/월/년
                else:
                    continue               # 둘 다 12 초과 — 날짜가 아니다
                rec = _mk(gd["y"], mm, dd, kind, (s, e))
            elif kind == "Y2_sep":
                # 2자리 연도는 앞에 올 수도(한국 YY.MM.DD) 뒤에 올 수도(유럽 DD.MM.YY) 있다.
                # 실측(홀드아웃 48장): '080921'(→2021-09-08), '020623'(→2023-06-02),
                # '281021'(→2021-10-28) 을 전부 한국식으로 읽어 틀렸다.
                # 날짜 뒤에 '시:분' 이 붙으면 생산 타임스탬프다. 소비기한은 보통
                # 시각까지 찍지 않는다. 2자리 연도에만 건다 — 4자리 연도에는
                # '2026.06.22 17:34' 처럼 소비기한에 시각이 붙은 실제 정답이 있다.
                if ENABLE_TIME_EXCLUDE and _TIME_AFTER.match(text, e):
                    continue
                a_, b_, c_ = int(gd["y"]), int(gd["m"]), int(gd["d"])
                ok_kr = 2015 <= 2000 + a_ <= 2035 and 1 <= c_ <= 31
                ok_eu = 2015 <= 2000 + c_ <= 2035 and 1 <= a_ <= 31
                # 미국식 MM.DD.YY — 바코드 힌트가 있을 때만 고려한다.
                ok_us = 2015 <= 2000 + c_ <= 2035 and 1 <= a_ <= 12 and 1 <= b_ <= 31
                if bucket == "mdy" and ok_us:
                    rec = _mk(c_, a_, b_, kind, (s, e))
                elif ok_kr and ok_eu:
                    # 둘 다 말이 되면: 바코드 국가 힌트 > 한글/라틴 비율
                    use_eu = True if bucket == "dmy" else latin
                    rec = (_mk(c_, b_, a_, kind, (s, e)) if use_eu
                           else _mk(a_, b_, c_, kind, (s, e)))
                else:
                    use_eu = ok_eu and not ok_kr
                    rec = (_mk(c_, b_, a_, kind, (s, e)) if use_eu
                           else _mk(a_, b_, c_, kind, (s, e)))
            else:
                rec = _mk(gd.get("y"), gd.get("m"), gd.get("d"), kind, (s, e))
            if rec is None:
                continue
            # 품목보고번호(20130628332176) 앞 8자리가 날짜로 파싱되는 것을 막는다
            if any(ls <= s and e <= le for ls, le in long_spans):
                continue
            out.append(rec)
            taken.append((s, e))
    if out or not ENABLE_TAIL_PASS:
        return out

    # ---- 보조 패스: 엄격 파싱이 아무것도 못 찾았을 때만 ----
    for kind, rx in TAIL_PATTERNS:
        for m in rx.finditer(text):
            s, e = m.span()
            # 이 매치가 속한 연속 숫자열이 너무 길면 코드·바코드다
            run = next((r for r in DIGIT_RUN.finditer(text)
                        if r.start() <= s < r.end()), None)
            if run and (run.end() - run.start()) > TAIL_MAX_RUN:
                continue
            # long_spans(9자리+) 검사는 여기서 하지 않는다 — 위의 자릿수 상한이
            # 같은 역할을 하고, 9~12자리(날짜+시각)는 오히려 살려야 하기 때문이다
            gd = m.groupdict()
            if kind == "Y2_tail":
                a_, b_, c_ = int(gd["y"]), int(gd["m"]), int(gd["d"])
                ok_kr = 2015 <= 2000 + a_ <= 2035 and 1 <= c_ <= 31
                ok_eu = 2015 <= 2000 + c_ <= 2035 and 1 <= a_ <= 31
                use_eu = latin if (ok_kr and ok_eu) else (ok_eu and not ok_kr)
                rec = (_mk(c_, b_, a_, kind, (s, e)) if use_eu
                       else _mk(a_, b_, c_, kind, (s, e)))
            else:
                rec = _mk(gd["y"], gd["m"], gd["d"], kind, (s, e))
            # 보조 패스는 오탐 위험이 크므로 타당 연도만 받는다
            if rec is None or rec["y"] not in PLAUSIBLE_YEARS:
                continue
            if any(s < te and ts < e for ts, te in taken):
                continue
            out.append(rec)
            taken.append((s, e))
    return out


# ----------------------------------------------------------------- 박스 병합
def _box_geom(b):
    q = np.asarray(b, dtype=float).reshape(4, 2)
    w = max(np.linalg.norm(q[0] - q[1]), np.linalg.norm(q[2] - q[3]))
    h = max(np.linalg.norm(q[0] - q[3]), np.linalg.norm(q[1] - q[2]))
    return q[:, 1].mean(), w, h, q[:, 0].min(), q[:, 0].max()


def merge_lines(lines, y_tol=0.6, gap_tol=2.0):
    """
    같은 행에 있고 가로로 인접한 박스들의 텍스트를 이어 붙인다.
    det 가 '2022/11/04' 를 '20221' + '104' 로 쪼개는 경우를 복구한다.
    추가 OCR 이 없어 추론 비용은 0. 정확도 48% → 68% (EDA 19).
    """
    items = []
    for txt, box in lines:
        if box is None:
            continue
        cy, w, h, x0, x1 = _box_geom(box)
        if h < 1:
            continue
        items.append({"t": txt, "cy": cy, "h": h, "x0": x0, "x1": x1})
    items.sort(key=lambda d: (d["cy"], d["x0"]))

    rows, used = [], [False] * len(items)
    for i, a in enumerate(items):
        if used[i]:
            continue
        row, used[i] = [a], True
        for j in range(i + 1, len(items)):
            if used[j]:
                continue
            b = items[j]
            if abs(b["cy"] - a["cy"]) > y_tol * max(a["h"], b["h"]):
                continue
            if not (0.5 <= b["h"] / a["h"] <= 2.0):
                continue
            row.append(b)
            used[j] = True
        row.sort(key=lambda d: d["x0"])
        rows.append(row)

    out = []
    for row in rows:
        if len(row) == 1:
            continue
        cur = [row[0]]
        for prev, nxt in zip(row, row[1:]):
            if nxt["x0"] - prev["x1"] <= gap_tol * max(prev["h"], nxt["h"]):
                cur.append(nxt)
            else:
                if len(cur) > 1:
                    out.append("".join(d["t"] for d in cur))
                cur = [nxt]
        if len(cur) > 1:
            out.append("".join(d["t"] for d in cur))
    return out


# ----------------------------------------------------------------- 선택 규칙
PLAUSIBLE_YEARS = range(2015, 2036)


def find_month_day(text):
    """연도 없이 월·일만 있는 경우를 뽑는다 (운영진 확정 케이스)."""
    out = []
    for m in M_D_ONLY.finditer(text or ""):
        mm, dd = int(m["m"]), int(m["d"])
        if 1 <= mm <= 12 and 1 <= dd <= 31:
            out.append((mm, dd))
    return out


def find_year_month(text):
    """일 없이 연·월만 있는 경우를 뽑는다 (수입품에 흔함)."""
    out = []
    for rx in Y_M_ONLY:
        for m in rx.finditer(text or ""):
            y, mm = int(m["y"]), int(m["m"])
            if 1 <= mm <= 12:
                out.append((y, mm))
    return out


def is_mfg_only(text, cands):
    """
    잡힌 날짜가 전부 '제조' 표시에 붙어 있는가.

    운영진: "주어진 이미지로 소비기한을 알 수 없으면 NONE".
    제조일만 찍힌 제품에 그 날짜를 소비기한으로 내면 오답이다 (실측 002311).
    단, 후보가 2개 이상이면 늦은 날짜 규칙이 작동하므로 적용하지 않는다.
    """
    if len(cands) != 1 or not text:
        return False
    only = next(iter(cands))
    y, m, d = only.split("-")
    # 앞뒤 숫자 경계를 막지 않으면 2자리 연도 패턴이 '2020' 안쪽('20')에 걸린다.
    pats = [r"(?<!\d)%s\D{0,2}%s\D{0,2}%s(?!\d)" % (y, m, d),
            r"(?<!\d)%s\D{0,2}%s\D{0,2}%s(?!\d)" % (y[2:], m, d)]
    found = False
    for pat in pats:
        for mt in re.finditer(pat, text):
            found = True
            before = text[max(0, mt.start() - 12):mt.start()]
            if not MFG_PREFIX.search(before):
                return False          # 제조 표시 없이 등장한 곳이 있다
    # 한 번도 매치되지 않았으면 판단 불가 → 건드리지 않는다
    return found


def needs_more(cands, anchors):
    """
    후보를 찾았어도 더 훑어야 하는가.

    `부터`/`까지` 가 함께 있거나 `제조`+소비기한 앵커가 함께 있으면 날짜가
    **두 개** 인쇄된 제품이다. 한 개만 찾고 멈추면 '부터'(또는 제조일)를
    집을 확률이 절반이다.

    실측(EDA 30): 선택 규칙 실패 15장 중 `002165` `001685` `001817` `001722`
    등이 전부 이 구조였다. 고른 게 틀린 게 아니라 **다른 하나를 못 읽은 것**이다.
    """
    if len(cands) >= 2:
        return False
    pair = ("부터" in anchors and "까지" in anchors)
    mfg_pair = bool(anchors & set(NEGATIVE)) and bool(anchors & set(POSITIVE))
    return pair or mfg_pair


def select_date(cands, anchors, md_cands=None, ym_cands=None, text=""):
    """
    후보 중 최종 1개를 고른다.

    cands:    {"YYYY-MM-DD": kind}   완전한 날짜
    md_cands: [(month, day), ...]    연도 없이 월·일만 잡힌 것

    반환: {"y": int|None, "m": int|None, "d": int|None} 또는 None

    운영진 확정(2026-09-12): year/month/day 가 각각 독립 채점되므로
    연도를 못 읽어도 월·일만 맞히면 그만큼 점수가 된다.
    """
    # 2자리 연도 패턴은 정밀도가 79.4% 뿐이다(4자리는 99.2%). 연도가 타당 범위를
    # 벗어나면 LOT 코드를 날짜로 오독한 것이므로 후보에서 완전히 제거한다.
    # (실측: '410414' → 2041-04-14 가 유일 후보로 채택되어 오답이 됐다)
    cands = {d: k for d, k in (cands or {}).items()
             if k.startswith("Y4") or int(d[:4]) in PLAUSIBLE_YEARS}

    # 유일한 날짜가 '제조' 표시에만 붙어 있으면 소비기한을 알 수 없는 것이다.
    # 부분 후보(월·일 / 연·월)도 같은 문자열에서 나오므로 함께 막는다.
    if cands and "별도표기" not in anchors and is_mfg_only(text, cands):
        return None

    if cands:
        def score(item):
            d, kind = item
            y = int(d[:4])
            s = 0.0
            s += 3.0 if kind.startswith("Y4") else 0.0   # 4자리 연도 정밀도 99.2%
            s += 2.0 if y in PLAUSIBLE_YEARS else -5.0
            return s

        ranked = sorted(cands.items(), key=lambda it: (-score(it), it[0]))
        top = score(ranked[0])
        tied = [d for d, k in ranked if score((d, k)) == top]
        # 동점이면 더 늦은 날짜 — 제조일/소비기한 쌍에서 소비기한을 집는다.
        # 운영진 답변으로 확인된 규칙이다.
        best = max(tied)
        y, m, d = best.split("-")
        return {"y": int(y), "m": int(m), "d": int(d)}

    # 완전한 날짜가 없을 때만 부분 후보를 쓴다 (오탐 위험이 크기 때문).
    # 연·월 쪽이 월·일 쪽보다 신뢰도가 높아 먼저 본다 — 4자리 연도가 박혀 있어
    # 우연히 맞을 확률이 훨씬 낮기 때문이다.
    if ym_cands:
        y, mm = max(ym_cands)
        return {"y": y, "m": mm, "d": None}
    if md_cands:
        # 여러 개면 가장 늦은 월·일 (소비기한 쪽일 가능성이 높다)
        mm, dd = max(md_cands)
        return {"y": None, "m": mm, "d": dd}
    return None


class TimeBudget:
    """
    남은 시간에 맞춰 박스 필터 강도를 스스로 조절한다.

    평가셋 장수가 비공개이므로 고정 필터는 둘 중 하나로 틀린다.
      - 약하게 걸면 데이터가 많을 때 타임아웃 → 정량 60점이 통째로 0
      - 세게 걸면 데이터가 적을 때 정확도를 공짜로 버린다 (실측 64.7% → 52.9%)

    그래서 처음에는 필터 없이 최고 정확도로 돌리다가, 예상 완주 시간이 예산을
    넘기면 그때부터 조인다. 작은 평가셋에서는 끝까지 필터가 걸리지 않는다.
    """
    # (long_side, keep, second_pass_frac) — 위에서부터 정확도 우선.
    #
    # 가장 강한 손잡이는 **입력 해상도**다. EDA 5 는 960px 이상만 봐서 놓쳤는데,
    # 960 아래로 내려가면 오히려 정확도가 오른다 (EDA 23/24, 사람 정답 34장):
    #
    #     960px  64.7%  1.350 s/장        576px  70.6%  0.826
    #     640px  73.5%  0.992             512px  70.6%  0.641
    #                                     448px  67.6%  0.499
    #                                     384px  52.9%  0.389  ← 급락
    #
    # 작게 넣으면 det 가 인접 글자를 한 박스로 묶어 날짜 줄이 쪼개지지 않는다.
    #
    # ⚠️ 운영진 확정(2026-09-12): 평가셋 **500장**, 제한 **40분 일괄**.
    #    장당 예산이 0.72초 → **4.8초** 로 6.7배 늘었다.
    #
    # ⚠️ 위 표의 '초/장' 은 **4프로세스 병렬 wall-clock 을 장수로 나눈 값**이다.
    #    즉 처리량이지 단일 프로세스 지연이 아니다. 노트북은 순차 루프이므로
    #    그 수치를 그대로 쓰면 4배 낙관하게 된다. 실측(EDA 52/55, 단일 프로세스):
    #
    #        640px 무필터            2.6 초/장   500장 1,315초 (제한의 55%)
    #        + ONNX YOLO 전패스      2.2 초/장   500장 1,096초 (제한의 46%)
    #
    #    설정 간 **비교**에는 위 표를 그대로 써도 된다(같은 조건에서 쟀다).
    #    절대 시간 예산을 말할 때만 이 실측치를 써야 한다.
    LADDER = [
        (640, None, 1.00),      # 기본. 단일 프로세스 500장 약 1,315초 (제한의 55%)
        (576, None, 1.00),
        (512, None, 1.00),
        (512, 0.40, 1.00),
        (448, None, 1.00),
        (448, 0.40, 1.00),
        (448, 0.25, 0.70),
        (448, 0.20, 0.40),
        (448, 0.12, 0.15),
        (448, 0.08, 0.00),
    ]

    def __init__(self, total, budget_sec=2000.0, check_every=10, window=30):
        self.total = max(1, total)
        self.budget = budget_sec
        self.check_every = check_every
        self.window = window
        self.level = 0
        self.t0 = time.time()
        self.done = 0
        self._last = self.t0
        self._recent = []

    @property
    def long_side(self):
        return self.LADDER[self.level][0]

    @property
    def keep(self):
        return self.LADDER[self.level][1]

    @property
    def second_pass_frac(self):
        return self.LADDER[self.level][2]

    def apply(self, pipe):
        """현재 단계를 파이프라인에 반영한다. 바뀌었으면 True."""
        changed = (pipe.long_side != self.long_side
                   or pipe.keep != self.keep
                   or pipe.second_pass_frac != self.second_pass_frac)
        pipe.long_side = self.long_side
        pipe.keep = self.keep
        pipe.second_pass_frac = self.second_pass_frac
        return changed

    def tick(self):
        now = time.time()
        self._recent.append(now - self._last)
        if len(self._recent) > self.window:
            self._recent.pop(0)
        self._last = now
        self.done += 1
        if self.done % self.check_every or self.done >= self.total:
            return

        remaining = self.total - self.done
        if remaining <= 0:
            return
        elapsed = now - self.t0
        # 남은 이미지에 허용되는 초/장. 누적 평균이 아니라 이 값을 직접 목표로 삼는다.
        allowed = (self.budget - elapsed) / remaining
        recent = sum(self._recent) / len(self._recent)

        if allowed <= 0:
            self.level = len(self.LADDER) - 1          # 이미 예산 초과 — 최대로 조인다
        elif recent > allowed * 0.90:
            # 얼마나 뒤처졌는지에 따라 한 번에 여러 칸 내려간다 (반응 속도 확보)
            steps = 1 if recent < allowed * 1.5 else (2 if recent < allowed * 3 else 3)
            self.level = min(len(self.LADDER) - 1, self.level + steps)
        elif recent < allowed * 0.45 and self.level > 0 and self.done > self.total * 0.25:
            # 여유가 확실할 때만, 그것도 25% 이상 진행한 뒤에만 완화한다.
            # (초반에 완화하면 진동해서 결국 예산을 넘긴다 — 실측으로 확인)
            self.level -= 1


# ----------------------------------------------------------------- 박스 사전 필터
# rec 가 1회 통과 비용의 80.9% 를 차지하고, 이미지당 박스 41개 중 날짜가 든 것은
# 0.63% 뿐이다 (EDA 7/8). rec 를 돌리기 전에 '숫자 줄일 가능성'을 싸구려 영상
# 특징만으로 판별해 대부분을 버린다.
#
# 아래 계수는 로지스틱 회귀를 자기지도로 학습해 얻었다 (EDA 18).
# 트리 모델보다 오히려 나았고(30% 유지 시 보존율 95.7% vs 93.5%),
# 숫자 15개뿐이라 모델 파일 배포가 필요 없다.
FILTER_FEATS = ["aspect", "h_norm", "fill", "n_cc", "cc_per_w", "cc_ar_mean",
                "cc_ar_std", "cc_fill_mean", "cc_h_ratio", "cc_h_std",
                "gap_ratio", "stroke", "edge_dens", "sat_mean", "val_std"]
FILTER_W = np.array([
    4.15204,    # aspect (log)
    23.40406,   # h_norm
    -3.04880,   # fill
    0.38957,    # n_cc (log)
    -0.36786,   # cc_per_w (log)
    0.30163,    # cc_ar_mean
    -0.18694,   # cc_ar_std
    2.77230,    # cc_fill_mean
    2.67884,    # cc_h_ratio
    4.29594,    # cc_h_std
    2.37132,    # gap_ratio (log)
    -1.17007,   # stroke (log)
    -7.50113,   # edge_dens
    -0.00096,   # sat_mean
    0.02704,    # val_std
])
FILTER_B = -12.11253
FILTER_LOG_IDX = [0, 3, 4, 10, 11]          # 학습 시 log1p 를 적용한 특징

# ⚠️ 전역 임계값(학습 데이터 분위수)을 쓰면 안 된다.
# 실측 결과 정확도가 22/34 → 13/34 로 무너졌다. 학습 시 점수 분포와 추론 시
# 분포가 어긋나기 때문이다(이미지 구성이 다르면 임계값이 통째로 밀린다).
# 대신 **이미지마다 점수 상위 K개**를 남긴다. 자기정규화되어 분포 이동에 영향받지
# 않고, 박스가 적은 이미지에서 전부 버려지는 사고도 없다.
# 이보다 박스가 적으면 필터를 걸지 않는다.
# ⚠️ 10 으로 뒀다가 낭패를 봤다. 이미지당 박스 중앙값이 19개라 keep=0.12 여도
#    max(10, 2)=10 개가 남아 실제 유지율이 53% 였고, 절감이 거의 없었다.
FILTER_MIN_BOXES = 4


def box_features(crop, img_h):
    """rec 없이 계산 가능한 특징. 한글 음절 블록과 숫자열을 가른다."""
    h, w = crop.shape[:2]
    if h < 3 or w < 3:
        return None
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    s = 32.0 / h
    gn = cv2.resize(g, (max(4, int(w * s)), 32), interpolation=cv2.INTER_AREA)
    H, W = gn.shape
    _, bw = cv2.threshold(gn, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if bw.mean() > 127:
        bw = 255 - bw
    fill = float((bw > 0).mean())

    n, _, stats, _ = cv2.connectedComponentsWithStats((bw > 0).astype(np.uint8), 8)
    comps = [stats[i] for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 4]
    if comps:
        cw = np.array([c[cv2.CC_STAT_WIDTH] for c in comps], float)
        ch = np.array([c[cv2.CC_STAT_HEIGHT] for c in comps], float)
        ca = np.array([c[cv2.CC_STAT_AREA] for c in comps], float)
        ar = cw / np.maximum(ch, 1)
        ccf = ca / np.maximum(cw * ch, 1)
        f_ncc, f_arm, f_ars = float(len(comps)), float(ar.mean()), float(ar.std())
        f_ccf, f_chr, f_chs = float(ccf.mean()), float((ch / H).mean()), float((ch / H).std())
    else:
        f_ncc = f_arm = f_ars = f_ccf = f_chr = f_chs = 0.0

    col = (bw > 0).any(axis=0)
    gaps = float(np.sum(np.diff(col.astype(int)) != 0))
    dist = cv2.distanceTransform((bw > 0).astype(np.uint8), cv2.DIST_L2, 3)
    stroke = float(dist[bw > 0].mean() * 2) if (bw > 0).any() else 0.0
    edges = cv2.Canny(gn, 60, 160)
    hsv = cv2.cvtColor(cv2.resize(crop, (W, H), interpolation=cv2.INTER_AREA),
                       cv2.COLOR_BGR2HSV)

    return np.array([
        w / max(h, 1), h / max(img_h, 1), fill, f_ncc, f_ncc / max(W / 32.0, 0.1),
        f_arm, f_ars, f_ccf, f_chr, f_chs, gaps / max(W / 32.0, 0.1), stroke,
        float((edges > 0).mean()), float(hsv[:, :, 1].mean()), float(hsv[:, :, 2].std()),
    ])


def box_score(feat):
    x = feat.copy()
    x[FILTER_LOG_IDX] = np.log1p(np.clip(x[FILTER_LOG_IDX], 0, None))
    return float(x @ FILTER_W + FILTER_B)


def crop_quad(img, quad):
    """
    검출 박스를 원근 보정해 잘라낸다.

    ⚠️ PaddleOCR 의 `get_rotate_crop_image` 와 **정확히 같아야 한다.**
       직접 구현했을 때 세 군데가 달라 정답률이 69.8% → 64.7% 로 떨어졌다.
         1. 세로 박스 회전 방향: paddle 은 np.rot90(반시계). 시계로 돌리면 글자가 뒤집힌다
         2. warpPerspective 보간: INTER_CUBIC
         3. rec 앞에서 미리 높이를 맞추면 안 된다 — 인식기가 자체 리사이즈를 하므로
            이중 리샘플링이 되어 화질이 깎인다
    """
    q = np.asarray(quad, dtype=np.float32)
    w = int(max(np.linalg.norm(q[0] - q[1]), np.linalg.norm(q[2] - q[3])))
    h = int(max(np.linalg.norm(q[0] - q[3]), np.linalg.norm(q[1] - q[2])))
    if w < 4 or h < 4:
        return None
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    out = cv2.warpPerspective(img, cv2.getPerspectiveTransform(q, dst), (w, h),
                              borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
    if out.shape[0] * 1.0 / out.shape[1] >= 1.5:
        out = np.rot90(out)          # 반시계 — paddle 과 동일
    return np.ascontiguousarray(out)


class OnnxDateDetector:
    """
    날짜 영역 검출기(YOLOv8n)를 **torch 없이** 돌린다.

    왜 ONNX 인가 (EDA 52~53 실측):
      torch 를 임포트하기만 해도 — YOLO 를 쓰지 않아도 — 파이프라인이
      장당 2.76초에서 5.88초로 느려졌다. paddle 과 torch 가 각자
      OpenMP 런타임(libiomp5md.dll)을 싣고 스레드를 다투기 때문이다.
      torch.set_num_threads(1), KMP_BLOCKTIME=0, OMP_WAIT_POLICY=PASSIVE 를
      전부 시도했지만 6초대에서 내려오지 않았다. 500장이 제한(2400초)을 넘는다.
      그래서 추론 시점에 torch 를 아예 싣지 않는다.

    ultralytics 가 해 주던 전·후처리를 직접 한다.
      입력  (1,3,640,640) RGB, 0~1
      출력  (1,5,8400) — 앞 4채널 cx,cy,w,h (640px 단위), 5번째가 클래스 점수
    """

    def __init__(self, onnx_path, threads=0, iou=0.45):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads      # 0 = 코어 수만큼
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(onnx_path, so,
                                         providers=["CPUExecutionProvider"])
        self.iname = self.sess.get_inputs()[0].name
        self.side = 640
        self.iou = iou

    def detect(self, canvas, conf=0.25):
        """정사각 canvas(BGR) → [[x1,y1,x2,y2], ...] (canvas 좌표계)."""
        s = self.side
        x = cv2.resize(canvas, (s, s), interpolation=cv2.INTER_LINEAR)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])
        y = self.sess.run(None, {self.iname: x})[0][0]     # (5, 8400)

        score = y[4]
        m = score >= conf
        if not m.any():
            return []
        cx, cy, bw, bh = y[0][m], y[1][m], y[2][m], y[3][m]
        score = score[m]
        # cv2.dnn.NMSBoxes 는 좌상단+크기 형식을 받는다
        rects = np.stack([cx - bw / 2, cy - bh / 2, bw, bh], 1)
        idx = cv2.dnn.NMSBoxes(rects.tolist(), score.tolist(), conf, self.iou)
        if idx is None or len(idx) == 0:
            return []
        idx = np.asarray(idx).reshape(-1)
        # 640px 좌표를 canvas 좌표로 되돌린다 (canvas 는 정사각이라 배율 하나)
        k = canvas.shape[0] / float(s)
        out = []
        for i in idx:
            x1, y1 = rects[i][0] * k, rects[i][1] * k
            out.append([x1, y1, x1 + rects[i][2] * k, y1 + rects[i][3] * k])
        return out


def fit_rec_height(crop, target=48, max_w=1600):
    h, w = crop.shape[:2]
    if h < 1:
        return None
    s = target / h
    return cv2.resize(crop, (max(8, min(max_w, int(w * s))), target),
                      interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)


# ----------------------------------------------------------------- 크롭 TTA
# 640×640 웹 수집 클러스터는 미검출이 1장뿐인데 오독이 11장이다(EDA 33).
# 박스는 제대로 찾는데 저해상도라 숫자를 틀리게 읽는다 (09→02, 11→01 등).
# 위치를 이미 아는 상태이므로, 그 박스만 변형을 바꿔 여러 번 읽어 후보를 넓힌다.
# rec 대상이 이미지당 2~7개뿐이라 비용이 작다.
def crop_variants(crop):
    """숫자 박스를 다시 읽기 위한 변형들. 원본은 호출부에서 이미 읽었다."""
    out = []
    h, w = crop.shape[:2]
    if h < 3 or w < 3:
        return out
    # 2배 확대 후 언샵 — 작은 글자의 획을 세운다
    big = cv2.resize(crop, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(big, (0, 0), 1.2)
    out.append(cv2.addWeighted(big, 1.7, blur, -0.7, 0))
    # 국소 대비 보정 — 잉크젯·각인처럼 대비가 낮은 인쇄
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l, u, v = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4)).apply(l)
    out.append(cv2.cvtColor(cv2.merge([l, u, v]), cv2.COLOR_LAB2BGR))
    return out


def digit_rich(txt):
    """저해상도 인식 결과가 '날짜일 법한' 박스인가 (크롭 TTA 대상 선별)."""
    if not txt:
        return False
    d = sum(c.isdigit() for c in txt)
    return d >= 4


# ----------------------------------------------------------------- 전처리
# 해상도를 다 낮춰도 못 찾았을 때만 쓰는 마지막 대안.
# EDA 10~17 에서 전처리를 '항상 적용'으로 실험했을 때는 유의성을 얻지 못했다
# (n=400, McNemar p=0.201). +43 얻고 −31 잃는 식이라 성공한 이미지를 망쳤기 때문이다.
# 실패한 이미지에만 걸면 잃을 것이 없다.
def pp_clahe(a):
    lab = cv2.cvtColor(a, cv2.COLOR_BGR2LAB)
    l, u, v = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, u, v]), cv2.COLOR_LAB2BGR)


def pp_adapt_bin(a):
    g = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    b = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY, 31, 10)
    return cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)


def pp_illum(a):
    g = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32) + 1.0
    bg = cv2.GaussianBlur(g, (0, 0), 25) + 1.0
    n = np.clip(g / bg * 128.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(n, cv2.COLOR_GRAY2BGR)


def pp_unsharp(a):
    blur = cv2.GaussianBlur(a, (0, 0), 2.0)
    return cv2.addWeighted(a, 1.6, blur, -0.6, 0)


PREPROCS = {"clahe": pp_clahe, "adapt_bin": pp_adapt_bin,
            "illum": pp_illum, "unsharp": pp_unsharp}


# ----------------------------------------------------------------- 이미지 IO
def imread(path):
    """한글 경로 대응. cv2 는 EXIF orientation 을 기본 적용한다 (EDA 2 검증)."""
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


def downscale(a, long_side=960):
    h, w = a.shape[:2]
    s = long_side / max(h, w)
    if s < 1.0:
        a = cv2.resize(a, (max(1, int(w * s)), max(1, int(h * s))),
                       interpolation=cv2.INTER_AREA)
    return a


# ----------------------------------------------------------------- 파이프라인
class DatePipeline:
    def __init__(self, ocr, long_side=640, rotations=(0, 90), adaptive=True,
                 keep=None):
        """
        keep: rec 를 돌릴 박스의 비율. None 이면 필터를 끄고 전부 인식한다.
              0.20 이면 rec 비용이 1/5 로 줄고 이미지 보존율은 94.2% (EDA 18).
        """
        self.ocr = ocr
        self.long_side = long_side
        self.rotations = rotations
        self.adaptive = adaptive   # 0°에서 찾으면 90°는 건너뛴다
        self.keep = keep
        self.second_pass_frac = 1.0   # 0°에서 실패한 이미지 중 90°도 볼 비율
        self._sp_credit = 0.0
        self.drop_score = 0.5     # PaddleOCR ocr() 기본값과 동일하게 맞춘다
        self.tta_drop_score = 0.8  # 크롭 변형본은 더 엄격하게
        self.crop_tta = False     # 숫자 박스 변형 재인식 (기각 — EDA 34)
        self.yolo = None          # 날짜 영역 전용 검출기 (EDA 45~48)
        self.yolo_conf = 0.25
        # "always" = 모든 패스에 YOLO 박스를 덧붙인다 (커버리지 최대, 그러나
        #            이미 맞추고 있던 이미지의 후보군도 바뀌어 오답이 늘 수 있다)
        # "fallback" = 기존 경로가 전부 실패한 이미지에만 쓴다. 성공한 이미지에
        #            닿지 않으므로 떨어질 수가 없고, 비용도 실패분에만 든다.
        #            fallback_preprocs 와 같은 논리다.
        self.yolo_mode = "fallback"
        self._yolo_on = True      # _ocr_lines 가 보는 스위치
        # YOLO 크롭 판독을 받아들일 최소 신뢰도. None 이면 drop_score(0.5) 와 같다.
        # always 모드에서 YOLO 크롭의 오독이 올바른 det 판독을 이겨 정답 4장을
        # 망가뜨렸다(EDA 48). 크롭 TTA 가 실패한 것과 같은 기제다.
        self.yolo_drop_score = None
        # 해상도를 다 낮춰도 실패했을 때만 쓰는 전처리 (정답 137장 실측, EDA 35):
        #   없음            69.8% / 무작위 65.9% / 미검출 20
        #   illum           70.6% / 68.2% / 19
        #   adapt_bin       70.6% / 68.2% / 20
        #   illum+adapt     70.6% / 68.2% / 19
        #   illum+adapt+unsharp  71.3% / 68.2% / 18   ← 채택
        #
        # '항상 적용' 으로 실험했을 때는 유의성이 없었다(EDA 12, p=0.201).
        # 성공한 이미지를 망가뜨렸기 때문이다. 실패한 것에만 걸면 잃을 게 없다.
        self.fallback_preprocs = [pp_illum, pp_adapt_bin, pp_unsharp]
        # 날짜를 못 찾았을 때 다시 훑어볼 더 작은 해상도들.
        #
        # 정답 137장 ablation (EDA 31, 부분점수 기준):
        #   640 단독            63.0%   미검출 30   500장 376초
        #   640→512→448         68.6%   미검출 21   500장 519초
        #   640→512→448→384     69.3%   미검출 20   500장 493초  ← 채택
        #   + 180/270 회전      68.4%   미검출 20   500장 701초  ← 기각(효과 없음)
        #
        # 384px 는 단독으로는 최악(52.9%)이지만 마지막 대안으로는 유효하다.
        # 다른 해상도가 전부 실패한 것만 넘어오기 때문이다.
        self.fallback_sides = (512, 448, 384)
        self._det = getattr(ocr, "text_detector", None)
        self._rec = getattr(ocr, "text_recognizer", None)

    # ------------------------------------------------------------- YOLO 보조 검출
    # 실패 원인 분석(EDA 44)에서 확인한 것:
    #   정답 날짜를 담은 박스를 det 가 찾은 비율이 성공 59% / 실패 13% / 못읽음 5%.
    #   이미지 품질·글자 크기·인식 신뢰도는 성공/실패 간 차이가 없었다.
    #   즉 병목은 **PP-OCRv3 검출기가 날짜 영역을 텍스트로 안 보는 것**이다.
    #   (실측: 같은 스티커에서 '03:17' 은 읽고 바로 위 '21.07.23' 은 못 읽었다)
    # → 날짜 영역 전용 검출기(YOLOv8n)를 붙여 det 가 놓친 영역을 보충한다.
    def _yolo_boxes(self, img):
        """YOLO 가 제안한 날짜 영역을 네 점 좌표로 돌려준다."""
        if self.yolo is None or not self._yolo_on:
            return []
        h, w = img.shape[:2]
        side = max(h, w)
        canvas = np.full((side, side, 3), 114, np.uint8)
        y0, x0 = (side - h) // 2, (side - w) // 2
        canvas[y0:y0 + h, x0:x0 + w] = img
        try:
            if hasattr(self.yolo, "detect"):          # OnnxDateDetector
                raw = self.yolo.detect(canvas, self.yolo_conf)
            else:                                     # ultralytics YOLO (torch)
                raw = []
                for r in self.yolo.predict(canvas, imgsz=640, conf=self.yolo_conf,
                                           verbose=False, device="cpu"):
                    for b in getattr(r, "boxes", []):
                        raw.append([float(v) for v in b.xyxy[0]])
        except Exception:
            return []
        out = []
        for x1, y1, x2, y2 in raw:
            # 레터박스 좌표를 원래 이미지 좌표로 되돌린다
            x1, x2 = x1 - x0, x2 - x0
            y1, y2 = y1 - y0, y2 - y0
            # 잘림 사고가 잦아 여유를 준다
            pw, ph = (x2 - x1) * 0.10, (y2 - y1) * 0.25
            x1, x2 = max(0, x1 - pw), min(w, x2 + pw)
            y1, y2 = max(0, y1 - ph), min(h, y2 + ph)
            if x2 - x1 < 6 or y2 - y1 < 6:
                continue
            out.append(np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                                dtype=np.float32))
        return out

    def _ocr_lines(self, img):
        """
        검출 → (필터) → 인식.

        기본 설정(keep=None)에서는 PaddleOCR 의 ocr() 를 그대로 쓴다.
        det/rec 를 직접 부르는 경로를 만들어 봤지만 미묘한 차이가 누적되어
        정답률이 69.8% → 68.4% 로 낮았다(크롭 함수·박스 정렬·drop_score).
        박스 필터가 필요할 때만 직접 경로를 쓴다.
        """
        yolo_live = self.yolo is not None and self._yolo_on
        if self._det is None or self._rec is None or (
                self.keep is None and not self.crop_tta and not yolo_live):
            res = self.ocr.ocr(img, cls=False)
            lines = res[0] if res and res[0] else []
            return [(ln[1][0], ln[0]) for ln in lines]

        # YOLO 만 붙인 기본 구성: 라이브러리 경로로 본선을 돌리고
        # YOLO 가 제안한 영역만 따로 인식해 덧붙인다.
        # (직접 det/rec 경로는 라이브러리 대비 1.4%p 낮았다 — EDA 34)
        if self.keep is None and not self.crop_tta and yolo_live:
            res = self.ocr.ocr(img, cls=False)
            base = res[0] if res and res[0] else []
            lines = [(ln[1][0], ln[0]) for ln in base]
            extra = []
            crops, quads = [], []
            for q in self._yolo_boxes(img):
                c = crop_quad(img, q)
                if c is not None:
                    crops.append(c)
                    quads.append(q)
            if crops:
                thr = self.drop_score if self.yolo_drop_score is None \
                    else self.yolo_drop_score
                rr = self._rec(crops)
                rr = rr[0] if isinstance(rr, tuple) else rr
                for i in range(min(len(rr), len(quads))):
                    if float(rr[i][1]) >= thr:
                        extra.append((rr[i][0], quads[i]))
            return lines + extra

        boxes, _ = self._det(img)
        if boxes is None or not len(boxes):
            return []
        H = img.shape[0]
        cand = []
        for b in boxes:
            c = crop_quad(img, b)
            if c is None:
                continue
            # 크롭을 그대로 넘긴다 — 인식기가 자체 리사이즈를 하므로
            # 여기서 높이를 맞추면 이중 리샘플링이 된다
            if self.keep is None:
                cand.append((0.0, c, b))
            else:
                ft = box_features(c, H)
                cand.append((-1e9 if ft is None else box_score(ft), c, b))

        if self.keep is not None and len(cand) > FILTER_MIN_BOXES:
            # 이미지별 상위 K개만 남긴다 (전역 임계값은 분포 이동에 취약하다)
            k = max(FILTER_MIN_BOXES, int(round(len(cand) * self.keep)))
            cand.sort(key=lambda t: -t[0])
            cand = cand[:k]

        if not cand:
            return []
        crops = [c[1] for c in cand]
        keep_boxes = [c[2] for c in cand]
        rr = self._rec(crops)
        rr = rr[0] if isinstance(rr, tuple) else rr
        # ⚠️ PaddleOCR 의 ocr() 은 신뢰도 0.5 미만 인식을 버린다(drop_score).
        #    직접 det/rec 를 부르면서 이걸 빠뜨렸더니 쓰레기 문자열이 후보로 들어와
        #    정답률이 69.8% → 65.5% 로 떨어졌다. 반드시 같은 기준으로 걸러야 한다.
        lines, kept_crops = [], []
        for i in range(min(len(rr), len(keep_boxes))):
            txt, score = rr[i][0], float(rr[i][1])
            if score < self.drop_score:
                continue
            lines.append((txt, keep_boxes[i]))
            kept_crops.append(crops[i])
        if self.crop_tta:
            lines += self._crop_tta(lines, kept_crops)
        return lines

    def _crop_tta(self, lines, crops):
        """숫자가 든 박스만 변형을 바꿔 다시 읽는다 (저해상도 오독 대응)."""
        extra_imgs, extra_boxes = [], []
        for i, (txt, box) in enumerate(lines):
            if i >= len(crops) or not digit_rich(txt):
                continue
            for v in crop_variants(crops[i]):
                extra_imgs.append(v)
                extra_boxes.append(box)
        if not extra_imgs:
            return []
        rr = self._rec(extra_imgs)
        rr = rr[0] if isinstance(rr, tuple) else rr
        out = []
        for i in range(min(len(rr), len(extra_boxes))):
            # 변형본은 원본보다 신뢰도가 낮게 나오기 쉬우므로 더 엄격하게 건다
            if float(rr[i][1]) >= self.tta_drop_score:
                out.append((rr[i][0], extra_boxes[i]))
        return out

    def _scan(self, img, long_side):
        """한 해상도에서 0°(→필요시 90°) 로 훑는다."""
        small = downscale(img, long_side)
        cands, anchors, md, ym, all_text = {}, set(), [], [], []
        for i, deg in enumerate(self.rotations):
            if i > 0:
                # 0°에서 이미 찾았으면 두 번째 패스는 불필요
                if self.adaptive and cands:
                    break
                # 예산이 빠듯하면 실패한 이미지 중 일부만 두 번째 패스를 준다.
                # 난수 대신 크레딧 누적 방식이라 결과가 결정적(재현 가능)이다.
                if self.second_pass_frac < 1.0:
                    self._sp_credit += self.second_pass_frac
                    if self._sp_credit < 1.0:
                        break
                    self._sp_credit -= 1.0
            view = small if deg == 0 else cv2.rotate(
                small, {90: cv2.ROTATE_90_CLOCKWISE,
                        180: cv2.ROTATE_180,
                        270: cv2.ROTATE_90_COUNTERCLOCKWISE}[deg])
            lines = self._ocr_lines(view)
            chunk = [t for t, _ in lines] + merge_lines(lines)
            blob = "\n".join(chunk)
            anchors |= find_anchors(blob)
            for d in find_dates(blob):
                key = "%04d-%02d-%02d" % (d["y"], d["m"], d["d"])
                cands.setdefault(key, d["kind"])
            md += find_month_day(blob)
            ym += find_year_month(blob)
            all_text.append(blob)
        return cands, anchors, md, ym, "\n".join(all_text)

    def predict_image(self, img):
        """
        BGR ndarray → (선택결과 dict|None, 후보dict, 앵커set)

        해상도 단계적 재시도(escalation):
          기본 해상도에서 날짜를 못 찾으면 더 작은 해상도로 다시 훑는다.
          det 는 작게 넣을수록 인접 글자를 한 박스로 묶어 날짜 줄이 덜 쪼개지므로,
          해상도마다 잡히는 이미지가 다르다 (EDA 27: 640/512/448 합집합이 단일 대비
          커버리지 +9.7%p).

          '항상 세 해상도를 다 돌려 합치는' 방식과 커버리지는 같으면서
          비용은 실패한 이미지에만 든다. 또 이미 성공한 이미지에 다른 해상도의
          후보를 섞지 않으므로 오답이 늘어날 위험도 없다.
        """
        sides = [self.long_side] + [s for s in self.fallback_sides if s < self.long_side]
        cands, anchors, md, ym, texts = {}, set(), [], [], []

        def absorb(res):
            c, a, m, y, t = res
            for k, v in c.items():
                cands.setdefault(k, v)
            anchors.update(a)
            md.extend(m)
            ym.extend(y)
            texts.append(t)

        # fallback 모드에서는 기존 경로를 건드리지 않기 위해 먼저 YOLO 를 끈다.
        self._yolo_on = (self.yolo is not None and self.yolo_mode == "always")

        for side in sides:
            absorb(self._scan(img, side))
            if cands and not needs_more(cands, anchors):
                break          # 충분히 찾았으면 더 내려갈 이유가 없다
        else:
            # 해상도를 다 낮춰도 못 찾았을 때만 전처리를 시도한다.
            # 성공한 이미지에는 아예 닿지 않으므로 오탐이 늘 위험이 없다.
            for fn in self.fallback_preprocs:
                try:
                    absorb(self._scan(fn(img), self.long_side))
                except Exception:
                    continue
                if cands:
                    break
            # 그래도 아무것도 못 찾았으면 날짜 전용 검출기를 켜고 한 번 더 본다.
            # 여기까지 온 이미지는 현재 전부 NONE 으로 나가므로 잃을 것이 없다.
            if not cands and self.yolo is not None and self.yolo_mode == "fallback":
                self._yolo_on = True
                try:
                    absorb(self._scan(img, self.long_side))
                except Exception:
                    pass
                self._yolo_on = False
        text = "\n".join(texts)
        return select_date(cands, anchors, md, ym, text), cands, anchors

    def predict_path(self, path):
        img = imread(path)
        if img is None:
            return None, {}, set()
        return self.predict_image(img)


def to_row(image_id, final):
    """
    제출 규격으로 변환 (운영진 확정 2026-09-12).

      전부 인식     → 2026 / 05 / 29 / 2026-05-29
      월·일만 인식  → NONE / 08 / 25 / NONE-08-25
      전부 미인식   → NONE / NONE / NONE / NONE

    final_date 는 세 값을 하이픈으로 연결하되, 셋 다 NONE 일 때만 'NONE' 하나.
    """
    if not final:
        y = m = d = "NONE"
    else:
        y = "%04d" % final["y"] if final.get("y") else "NONE"
        m = "%02d" % final["m"] if final.get("m") else "NONE"
        d = "%02d" % final["d"] if final.get("d") else "NONE"
    fd = "NONE" if y == m == d == "NONE" else "%s-%s-%s" % (y, m, d)
    return {"image_id": image_id, "year": y, "month": m, "day": d, "final_date": fd}
