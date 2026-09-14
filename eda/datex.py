"""
날짜 후보 추출 / 앵커 키워드 탐지 유틸.

EDA 분석과 이후 추론 파이프라인 후처리에서 공용으로 쓴다.
"""
import re
import unicodedata
from datetime import date

import numpy as np

# ---------------------------------------------------------------- 앵커 키워드
# OCR 오인식을 감안해 '핵심 부분 문자열'로 느슨하게 잡는다.
ANCHORS = {
    # 소비기한 계열 (정답 앵커)
    "소비기한": ["소비기한", "소비기 한", "소비기힌", "소비기한까지", "비기한"],
    "유통기한": ["유통기한", "유통기 한", "유통기힌", "통기한"],
    "품질유지기한": ["품질유지기한", "품질유지"],
    # 제조 계열 (반대 앵커 — 이 날짜를 고르면 오답)
    "제조일자": ["제조일자", "제조년월일", "제조일", "제조(년", "제조:", "제조 "],
    "생산일자": ["생산일자", "생산일"],
    "포장일자": ["포장일자", "포장일", "포장년월일"],
    # 다국어
    "賞味期限": ["賞味期限", "賞味期"],
    "消費期限": ["消費期限", "消費期"],
    "BEST_BEFORE": ["bestbefore", "bestbefor", "bestbe", "bbe", "bestbefend"],
    "EXP": ["exp.", "expiry", "expdate", "expirydate", "useby", "usebefore"],
    "MFG": ["mfg", "mfd", "manufactur", "proddate", "productiondate"],
    # 보조 신호
    "까지": ["까지"],
    "별도표기": ["별도표기", "별도표시", "표기일까지", "하단표기", "상단표기",
              "제품별도", "별도기재", "우측표기", "전면표기"],
}

POSITIVE = ("소비기한", "유통기한", "품질유지기한", "賞味期限", "消費期限", "BEST_BEFORE", "EXP")
NEGATIVE = ("제조일자", "생산일자", "포장일자", "MFG")

EN_MONTH = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}


def norm(s: str) -> str:
    """공백/기호 제거 + 소문자화 — 앵커 substring 매칭용."""
    s = unicodedata.normalize("NFKC", s or "")
    return re.sub(r"[\s·:;,_()\[\]/\\|]+", "", s).lower()


def find_anchors(text: str):
    """텍스트에 등장하는 앵커 종류 집합을 반환."""
    n = norm(text)
    hit = set()
    for key, pats in ANCHORS.items():
        for p in pats:
            if norm(p) and norm(p) in n:
                hit.add(key)
                break
    return hit


# ---------------------------------------------------------------- 날짜 패턴
# OCR 은 잉크젯 날짜의 구분자(. - /)를 자주 통째로 날리거나 엉뚱한 기호로 읽는다.
# (실측: "21.02.11" → "210211", "2027.06.26" → "2027 06 26")
# 따라서 구분자는 '없어도 되고, 아무 비숫자 1~2글자여도 되는' 것으로 본다.
S = r"[^\d\n]{0,2}"

PATTERNS = [
    # 2026.05.29 / 2026-05-29 / 2026년 5월 29일 / 2026 05 29 / 20260529
    ("Y4_sep", re.compile(
        r"(?<![\d])(?P<y>19\d{2}|20[0-4]\d)" + S +
        r"(?P<m>1[0-2]|0[1-9])" + S + r"(?P<d>3[01]|[12]\d|0[1-9])(?![\d])")),
    # 2026.5.29 처럼 월/일이 1자리인 경우 (구분자 필수 — 없으면 자릿수 모호)
    ("Y4_sep1", re.compile(
        r"(?<![\d])(?P<y>19\d{2}|20[0-4]\d)\s*[.\-/년]\s*(?P<m>1[0-2]|[1-9])\s*[.\-/월]\s*"
        r"(?P<d>3[01]|[12]\d|[1-9])\s*일?(?![\d])")),
    # 26.05.29 / 260529 (2자리 연도) — 오탐 위험이 커서 별도 등급으로 취급
    ("Y2_sep", re.compile(
        r"(?<![\d])(?P<y>[0-4]\d)" + S +
        r"(?P<m>1[0-2]|0[1-9])" + S + r"(?P<d>3[01]|[12]\d|0[1-9])(?![\d])")),
    # 29/SEP/2022, 29 SEP 2022 (영문월은 OCR 오인식이 잦아 느슨하게)
    ("D_MON_Y", re.compile(
        r"(?<![\d])(?P<d>3[01]|[12]\d|0?[1-9])[^\dA-Za-z]{0,2}(?P<mon>[A-Za-z0-9]{2,5})"
        r"[^\dA-Za-z]{0,2}(?P<y>19\d{2}|20[0-4]\d)(?![\d])")),
    # SEP/29/2022
    ("MON_D_Y", re.compile(
        r"(?<![A-Za-z])(?P<mon>[A-Za-z]{3,4})[^\dA-Za-z]{0,2}(?P<d>3[01]|[12]\d|0?[1-9])"
        r"[^\dA-Za-z]{0,2}(?P<y>19\d{2}|20[0-4]\d)(?![\d])")),
    # 2026.05 (일 없음 — 연월만)
    ("Y4_M_only", re.compile(
        r"(?<![\d])(?P<y>20[0-4]\d)\s*[.\-/년]\s*(?P<m>1[0-2]|0?[1-9])\s*월?(?![\d.\-/])")),
]

# 영문 월 약어의 OCR 오인식 보정 (예: SEP → 48P, 9EP, 5EP)
_MON_KEYS = {
    1: "jan", 2: "feb", 3: "mar", 4: "apr", 5: "may", 6: "jun",
    7: "jul", 8: "aug", 9: "sep", 10: "oct", 11: "nov", 12: "dec",
}
_CONFUSE = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s",
                          "6": "g", "8": "b", "9": "g"})


def fuzzy_month(tok: str):
    """OCR 오인식된 영문 월 약어를 월 번호로 되돌린다. 실패 시 None."""
    t = re.sub(r"[^a-z0-9]", "", (tok or "").lower())
    if not t:
        return None
    if t.isdigit():
        return None
    t2 = t.translate(_CONFUSE)
    for num, name in _MON_KEYS.items():
        if name in t or name in t2:
            return num
    # 3글자 중 2글자 일치 허용
    for num, name in _MON_KEYS.items():
        if len(t2) >= 3 and sum(a == b for a, b in zip(t2[:3], name)) >= 2:
            return num
    return None

# 날짜로 오인되기 쉬운 것들
DISTRACTORS = {
    "품목보고번호(14자리+)": re.compile(r"(?<![\d])\d{14,}(?![\d])"),
    "바코드(13자리)": re.compile(r"(?<![\d])\d{13}(?![\d])"),
    "9~12자리 숫자": re.compile(r"(?<![\d])\d{9,12}(?![\d])"),
    "전화번호(1588형)": re.compile(r"(?<![\d])(?:1[3-9]\d{2}|080|02|0[3-6]\d)[-\s]?\d{3,4}[-\s]?\d{4}(?![\d])"),
    "신고번호 1399": re.compile(r"(?<![\d])1399(?![\d])"),
    "영양성분 %": re.compile(r"\d+\s*%"),
    "용량 g/ml/kcal": re.compile(r"\d[\d,.]*\s*(?:g|kg|ml|l|kcal|mg)(?![a-z])", re.I),
    "LOT/로트": re.compile(r"(?:lot|로트|롯트)\s*(?:no)?\.?\s*[:\s]?\s*[A-Z0-9]{3,}", re.I),
    "시각(hh:mm)": re.compile(r"(?<![\d])(?:[01]?\d|2[0-3])\s*:\s*[0-5]\d(?![\d])"),
    "시각(NN시)": re.compile(r"(?<![\d])(?:[01]?\d|2[0-3])\s*시(?![간])"),
}

# 긴 숫자열 안에 박힌 8자리(=품목보고번호 앞자리 등)를 걸러내기 위한 패턴
LONG_DIGITS = re.compile(r"\d{9,}")


def _mk(y, m, d, kind, raw, span):
    try:
        y, m, d = int(y), int(m), int(d)
    except (TypeError, ValueError):
        return None
    if y < 100:                       # 2자리 연도 → 2000년대로 확장
        y = 2000 + y
    if not (1990 <= y <= 2049 and 1 <= m <= 12 and 1 <= d <= 31):
        return None
    try:
        date(y, m, d)                 # 2월 30일 같은 것 제거
    except ValueError:
        return None
    return {"y": y, "m": m, "d": d, "kind": kind, "raw": raw, "span": span}


# 앞자리 0 이 탈락한 7자리 압축형 (실측: '2022.07.16' → '2022716')
# 오탐 위험이 있어 기본 비활성 — find_dates(..., loose=True) 로만 켠다.
LOOSE_PATTERNS = [
    ("Y4_M1_D2", re.compile(
        r"(?<![\d])(?P<y>20[0-4]\d)(?P<m>[1-9])(?P<d>0[1-9]|[12]\d|3[01])(?![\d])")),
    ("Y4_M2_D1", re.compile(
        r"(?<![\d])(?P<y>20[0-4]\d)(?P<m>0[1-9]|1[0-2])(?P<d>[1-9])(?![\d])")),
]


def _box_geom(b):
    q = np.array(b, dtype=float).reshape(4, 2)
    w = max(np.linalg.norm(q[0] - q[1]), np.linalg.norm(q[2] - q[3]))
    h = max(np.linalg.norm(q[0] - q[3]), np.linalg.norm(q[1] - q[2]))
    return q[:, 1].mean(), w, h, q[:, 0].min(), q[:, 0].max()


def merge_lines(lines, y_tol=0.6, gap_tol=2.0):
    """
    같은 행에 있고 가로로 인접한 박스들의 텍스트를 이어 붙여 반환한다.

    det 가 날짜 한 줄을 여러 박스로 쪼개는 경우가 있다
    (실측: '2022/11/04' → '20221' + '104'). 이어 붙이면 복구된다.
    추가 OCR 이 필요 없어 추론 비용은 0.

    lines: [{"t": 텍스트, "box": [x1,y1,...,x4,y4]}, ...]
    """
    items = []
    for l in lines:
        b = l.get("box")
        if not b:
            continue
        cy, w, h, x0, x1 = _box_geom(b)
        if h < 1:
            continue
        items.append({"t": l["t"], "cy": cy, "h": h, "x0": x0, "x1": x1})
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


def find_dates(text: str, drop_inside_long_digits=True, loose=False):
    """텍스트에서 날짜 후보를 모두 뽑는다. 겹치는 매치는 앞선 패턴 우선."""
    if not text:
        return []
    long_spans = [mm.span() for mm in LONG_DIGITS.finditer(text)] if drop_inside_long_digits else []
    out, taken = [], []
    for kind, rx in (PATTERNS + LOOSE_PATTERNS if loose else PATTERNS):
        for m in rx.finditer(text):
            s, e = m.span()
            if any(s < te and ts < e for ts, te in taken):
                continue
            gd = m.groupdict()
            if "mon" in gd and gd.get("mon"):
                mm = EN_MONTH.get(gd["mon"].lower().rstrip(".")) or fuzzy_month(gd["mon"])
                if mm is None:
                    continue
                rec = _mk(gd["y"], mm, gd["d"], kind, m.group(0), (s, e))
            else:
                rec = _mk(gd.get("y"), gd.get("m"), gd.get("d", 1), kind, m.group(0), (s, e))
            if rec is None:
                continue
            rec["in_long_digits"] = any(ls <= s and e <= le for ls, le in long_spans)
            rec["day_missing"] = kind == "Y4_M_only"
            out.append(rec)
            taken.append((s, e))
    return sorted(out, key=lambda r: r["span"][0])


def count_distractors(text: str):
    return {k: len(rx.findall(text or "")) for k, rx in DISTRACTORS.items()}
