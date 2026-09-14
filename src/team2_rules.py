"""
팀원2(박안젤라) 날짜 판별 규칙 — 원문 그대로 이식.

출처: `팀원 기록/2/박안젤라_predict.ipynb` 의 규칙 셀 2개.
**재구현하지 않고 복사**했다. 680줄짜리 키워드/정규식 규칙을 옮겨 적으면
전사 오류가 섞이고, 그 오류가 성능 차이로 오해될 수 있기 때문이다.

이 규칙을 쓰는 이유 (홀드아웃2 198장 실측, EDA 63~64):
    우리 파이프라인          66.8%
    팀원2 (RapidOCR + 이 규칙) 79.8%   (+13.0%p, McNemar p=0.0000)
    둘을 합침                82.3%

규칙의 성격
  · OCR 조각을 원문 + 인접 2개 + 인접 3개로 이어 붙여 후보를 만든다
    (우리 merge_lines 가 박스 좌표로 하는 일을 텍스트 수준에서 한다)
  · 소비기한/유통기한/EXP/BEST BEFORE 가점, 제조일/포장일/MFG/PROD 감점
  · 완전한 날짜를 못 찾으면 부분 날짜(연·월만 등)로 내려간다

함수
  choose_expiration_date(texts) -> "YYYY-MM-DD" | None
  extract_partial_date(texts)   -> {"year":..,"month":..,"day":..} | None
"""
# flake8: noqa
# 아래는 팀원2 원문이다. 스타일을 고치지 않는다 — 원문과 대조 가능해야 한다.

def choose_expiration_date(texts):


    import re
    from datetime import datetime

    MIN_YEAR = 2015
    MAX_YEAR = 2035

    # =========================
    # 1. OCR 텍스트 정리
    # =========================
    clean_texts = [
        str(t).strip()
        for t in texts
        if str(t).strip()
    ]

    # OCR이 날짜를 여러 조각으로 나눌 수 있으므로
    # 원문 + 인접 2개 + 인접 3개를 후보로 사용
    candidates = []

    for i, text in enumerate(clean_texts):

        candidates.append(text)

        if i + 1 < len(clean_texts):
            candidates.append(
                clean_texts[i]
                + " "
                + clean_texts[i + 1]
            )

        if i + 2 < len(clean_texts):
            candidates.append(
                clean_texts[i]
                + " "
                + clean_texts[i + 1]
                + " "
                + clean_texts[i + 2]
            )

    # =========================
    # 2. 키워드
    # =========================
    strong_positive_keywords = [
        "소비기한",
        "유통기한",
        "EXPIRY",
        "EXP DATE",
        "EXP:",
        "BEST BEFORE",
        "BEST BY",
        "BEST IF USED BY",
        "USE BY",
        "BB:"
    ]

    weak_positive_keywords = [
        "EXP",
        "BB",
        "까지"
    ]

    negative_keywords = [
        "제조일",
        "제조일자",
        "제조일시",
        "생산일",
        "생산일자",

        "포장일",
        "포장일자",
        "포장년",

        "MFG",
        "PROD",
        "PROD DATE",
        "PRODUCTION DATE",

        "PACK DATE",
        "PACKED",
        "PACKING DATE",

        "PD:",
        "P.D.",
        "P.D:"
    ]

    month_map = {
        "JAN": 1,
        "FEB": 2,
        "MAR": 3,
        "APR": 4,
        "MAY": 5,
        "JUN": 6,
        "JUL": 7,
        "AUG": 8,
        "SEP": 9,
        "OCT": 10,
        "NOV": 11,
        "DEC": 12
    }

    # =========================
    # 3. 날짜 추출 함수
    # =========================
    def extract_dates(text):

        text_upper = text.upper()
        found = []

        # ---------------------------------
        # OCR 숫자 오인 보정
        # 예:
        # 2020.C6.30 -> 2020.06.30
        # 2020C6.30  -> 2020.06.30
        # 2020:C6.30 -> 2020.06.30
        # ---------------------------------
        text_upper = re.sub(
            r'(?<!\d)'
            r'(20\d{2})'
            r'[\.\-/:]?\s*'
            r'[COQ]'
            r'(\d)'
            r'[\.\-/:]'
            r'(\d{1,2})'
            r'(?!\d)',
            r'\1.0\2.\3',
            text_upper
        )

        # ---------------------------------
        # A. YYYY.MM.DD / YYYY/MM/DD
        # 예: 2022/12/02
        # OCR 분할 예: 2022/12. 02
        # ---------------------------------
        pattern1 = re.findall(
            r'(?<!\d)'
            r'(20\d{2})'
            r'[\.\-/\s]+'
            r'(\d{1,2})'
            r'[\.\-/\s]+'
            r'(\d{1,2})'
            r'(?!\d)',
            text_upper
        )

        for y, m, d in pattern1:

            year = int(y)

            if not (MIN_YEAR <= year <= MAX_YEAR):
                continue

            try:
                dt = datetime(
                    year,
                    int(m),
                    int(d)
                )
                found.append(dt)
            except:
                pass

        # ---------------------------------
        # A-1. YYYY.MMDD
        # 예:
        # 2021.1117 -> 2021-11-17
        # 2023.1015 -> 2023-10-15
        # ---------------------------------
        pattern_year_monthday = re.findall(
            r'(?<!\d)'
            r'(20\d{2})'
            r'\s*[\.\-/]\s*'
            r'(\d{2})'
            r'(\d{2})'
            r'(?!\d)',
            text_upper
        )

        for year, month, day in pattern_year_monthday:

            year = int(year)
            month = int(month)
            day = int(day)

            if not (MIN_YEAR <= year <= MAX_YEAR):
                continue

            try:
                dt = datetime(
                    year,
                    month,
                    day
                )
                found.append(dt)

            except:
                pass

        # ---------------------------------
        # B. DD.MM.YYYY / DD/MM/YYYY
        # 예: 24/12/2021
        # ---------------------------------
        pattern2 = re.findall(
            r'(?<!\d)'
            r'(\d{1,2})'
            r'\s*[\.\-/]+\s*'
            r'(\d{1,2})'
            r'\s*[\.\-/]+\s*'
            r'(20\d{2})'
            r'(?!\d)',
            text_upper
        )

        for d, m, y in pattern2:

            year = int(y)

            if not (MIN_YEAR <= year <= MAX_YEAR):
                continue

            try:
                dt = datetime(
                    year,
                    int(m),
                    int(d)
                )
                found.append(dt)
            except:
                pass
        # ---------------------------------
        # B-2. DD MM YYYY
        # 예: BEST BEFORE 02 12 2021
        # ---------------------------------
        pattern_space_dmy = re.findall(
            r'(?<!\d)'
            r'(\d{1,2})'
            r'\s+'
            r'(\d{1,2})'
            r'\s+'
            r'(20\d{2})'
            r'(?!\d)',
            text_upper
        )

        for day, month, year in pattern_space_dmy:

            day = int(day)
            month = int(month)
            year = int(year)

            if not (MIN_YEAR <= year <= MAX_YEAR):
                continue

            try:
                dt = datetime(
                    year,
                    month,
                    day
                )
                found.append(dt)

            except:
                pass

        # ---------------------------------
        # C. YYYY/JUN/14
        # ---------------------------------
        pattern3 = re.findall(
            r'(20\d{2})'
            r'[\.\-/\s]+'
            r'(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)'
            r'[\.\-/\s]+'
            r'(\d{1,2})',
            text_upper
        )

        for y, mon, d in pattern3:

            year = int(y)

            if not (MIN_YEAR <= year <= MAX_YEAR):
                continue

            try:
                dt = datetime(
                    year,
                    month_map[mon],
                    int(d)
                )
                found.append(dt)
            except:
                pass

        # ---------------------------------
        # D. JUN 08 2018
        # ---------------------------------
        pattern4 = re.findall(
            r'(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)'
            r'[\s\.\-/]+'
            r'(\d{1,2})'
            r'[\s,\.\-/]+'
            r'(20\d{2})',
            text_upper
        )

        for mon, d, y in pattern4:

            year = int(y)

            if not (MIN_YEAR <= year <= MAX_YEAR):
                continue

            try:
                dt = datetime(
                    year,
                    month_map[mon],
                    int(d)
                )
                found.append(dt)
            except:
                pass

        # ---------------------------------
        # E. 31JUL21 / 31 JUL 21
        # ---------------------------------
        pattern5 = re.findall(
            r'(?<!\d)'
            r'(\d{1,2})'
            r'[\s\.\-/]*'
            r'(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)'
            r'[\s\.\-/]*'
            r'(\d{2})'
            r'(?!\d)',
            text_upper
        )

        for d, mon, yy in pattern5:

            year = 2000 + int(yy)

            if not (MIN_YEAR <= year <= MAX_YEAR):
                continue

            try:
                dt = datetime(
                    year,
                    month_map[mon],
                    int(d)
                )
                found.append(dt)
            except:
                pass

        # ---------------------------------
        # F. 2자리 연도 날짜
        # 예: 17.12.20 / 24/12/21
        # ---------------------------------
        pattern6 = re.findall(
            r'(?<!\d)'
            r'(\d{1,2})'
            r'[\.\-/]'
            r'(\d{1,2})'
            r'[\.\-/]'
            r'(\d{2})'
            r'(?!\d)',
            text_upper
        )

        for a, b, c in pattern6:

          a = int(a)
          b = int(b)
          c = int(c)

          ymd_date = None
          dmy_date = None

          # -------------------------
          # 해석 1: YY.MM.DD
          # 예: 20.01.03 -> 2020-01-03
          # -------------------------
          year_ymd = 2000 + a

          if MIN_YEAR <= year_ymd <= MAX_YEAR:
              try:
                  ymd_date = datetime(
                      year_ymd,
                      b,
                      c
                  )
              except:
                  pass

          # -------------------------
          # 해석 2: DD.MM.YY
          # 예: 17.12.20 -> 2020-12-17
          # -------------------------
          year_dmy = 2000 + c

          if MIN_YEAR <= year_dmy <= MAX_YEAR:
              try:
                  dmy_date = datetime(
                      year_dmy,
                      b,
                      a
                  )
              except:
                  pass

          # -------------------------
          # 선택 규칙
          # -------------------------

          # 부터/까지가 있으면 YY.MM.DD 우선
          if ("부터" in text) or ("까지" in text):

              if ymd_date is not None:
                  found.append(ymd_date)

              elif dmy_date is not None:
                  found.append(dmy_date)

          else:

              # 한쪽만 유효하면 그쪽 사용
              if ymd_date is not None and dmy_date is None:
                  found.append(ymd_date)

              elif dmy_date is not None and ymd_date is None:
                  found.append(dmy_date)

              # 둘 다 가능하면 기존처럼 DD.MM.YY 우선
              elif ymd_date is not None and dmy_date is not None:
                  found.append(dmy_date)


        # ---------------------------------
        # G. 공백으로 구분된 2자리 날짜
        # 예: EXP:09 11 22
        # 소비기한 문맥이 같은 텍스트에 있을 때만 허용
        # DD MM YY로 해석
        # ---------------------------------

        has_expiry_context = any(
            kw in text_upper
            for kw in [
                "EXP",
                "EXPIRY",
                "BEST",
                "USE BY",
                "BB",
                "소비기한",
                "유통기한"
            ]
        )

        if has_expiry_context:

            pattern7 = re.findall(
                r'(?<!\d)'
                r'(\d{1,2})'
                r'\s+'
                r'(\d{1,2})'
                r'\s+'
                r'(\d{2})'
                r'(?!\d)',
                text_upper
            )

            for d, m, yy in pattern7:

                d = int(d)
                m = int(m)
                yy = int(yy)

                year = 2000 + yy

                if not (2015 <= year <= 2035):
                    continue

                try:
                    dt = datetime(
                        year,
                        m,
                        d
                    )
                    found.append(dt)
                except:
                    pass


        # ---------------------------------
        # H. 구분자 없는 숫자 날짜
        # 예: EXP 260529   -> 2026-05-29
        #     EXP 20260529 -> 2026-05-29
        #
        # 6/8자리 숫자는 LOT 번호 등과 혼동하기 쉬우므로
        # 소비기한 문맥이 있을 때만 날짜로 인정
        # ---------------------------------

        compact_positive_keywords = [
            "소비기한",
            "유통기한",
            "EXP",
            "EXPIRY",
            "EXP DATE",
            "BEST BEFORE",
            "BEST BY",
            "BEST",
            "USE BY",
            "BB",
            "까지"
        ]

        compact_negative_keywords = [
            "제조일",
            "제조일자",
            "생산일",
            "생산일자",
            "포장일",
            "MFG",
            "PROD",
            "PRODUCTION",
            "PACK",
            "PD:",
            "P.D.",
            "LOT",
            "LOTTO"
        ]

        has_compact_positive = any(
            kw in text_upper
            for kw in compact_positive_keywords
        )

        has_compact_negative = any(
            kw in text_upper
            for kw in compact_negative_keywords
        )

        # 소비기한 문맥이 있고,
        # 제조/LOT 문맥이 아닐 때만 검사
        compact_only = bool(
            re.fullmatch(
                r'\s*(?:20\d{6}|\d{6})\s*',
                text_upper
            )
        )

        if (
            (has_compact_positive or compact_only)
            and not has_compact_negative
        ):
            # YYYYMMDD
            pattern8 = re.findall(
                r'(?<!\d)'
                r'(20\d{2})'
                r'(\d{2})'
                r'(\d{2})'
                r'(?!\d)',
                text_upper
            )

            for year, month, day in pattern8:

                year = int(year)
                month = int(month)
                day = int(day)

                if not (MIN_YEAR <= year <= MAX_YEAR):
                    continue

                try:
                    dt = datetime(
                        year,
                        month,
                        day
                    )
                    found.append(dt)
                except:
                    pass

            # YYMMDD
            pattern9 = re.findall(
                r'(?<!\d)'
                r'(\d{2})'
                r'(\d{2})'
                r'(\d{2})'
                r'(?!\d)',
                text_upper
            )

            for yy, month, day in pattern9:

                yy = int(yy)
                month = int(month)
                day = int(day)

                year = 2000 + yy

                if not (MIN_YEAR <= year <= MAX_YEAR):
                    continue

                try:
                    dt = datetime(
                        year,
                        month,
                        day
                    )
                    found.append(dt)
                except:
                    pass

        return found

    # =========================
    # 4. 날짜 후보 점수 계산
    # =========================
    scored_dates = []

    for text in candidates:

        upper = text.upper()
        dates = extract_dates(text)

        if not dates:
            continue

        score = 0

        # 소비기한 관련 키워드
        for kw in strong_positive_keywords:
            if kw in upper:
                score += 10

        for kw in weak_positive_keywords:
            if kw in upper:
                score += 4

        # 제조 / 생산 / 포장 날짜 감점
        for kw in negative_keywords:
            if kw in upper:
                score -= 12

        # 종료일 우선
        if "까지" in text:
            score += 15

        # 시작일 감점
        if "부터" in text:
            score -= 8

        for dt in dates:
            scored_dates.append(
                (score, dt, text)
            )

    if not scored_dates:
        return None

    # =========================
    # 5. 가장 적절한 날짜 선택
    # =========================
    scored_dates.sort(
        key=lambda x: (x[0], x[1]),
        reverse=True
    )

    best_score, best_date, best_text = scored_dates[0]

    # 제조/생산 날짜처럼
    # 음수 점수 후보만 남은 경우는 사용하지 않음
    if best_score < 0:
        return None

    return best_date.strftime("%Y-%m-%d")


def extract_partial_date(texts):

    import re

    clean_texts = [
        str(t).strip()
        for t in texts
        if str(t).strip()
    ]

    # OCR이 키워드와 날짜를 서로 다른 박스로 나눌 수 있으므로
    # 원문 + 인접 2개 + 인접 3개를 같이 검사
    candidates = []

    for i, text in enumerate(clean_texts):

        candidates.append(text)

        if i + 1 < len(clean_texts):
            candidates.append(
                clean_texts[i]
                + " "
                + clean_texts[i + 1]
            )

        if i + 2 < len(clean_texts):
            candidates.append(
                clean_texts[i]
                + " "
                + clean_texts[i + 1]
                + " "
                + clean_texts[i + 2]
            )

    # =========================
    # 소비기한 관련 키워드
    # =========================
    positive_keywords = [
        "소비기한",
        "유통기한",
        "EXP",
        "EXPIRY",
        "EXP DATE",
        "BEST BEFORE",
        "BEST BY",
        "BEST IF USED BY",
        "USE BY",
        "BB",
        "까지"
    ]

    # 제조 / 생산 / 포장 날짜
    negative_keywords = [
        "제조일",
        "제조일자",
        "제조일시",
        "생산일",
        "생산일자",
        "포장일",
        "포장일자",
        "포장년",

        "MFG",
        "PROD",
        "PROD DATE",
        "PRODUCTION DATE",

        "PACK DATE",
        "PACKED",
        "PACKING DATE",

        "PD:",
        "P.D.",
        "P.D:"
    ]

    partial_candidates = []

    for text in candidates:

        upper = text.upper()

        # 소비기한 관련 문맥이 있는지 확인
        has_positive = any(
            kw in upper
            for kw in positive_keywords
        )

        # 제조/생산 문맥인지 확인
        has_negative = any(
            kw in upper
            for kw in negative_keywords
        )

        # 제조/생산/포장 날짜 문맥이면 제외
        if has_negative:
          continue

        # 소비기한 관련 키워드가 있으면 우선순위를 높임
        if has_positive:
            score = 10
        else:
            score = 0

        # "까지"는 종료일 의미가 강하므로 가점
        if "까지" in text:
            score += 5

        # =========================
        # A. MM/YYYY
        # 예: EXP 02/2023
        # =========================
        pattern_month_year = re.findall(
            r'(?<!\d)'
            r'(\d{1,2})'
            r'[\.\-/]'
            r'(20\d{2})'
            r'(?!\d)',
            upper
        )

        for month, year in pattern_month_year:

            month = int(month)
            year = int(year)

            if not (2015 <= year <= 2035):
                continue

            if not (1 <= month <= 12):
                continue

            partial_candidates.append({
                "score": score,
                "year": f"{year:04d}",
                "month": f"{month:02d}",
                "day": "NONE",
                "text": text
            })

        # =========================
        # A-1. MM/YY
        # 예: Exp.Date: 12/22
        # =========================
        pattern_month_year_short = re.findall(
            r'(?<!\d)'
            r'(\d{1,2})'
            r'\s*[\.\-/]\s*'
            r'(\d{2})'
            r'(?!\d)',
            upper
        )

        if has_positive:

            for month, yy in pattern_month_year_short:

                month = int(month)
                yy = int(yy)

                year = 2000 + yy

                if not (2015 <= year <= 2035):
                    continue

                if not (1 <= month <= 12):
                    continue

                partial_candidates.append({
                    "score": score + 2,
                    "year": f"{year:04d}",
                    "month": f"{month:02d}",
                    "day": "NONE",
                    "text": text
                })

        # =========================
        # A-2. MM YYYY
        # 예: Best before end: 12 2021
        # =========================
        pattern_month_year_space = re.findall(
            r'(?<!\d)'
            r'(\d{1,2})'
            r'\s+'
            r'(20\d{2})'
            r'(?!\d)',
            upper
        )

        for month, year in pattern_month_year_space:

            month = int(month)
            year = int(year)

            if not (2015 <= year <= 2035):
                continue

            if not (1 <= month <= 12):
                continue

            partial_candidates.append({
                "score": score,
                "year": f"{year:04d}",
                "month": f"{month:02d}",
                "day": "NONE",
                "text": text
            })

        # =========================
        # B. MM.DD / MM-DD / MM/DD
        # 연도 없음
        # 예: 유통기한 04.14
        # =========================

        # 소비기한 문맥이 있으면 한 자리 월/일도 허용
        if has_positive:
            pattern_month_day = re.findall(
                r'(?<!\d)'
                r'(\d{1,2})'
                r'\s*[\.\-/]\s*'
                r'(\d{1,2})'
                r'(?!\d)',
                upper
            )

        # 소비기한 문맥이 없으면 오탐 방지를 위해
        # 두 자리 월/일만 허용
        else:
            pattern_month_day = re.findall(
                r'(?<!\d)'
                r'(\d{2})'
                r'\s*[\.\-/]\s*'
                r'(\d{2})'
                r'(?!\d)',
                upper
            )

        for month, day in pattern_month_day:

            month = int(month)
            day = int(day)

            if not (1 <= month <= 12):
                continue

            if not (1 <= day <= 31):
                continue

            partial_candidates.append({
                "score": score,
                "year": "NONE",
                "month": f"{month:02d}",
                "day": f"{day:02d}",
                "text": text
            })

    if not partial_candidates:
        return None

    partial_candidates.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    best = partial_candidates[0]

    return {
        "year": best["year"],
        "month": best["month"],
        "day": best["day"]
    }
