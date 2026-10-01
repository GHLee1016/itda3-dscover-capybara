"""
합친 파이프라인 — RapidOCR 을 주력으로, 기존 파이프라인을 보조로.

## 왜 이 구조인가 (홀드아웃2 198장 실측, EDA 63~64)

| 구성 | 부분점수 | 완전일치 | 장당 |
| --- | ---: | ---: | ---: |
| 기존 (PP-OCRv3 det + 한국어 PP-OCRv4 rec + YOLOv8n) | 66.8% | 59.1% | 2.19초 |
| RapidOCR (PP-OCRv6 det/rec, ONNX) + 팀원2 규칙 | **79.8%** | 75.8% | 0.36초 |
| **둘을 합침 (아래 구조)** | **82.3%** | **77.8%** | 약 0.7초 |

RapidOCR 단독이 기존 대비 **+13.0%p** (개선 49장 / 악화 14장, McNemar p=0.0000).
정확도와 속도가 동시에 좋다. 주된 원인은 모델 세대로 보인다 —
RapidOCR 이 PP-OCR**v6** det/rec 를 쓰고, 기존 경로는 v3 det + v4 rec 다.

합치는 방식은 네 가지를 다 재보고 골랐다.
  A. RapidOCR 주, 실패 시 기존      82.3%  (개선 6 / 악화 **0**, p=0.031)  ← 채택
  B. RapidOCR 주, NONE 칸만 기존    82.2%  (개선 8 / 악화 3, p=0.227)
  C. 기존 주, 실패 시 RapidOCR      78.6%  (RapidOCR 단독보다 낮다)
  D. 셋 다수결                      69.2%  (크게 나쁘다)

A 를 고른 이유는 점수가 가장 높으면서 **악화가 0장**이기 때문이다.
B 는 필드를 섞다가 3장을 망쳤다. C 는 열등한 쪽을 주력에 두어 손해가 났다.

## 상호보완성

기존이 전부 NONE 으로 낸 48장 중 RapidOCR 이 25장을 건진다(맞은 필드 69개).
반대로 RapidOCR 이 놓친 29장 중 기존이 6장을 건진다(15개). 양이 적어 보이지만
보조 경로는 **이미 실패한 이미지에만** 도므로 잃을 것이 없다.

## 비용

RapidOCR 은 onnxruntime 기반이라 torch 를 싣지 않는다. 이것이 팀원1(EasyOCR)
경로를 기각한 이유(torch 임포트만으로 2.4배 저하, EDA 52~53)를 피해 가는 지점이다.
날짜 검출기(YOLOv8n)도 같은 onnxruntime 위에서 돈다.
"""
import os

import numpy as np
import cv2


RAPID_FILES = {
    "Det.model_path": "PP-OCRv6_det_small.onnx",
    "Rec.model_path": "PP-OCRv6_rec_small.onnx",
    "Cls.model_path": "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
}


def build_rapid_ocr(weights_dir):
    """
    RapidOCR 을 만든다. **weights_dir 에 있는 파일만** 쓴다.

    ⚠️ 절대로 자동 다운로드로 넘어가지 않는다. 채점 서버는 인터넷이 차단되어 있고,
       실행 중 가중치를 받으려 하면 그 자리에서 예외가 나 노트북이 멈춘다(정량 0점).
       운영진이 채점 전에 download_weights.sh 를 1회 실행해 weights/ 를 채워 준다.
       `RapidOCR()` 를 인자 없이 부르면 모델을 받으러 나가므로 그 경로를 막아 둔다.

    반환: (RapidOCR|None, 상태문자열)
    """
    paths = {k: os.path.join(weights_dir, v) for k, v in RAPID_FILES.items()}
    missing = [os.path.basename(p) for p in paths.values() if not os.path.exists(p)]
    if missing:
        return None, "가중치 없음: " + ", ".join(missing)
    try:
        from rapidocr import RapidOCR
        return RapidOCR(params=paths), "bundled"
    except Exception as e:
        return None, "로딩 실패: %s" % e


RETRY_DET = "PP-OCRv6_det_medium.onnx"
RETRY2_DET = "ch_PP-OCRv5_det_server.onnx"


def build_rapid_retry2(weights_dir):
    """
    세 번째 칸: ch PP-OCRv5 det server (84MB).

    ⚠️ **기각됐다 (EDA 76).** 배포에는 쓰지 않는다. 코드는 근거와 함께 남긴다.

    기각 경위: EDA 74 에서 '어려운 실패 252장' 중 det medium 이 못 잡은 17장을
    server 가 잡았다. 그래서 계단을 하나 더 두면 이득이 있을 줄 알았다.
    그런데 홀드아웃2 에서 **0장**이었다.

    이유는 모집단 착각이다. 17/252 는 medium 까지 실패한 이미지 기준이고,
    홀드아웃2 에서 그 단계까지 내려오는 이미지는 9장뿐이다. 기댓값이 0.6장이라
    0장은 예상 범위다. '17장 더 회수' 라는 표현 자체가 잘못된 프레이밍이었다.

    비용은 모델 84MB 와 장당 3.01 → 3.15초. 관측되지 않은 이득에 지불할 값이 아니다.
    """
    det = os.path.join(weights_dir, RETRY2_DET)
    need = {k: os.path.join(weights_dir, v) for k, v in RAPID_FILES.items()}
    need["Det.model_path"] = det
    missing = [os.path.basename(p) for p in need.values() if not os.path.exists(p)]
    if missing:
        return None, "가중치 없음: " + ", ".join(missing)
    try:
        from rapidocr import RapidOCR
        return RapidOCR(params=need), "bundled"
    except Exception as e:
        return None, "로딩 실패: %s" % e


def build_rapid_retry(weights_dir):
    """
    더 큰 검출기(PP-OCRv6 det medium, 59MB)를 쓰는 RapidOCR. 재시도 단계 전용이다.

    ⚠️ **weights_dir 의 파일만** 쓴다. 자동 다운로드로 넘어가지 않는다.
       채점 서버는 오프라인이라 실행 중 모델을 받으려 하면 그 자리에서 죽는다.
       파일이 없으면 None 을 돌려주고, 파이프라인은 재시도 단계 없이 동작한다
       (홀드아웃2 기준 87.2% → 83.5%).

    검출기만 medium 으로 바꾸고 인식기·분류기는 기존 small 을 그대로 쓴다.
    EDA 74 실측에서 검출기 교체가 69장을 회수한 반면 인식기 교체는 19장이었다.
    """
    det = os.path.join(weights_dir, RETRY_DET)
    need = {k: os.path.join(weights_dir, v) for k, v in RAPID_FILES.items()}
    need["Det.model_path"] = det
    missing = [os.path.basename(p) for p in need.values() if not os.path.exists(p)]
    if missing:
        return None, "가중치 없음: " + ", ".join(missing)
    try:
        from rapidocr import RapidOCR
        return RapidOCR(params=need), "bundled"
    except Exception as e:
        return None, "로딩 실패: %s" % e


REC_KOREAN = "korean_PP-OCRv5_rec_mobile.onnx"
REC_EN = "en_PP-OCRv5_rec_mobile.onnx"


def build_rapid_rec(weights_dir, rec_file):
    """
    인식기만 바꾼 RapidOCR. 검출기는 small 그대로 쓴다. 재시도 단계 전용이다.

    왜 필요한가 (EDA 101~102):
      주력 인식기 `PP-OCRv6_rec_small` 은 중국어 모델이라 **한글을 제대로 읽지
      못한다.** 같은 이미지에서
          rec_small   `内213-3，`  `苦号卫电立20130628332176`
          rec_korean  `별자치도 제주시 첨단로 213-3,`  `품목보고번호 20130628332176`
      날짜는 숫자라 둘 다 읽지만, 한글이 날짜에 **바로 붙어 있으면**
      (`나 2026.03.11 까지 홍병현`, `유통2023.01.05까지`) 한글을 엉뚱한 한자로
      읽으면서 날짜까지 함께 깨뜨린다. 한국어·영문 모델은 그 글자를 제대로 읽어
      날짜가 온전히 남는다.

      580장에서 이 두 칸이 6장을 회수하고 1장을 망가뜨렸다(오탐). 순 +5장,
      필드 정확도 +0.9%p, 비용 1.1배.

    ⚠️ 여기서도 **weights_dir 의 파일만** 쓴다. 없으면 None 이고 그 칸은 건너뛴다.
    """
    rec = os.path.join(weights_dir, rec_file)
    need = {k: os.path.join(weights_dir, v) for k, v in RAPID_FILES.items()}
    need["Rec.model_path"] = rec
    missing = [os.path.basename(p) for p in need.values() if not os.path.exists(p)]
    if missing:
        return None, "가중치 없음: " + ", ".join(missing)
    try:
        from rapidocr import RapidOCR
        return RapidOCR(params=need), "bundled"
    except Exception as e:
        return None, "로딩 실패: %s" % e


def imread_any(path):
    """한글 경로 대응. cv2.imread 는 비 ASCII 경로에서 None 을 준다."""
    try:
        return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def downscale_max(img, max_side):
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s >= 1.0:
        return img
    return cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))),
                      interpolation=cv2.INTER_AREA)


class CombinedPipeline:
    """
    1) RapidOCR 로 읽고 팀원2 규칙으로 날짜를 고른다 (완전 날짜 → 부분 날짜)
    2) 실패하면 adaptive threshold 로 한 번 더 (팀원2 원본과 동일)
    3) 그래도 전부 NONE 이면 기존 파이프라인(PP-OCR + YOLO)에 넘긴다

    `fallback` 이 None 이면 RapidOCR 단독으로 동작한다 (79.8% 구성).
    """

    MAX_SIDE = 1024          # 팀원2 원본과 동일
    # 마지막 두 칸에서만 쓰는 큰 쪽. RapidOCR 은 내부에서 Global.max_side_len
    # 2000 으로 한 번 더 자르므로 그 아래로 잡는다.
    BIG_SIDE = 1600

    def __init__(self, rapid_ocr, rules, fallback=None, rapid_retry=None,
                 rapid_retry2=None, big_rungs=False,
                 rec_korean=None, rec_en=None):
        self.ocr = rapid_ocr
        self.rules = rules            # src/team2_rules.py 모듈
        self.fallback = fallback      # pipeline.DatePipeline 또는 None
        # 인식기만 바꾼 재시도 칸. 한글·영문이 날짜에 붙어 있을 때 살린다.
        self.rec_korean = rec_korean
        self.rec_en = rec_en
        # 더 큰 검출기를 쓰는 재시도 단계들. **실패한 이미지에만** 돈다 —
        # 전면 교체하면 맞히던 것을 망가뜨린다(EDA 75에서 악화 2장 확인).
        self.retry = rapid_retry       # PP-OCRv6 det medium
        self.retry2 = rapid_retry2     # ch PP-OCRv5 det server
        # 1600px 칸. **기본으로 끈다** — 아래 _rapid 의 기각 사유를 볼 것.
        self.big_rungs = big_rungs
        self.stats = {"rapid": 0, "adaptive": 0, "retry": 0, "clahe": 0,
                      "rec_ko": 0, "rec_en": 0, "big_med": 0, "big_small": 0,
                      "retry2": 0, "partial": 0, "fallback": 0, "none": 0}

    # ------------------------------------------------------------ RapidOCR 경로
    def _texts(self, img, ocr=None):
        engine = ocr if ocr is not None else self.ocr
        try:
            r = engine(img)
        except Exception:
            return []
        return list(r.txts) if getattr(r, "txts", None) else []

    def _from_texts(self, texts):
        if not texts:
            return None
        full = self.rules.choose_expiration_date(texts)
        if full is not None:
            y, m, d = full.split("-")
            return {"year": y, "month": m, "day": d}
        return self.rules.extract_partial_date(texts)

    def _full(self, texts):
        """완전한 날짜만. 부분 결과는 내지 않는다."""
        if not texts:
            return None
        full = self.rules.choose_expiration_date(texts)
        if full is None:
            return None
        y, m, d = full.split("-")
        return {"year": y, "month": m, "day": d}

    def _rapid(self, img):
        if self.ocr is None:          # 가중치가 없으면 보조 경로만으로 간다
            return None
        small = downscale_max(img, self.MAX_SIDE)

        # ⚠️ 예전에는 각 칸에서 _from_texts 를 불렀다. 그 함수는 완전한 날짜가
        #    없으면 **부분 결과**를 내는데, 부분 결과도 None 이 아니라서 계단이
        #    거기서 멈췄다. 그래서 다음 칸에서 온전한 날짜를 읽을 수 있는데도
        #    반쪽짜리에 머물렀다(EDA 89 에서 확인).
        #      002335  OCR 1차 `2021:11.19` → 부분 `NONE-11-19` 로 멈춤
        #              det medium 은 `2021.11.19` 를 제대로 읽는다
        #    이제 각 칸은 **완전한 날짜만** 성공으로 치고, 부분 결과는 따로
        #    모아 두었다가 끝까지 실패했을 때만 쓴다. 잃는 것은 없다.
        partial = None

        def keep_partial(texts):
            nonlocal partial
            if partial is None and texts:
                p = self.rules.extract_partial_date(texts)
                if p is not None:
                    partial = p

        texts = self._texts(small)
        got = self._full(texts)
        if got is not None:
            self.stats["rapid"] += 1
            return got
        keep_partial(texts)

        # adaptive threshold 재시도 (팀원2 원본 2단계)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11)
        texts = self._texts(adaptive)
        got = self._full(texts)
        if got is not None:
            self.stats["adaptive"] += 1
            return got
        keep_partial(texts)

        # 더 큰 검출기로 한 번 더. 여기까지 온 이미지는 어차피 NONE 이므로 잃을 것이 없다.
        #
        # 왜 전면 교체가 아니라 계단인가 (EDA 74~75 실측):
        #   det medium 을 처음부터 쓰면 라벨 179장에서 85.1% → 89.4% 로 오르지만
        #   **맞히던 2장을 망가뜨린다**(`002892` 2021-03-31 → 2021-05-NONE).
        #   실패분에만 걸면 그 회귀가 구조적으로 0 이 되고, 미검출만 줄어든다.
        #   비용도 실패한 8.6% 에만 든다.
        if self.retry is not None:
            texts = self._texts(small, self.retry)
            got = self._full(texts)
            if got is not None:
                self.stats["retry"] += 1
                return got
            keep_partial(texts)

        # 네 번째 칸: 국소 대비를 올리고(CLAHE) det medium.
        #
        # 잉크가 옅거나 도트 프린터로 점이 끊겨 찍힌 인쇄가 여기서 살아난다.
        # EDA 91 에서 전처리 열한 가지를 겨뤄 본 결과가 이것이었다.
        #   아무 답도 못 내던 23장 중 3장 회수 (닫기 연산 1장, 언샤프 2장)
        # 이 칸도 실패분에만 도니 악화는 구조적으로 0 이다.
        #
        # 기대를 낮춰 적어 둔다: 나머지 20장은 어떤 전처리로도 살아나지 않았다.
        if self.retry is not None:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
            texts = self._texts(cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR), self.retry)
            got = self._full(texts)
            if got is not None:
                self.stats["clahe"] += 1
                return got
            keep_partial(texts)

        # 다섯·여섯 번째 칸: 인식기만 바꿔 다시 읽는다 (한국어 → 영문).
        #
        # 주력 인식기는 중국어 모델이라 한글을 엉뚱한 한자로 읽고, 그 글자가
        # 날짜에 붙어 있으면 날짜까지 함께 깨뜨린다(build_rapid_rec 주석 참고).
        #   000528 `나 2026.03.11 까지 홍병현`  →  NONE-03-11 에 머물렀다
        #   002400 `유통2023.01.05까지`         →  아무것도 못 냈다
        # 이 칸들도 실패분에만 돈다. 580장에서 개선 6장 / 악화 1장(오탐)이었다.
        for eng, key in ((self.rec_korean, "rec_ko"), (self.rec_en, "rec_en")):
            if eng is None:
                continue
            texts = self._texts(small, eng)
            got = self._full(texts)
            if got is not None:
                self.stats[key] += 1
                return got
            keep_partial(texts)

        # 1600px 칸 — **기각했다.** 기본값 False 라 돌지 않는다.
        #
        # 장당 예산 4.8초 중 0.81초만 쓰고 있어, 속도를 내주고 정확도를 사려 했다.
        #   EDA 99  여덟 칸을 다 돌렸을 때의 오라클 상한이 94.9% 로 현재(93.6%)보다
        #           1.3%p 뿐이었고, 라벨 없이 고르는 다수결은 오히려 더 나빴다(93.4%).
        #   EDA 100 순서를 바꿔 시뮬레이션하니 이 두 칸을 끝에 붙이는 것이
        #           유일하게 이득으로 보였다(+0.6%p · 비용 1.16배, 580장).
        #   EDA 96  그런데 홀드아웃4 148장에서는 **한 장도 건지지 못했다.**
        #           91.9% → 91.9%, 미검출 8장 그대로. 비용만 30% 늘었다.
        #
        # 580장은 오답을 들여다보며 고친 오염된 표본이다. 거기서 고른 개선이
        # 새 표본에서 재현되지 않았다. 코드는 근거로 남기고 기본값만 끈다.
        if self.big_rungs:
            big = downscale_max(img, self.BIG_SIDE)
            for eng, key in ((self.retry, "big_med"), (self.ocr, "big_small")):
                if eng is None:
                    continue
                texts = self._texts(big, eng)
                got = self._full(texts)
                if got is not None:
                    self.stats[key] += 1
                    return got
                keep_partial(texts)

        # det server 는 74단계에서 17장을 더 건진다고 봤으나 홀드아웃2 에서
        # 0 장이어서 기각했다. 기본값은 None 이라 돌지 않는다.
        if self.retry2 is not None:
            texts = self._texts(small, self.retry2)
            got = self._full(texts)
            if got is not None:
                self.stats["retry2"] += 1
                return got
            keep_partial(texts)

        # 어느 칸도 온전한 날짜를 못 냈다. 그제서야 반쪽짜리를 쓴다.
        if partial is not None:
            self.stats["partial"] += 1
        return partial

    # ------------------------------------------------------------- 보조 경로
    def _fallback(self, path, img):
        """기존 파이프라인. RapidOCR 이 아무것도 못 냈을 때만 부른다."""
        if self.fallback is None:
            return None
        try:
            fin, _, _ = self.fallback.predict_image(img)
        except Exception:
            return None
        if not fin:
            return None
        out = {}
        for k in ("year", "month", "day"):
            v = fin.get(k[0])
            out[k] = "NONE" if v is None else (
                "%04d" % v if k == "year" else "%02d" % v)
        if all(v == "NONE" for v in out.values()):
            return None
        self.stats["fallback"] += 1
        return out

    # ------------------------------------------------------------------ 진입점
    def predict(self, path, img=None):
        """-> {"year":..,"month":..,"day":..}  값은 전부 문자열, 미검출은 'NONE'."""
        if img is None:
            img = imread_any(path)
        if img is None:
            self.stats["none"] += 1
            return {"year": "NONE", "month": "NONE", "day": "NONE"}

        got = self._rapid(img)
        if got is None:
            got = self._fallback(path, img)
        if got is None:
            self.stats["none"] += 1
            return {"year": "NONE", "month": "NONE", "day": "NONE"}

        return {k: str(got.get(k, "NONE") or "NONE") for k in ("year", "month", "day")}


def to_row(image_id, fields):
    """
    제출 행. year/month/day 는 독립 채점이라 확인된 것만 채운다.
    셋 다 NONE 이면 final_date 는 단일 'NONE'.
    """
    y = fields.get("year", "NONE") or "NONE"
    m = fields.get("month", "NONE") or "NONE"
    d = fields.get("day", "NONE") or "NONE"
    if m != "NONE":
        m = str(m).zfill(2)
    if d != "NONE":
        d = str(d).zfill(2)
    final = "NONE" if (y == "NONE" and m == "NONE" and d == "NONE") else f"{y}-{m}-{d}"
    return {"image_id": image_id, "year": y, "month": m, "day": d,
            "final_date": final}
