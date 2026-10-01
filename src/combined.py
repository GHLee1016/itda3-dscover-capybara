"""
최종 파이프라인 — 계단(cascade) 구조.

## 한 줄 요약

**각 칸은 앞 칸이 실패한 이미지에만 돈다.** 그래서 칸을 추가해도 이미 맞힌
답이 깨지지 않는다. 회귀 위험이 구조적으로 0 이라 실험을 공격적으로 할 수 있었고,
수단 12가지를 재서 아래 7칸 + 보조 경로만 남겼다.

## 구조

    1칸  RapidOCR PP-OCRv6 det small @1024px          80% 가 여기서 끝난다
    2칸  adaptive 이진화 후 재시도                      2%
    3칸  det medium (더 큰 검출기)                      5%
    4칸  CLAHE 대비 보정 + det medium                   1%
    5칸  인식기 교체 — 한국어                           1%
    6칸  인식기 교체 — 영문                             1%
    7칸  도트 인쇄 전용 판독기 (직접 학습 CRNN+CTC)      1%
    보조  직접 학습 YOLOv8n 검출 → PP-OCR 인식           1%
          ↓
    날짜 규칙 엔진 (team2_rules) → year / month / day

## 왜 RapidOCR 이 주력인가 (홀드아웃2 198장 실측, EDA 63~64)

| 구성 | 부분점수 | 장당 |
| --- | ---: | ---: |
| 기존 (PP-OCRv3 det + 한국어 PP-OCRv4 rec + YOLOv8n) | 66.8% | 2.19초 |
| RapidOCR (PP-OCRv6 det/rec, ONNX) + 규칙 | **79.8%** | 0.36초 |
| **둘을 합침 (이 구조)** | **82.3%** | 약 0.7초 |

정확도와 속도가 동시에 좋다. 주된 원인은 모델 세대다 — RapidOCR 이 PP-OCR**v6**
det/rec 를 쓰고 기존 경로는 v3 det + v4 rec 다.

합치는 방식은 네 가지를 다 재보고 골랐다.
  A. RapidOCR 주, 실패 시 기존      82.3%  (개선 6 / 악화 **0**, p=0.031)  ← 채택
  B. RapidOCR 주, NONE 칸만 기존    82.2%  (개선 8 / 악화 3, p=0.227)
  C. 기존 주, 실패 시 RapidOCR      78.6%
  D. 셋 다수결                      69.2%

A 는 점수가 가장 높으면서 **악화가 0장**이다. B 는 필드를 섞다가 3장을 망쳤다.

## 비용

RapidOCR 은 onnxruntime 기반이라 torch 를 싣지 않는다. torch 가 환경에 있으면
OpenMP 충돌로 2.4배 느려진다(EDA 52~53). 날짜 검출기(YOLOv8n)도 같은
onnxruntime 위에서 돈다. 최종 속도는 **500장 순차 503초 = 장당 1.01초**.

## 기각한 것

기각 사유와 코드는 `src/rejected_rungs.py` 와 JOURNAL.md 에 남겼다.
det server 칸 · 1600px 칸 · 8칸 전부 실행 후 다수결 · YOLO 도트 파인튜닝 ·
검출 문턱 완화 · 전처리 8종.
"""
import os

import numpy as np
import cv2


RAPID_FILES = {
    "Det.model_path": "PP-OCRv6_det_small.onnx",
    "Rec.model_path": "PP-OCRv6_rec_small.onnx",
    "Cls.model_path": "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
}

RETRY_DET = "PP-OCRv6_det_medium.onnx"
REC_KOREAN = "korean_PP-OCRv5_rec_mobile.onnx"
REC_EN = "en_PP-OCRv5_rec_mobile.onnx"


def _make_rapid(weights_dir, override=None):
    """
    RapidOCR 을 만든다. **weights_dir 에 있는 파일만** 쓴다.

    ⚠️ 절대로 자동 다운로드로 넘어가지 않는다. 채점 서버는 인터넷이 차단되어 있고,
       실행 중 가중치를 받으려 하면 그 자리에서 예외가 나 노트북이 멈춘다(정량 0점).
       운영진이 채점 전에 download_weights.sh 를 1회 실행해 weights/ 를 채워 준다.
       `RapidOCR()` 를 인자 없이 부르면 모델을 받으러 나가므로 그 경로를 막아 둔다.

    override 로 특정 모델만 바꾼다 (검출기 또는 인식기).
    반환: (RapidOCR|None, 상태문자열)
    """
    need = {k: os.path.join(weights_dir, v) for k, v in RAPID_FILES.items()}
    if override:
        need.update({k: os.path.join(weights_dir, v)
                     for k, v in override.items()})
    missing = [os.path.basename(p) for p in need.values() if not os.path.exists(p)]
    if missing:
        return None, "가중치 없음: " + ", ".join(missing)
    try:
        from rapidocr import RapidOCR
        return RapidOCR(params=need), "bundled"
    except Exception as e:
        return None, "로딩 실패: %s" % e


def build_rapid_ocr(weights_dir):
    """1칸·2칸용 주력 엔진 (det small + rec small)."""
    return _make_rapid(weights_dir)


def build_rapid_retry(weights_dir):
    """
    3칸·4칸용. 검출기만 medium(59MB)으로 올린다.

    검출기만 바꾸고 인식기·분류기는 small 을 그대로 쓴다. EDA 74 실측에서
    검출기 교체가 69장을 회수한 반면 인식기 교체는 19장이었다.

    파일이 없으면 None 을 돌려주고 그 칸은 건너뛴다(홀드아웃2 87.2% → 83.5%).
    """
    return _make_rapid(weights_dir, {"Det.model_path": RETRY_DET})


def build_rapid_rec(weights_dir, rec_file):
    """
    5칸·6칸용. 인식기만 바꾸고 검출기는 small 그대로 쓴다.

    왜 필요한가 (EDA 101~102):
      주력 인식기 `PP-OCRv6_rec_small` 은 중국어 모델이라 **한글을 제대로 읽지
      못한다.** 같은 이미지에서
          rec_small   `内213-3，`  `苦号卫电立20130628332176`
          rec_korean  `별자치도 제주시 첨단로 213-3,`  `품목보고번호 20130628332176`
      날짜는 숫자라 둘 다 읽지만, 한글이 날짜에 **바로 붙어 있으면**
      (`나 2026.03.11 까지 홍병현`, `유통2023.01.05까지`) 한글을 엉뚱한 한자로
      읽으면서 날짜까지 함께 깨뜨린다. 한국어·영문 모델은 그 글자를 제대로 읽어
      날짜가 온전히 남는다.

      580장에서 이 두 칸이 6장을 회수하고 1장을 망가뜨렸다. 순 +5장, 비용 1.1배.
    """
    return _make_rapid(weights_dir, {"Rec.model_path": rec_file})


def _filled(fields):
    """연·월·일 중 실제로 채워진 필드 수. None 이면 0."""
    if not fields:
        return 0
    return sum(1 for k in ("year", "month", "day")
               if str(fields.get(k, "NONE")) not in ("NONE", "", "None"))


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
    계단형 파이프라인. 각 칸은 앞 칸이 실패한 이미지에만 돈다.

    인자는 전부 선택이다. 없으면 그 칸을 건너뛴다 — 가중치가 빠진 환경에서도
    멈추지 않고 강하(degrade)한다.
        rapid_ocr    1·2칸   없으면 보조 경로 단독으로 동작
        rapid_retry  3·4칸   det medium
        rec_korean   5칸
        rec_en       6칸
        dot+dot_boxes 7칸    둘 다 있어야 돈다
        fallback     보조    pipeline.DatePipeline
    """

    MAX_SIDE = 1024

    def __init__(self, rapid_ocr, rules, fallback=None, rapid_retry=None,
                 rec_korean=None, rec_en=None, dot=None, dot_boxes=None):
        self.ocr = rapid_ocr
        self.rules = rules            # src/team2_rules.py 모듈
        self.fallback = fallback      # pipeline.DatePipeline 또는 None
        # 더 큰 검출기를 쓰는 재시도 칸. **실패한 이미지에만** 돈다 —
        # 전면 교체하면 맞히던 것을 망가뜨린다(EDA 75 에서 악화 2장 확인).
        self.retry = rapid_retry
        # 인식기만 바꾼 재시도 칸. 한글·영문이 날짜에 붙어 있을 때 살린다.
        self.rec_korean = rec_korean
        self.rec_en = rec_en
        # 도트 매트릭스 전용 판독기와 그 영역을 찾을 검출기(YOLO).
        self.dot = dot
        self.dot_boxes = dot_boxes
        self.stats = {"rapid": 0, "adaptive": 0, "retry": 0, "clahe": 0,
                      "rec_ko": 0, "rec_en": 0, "dot": 0,
                      "partial": 0, "fallback": 0, "none": 0}

    # ------------------------------------------------------------ RapidOCR 경로
    def _texts(self, img, ocr=None):
        engine = ocr if ocr is not None else self.ocr
        try:
            r = engine(img)
        except Exception:
            return []
        return list(r.txts) if getattr(r, "txts", None) else []

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

        # ⚠️ 이 프로젝트에서 가장 큰 버그가 여기 있었다 (EDA 89~90).
        #    예전에는 각 칸이 완전한 날짜와 **부분 결과**를 함께 반환했다.
        #    부분 결과도 None 이 아니라서 계단이 거기서 멈췄고, 다음 칸이
        #    온전한 날짜를 읽을 수 있는데도 반쪽짜리에 머물렀다.
        #      002335  1차 `2021:11.19` → 부분 `NONE-11-19` 로 멈춤
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

        # ── 1칸. 여기서 80% 가 끝난다.
        texts = self._texts(small)
        got = self._full(texts)
        if got is not None:
            self.stats["rapid"] += 1
            return got
        keep_partial(texts)

        # ── 2칸. adaptive 이진화. 조명 얼룩이 있는 라벨에서 살아난다.
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11)
        texts = self._texts(adaptive)
        got = self._full(texts)
        if got is not None:
            self.stats["adaptive"] += 1
            return got
        keep_partial(texts)

        # ── 3칸. 더 큰 검출기.
        #
        # 왜 전면 교체가 아니라 계단인가 (EDA 74~75 실측):
        #   det medium 을 처음부터 쓰면 라벨 179장에서 85.1% → 89.4% 로 오르지만
        #   **맞히던 2장을 망가뜨린다**(`002892` 2021-03-31 → 2021-05-NONE).
        #   실패분에만 걸면 그 회귀가 구조적으로 0 이 되고 미검출만 줄어든다.
        if self.retry is not None:
            texts = self._texts(small, self.retry)
            got = self._full(texts)
            if got is not None:
                self.stats["retry"] += 1
                return got
            keep_partial(texts)

        # ── 4칸. 국소 대비를 올리고(CLAHE) det medium.
        #
        # 잉크가 옅거나 점이 끊겨 찍힌 인쇄가 여기서 살아난다. EDA 91 에서
        # 전처리 열한 가지를 겨뤄 본 결과가 이것이었다 — 아무 답도 못 내던
        # 23장 중 3장 회수. 나머지 20장은 어떤 전처리로도 살아나지 않았다.
        if self.retry is not None:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
            texts = self._texts(cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR), self.retry)
            got = self._full(texts)
            if got is not None:
                self.stats["clahe"] += 1
                return got
            keep_partial(texts)

        # ── 5·6칸. 인식기만 바꿔 다시 읽는다 (한국어 → 영문).
        #
        # 주력 인식기는 중국어 모델이라 한글을 엉뚱한 한자로 읽고, 그 글자가
        # 날짜에 붙어 있으면 날짜까지 함께 깨뜨린다(build_rapid_rec 주석 참고).
        #   000528 `나 2026.03.11 까지 홍병현`  →  NONE-03-11 에 머물렀다
        #   002400 `유통2023.01.05까지`         →  아무것도 못 냈다
        for eng, key in ((self.rec_korean, "rec_ko"), (self.rec_en, "rec_en")):
            if eng is None:
                continue
            texts = self._texts(small, eng)
            got = self._full(texts)
            if got is not None:
                self.stats[key] += 1
                return got
            keep_partial(texts)

        # ── 7칸. 도트 매트릭스 전용 판독기.
        #
        # 일반 인식기는 그 글꼴을 학습한 적이 없어 `26.08.13` 을 `2:88:13` 으로
        # 읽는다(EDA 107). 그래서 도트 날짜를 합성해 CTC 시퀀스 모델을 따로
        # 학습했다(EDA 111). 글자를 자르지 않고 줄 단위로 통째로 읽는다 —
        # 자르는 방식은 여덟 글자가 전부 맞아야 해서 실패했다(EDA 110: 정답 0장).
        #
        # ⚠️ 이 칸은 **손해를 끼칠 수 있다.** 앞 칸들이 반쪽짜리(`NONE-08-25`)를
        #    모아 둔 상태일 수 있고, 그건 이미 두 필드를 맞힌 것이다. 도트
        #    판독기가 엉뚱한 날짜를 내면 그 두 필드까지 잃는다. 실제로 처음
        #    붙였을 때 홀드아웃4 가 93.0% → 92.8% 로 내려갔다.
        #    그래서 **채운 필드 수가 늘어날 때만** 바꿔 끼운다.
        if self.dot is not None and self.dot_boxes is not None:
            got = self._dot_read(img)
            if got is not None and _filled(got) > _filled(partial):
                self.stats["dot"] += 1
                return got

        # 어느 칸도 온전한 날짜를 못 냈다. 그제서야 반쪽짜리를 쓴다.
        if partial is not None:
            self.stats["partial"] += 1
        return partial

    def _dot_read(self, img):
        """도트 판독기 경로. 실패하면 None."""
        try:
            h, w = img.shape[:2]
            side = max(h, w)
            canvas = np.zeros((side, side, 3), np.uint8)
            canvas[:h, :w] = img
            dets = self.dot_boxes if isinstance(self.dot_boxes, (list, tuple)) \
                else [self.dot_boxes]
            boxes = []
            for dd in dets:
                if dd is None:
                    continue
                try:
                    # 문턱을 0.05 로 내리면 검출률이 4%→65% 로 오른다. 그런데
                    # 끝단 정확도는 92.3%→92.2% 로 움직이지 않았다(EDA 119).
                    # 검출은 병목이 아니었다. 0.25 를 유지한다.
                    boxes += list(dd.detect(canvas, conf=0.25))
                except Exception:
                    pass
            if not boxes:
                return None
            texts = []
            for x1, y1, x2, y2 in boxes[:5]:
                bw, bh = x2 - x1, y2 - y1
                pad = 0.12
                X1 = max(0, int(x1 - bw * pad)); Y1 = max(0, int(y1 - bh * pad))
                X2 = min(w, int(x2 + bw * pad)); Y2 = min(h, int(y2 + bh * pad))
                if X2 - X1 < 12 or Y2 - Y1 < 8:
                    continue
                texts += self.dot.read(
                    cv2.cvtColor(img[Y1:Y2, X1:X2], cv2.COLOR_BGR2GRAY))
            if not texts:
                return None
            full = self.rules.choose_expiration_date(texts)
            if full is not None:
                y, m, d = full.split("-")
                return {"year": y, "month": m, "day": d}
            # `2022.07` 처럼 월·연만 있는 경우도 받는다
            return self.rules.extract_partial_date(texts)
        except Exception:
            return None

    # ------------------------------------------------------------- 보조 경로
    def _fallback(self, path, img):
        """기존 파이프라인(PP-OCR + YOLO). 앞이 전부 실패했을 때만 부른다."""
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
