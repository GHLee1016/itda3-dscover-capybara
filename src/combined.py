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

    def __init__(self, rapid_ocr, rules, fallback=None):
        self.ocr = rapid_ocr
        self.rules = rules            # src/team2_rules.py 모듈
        self.fallback = fallback      # pipeline.DatePipeline 또는 None
        self.stats = {"rapid": 0, "adaptive": 0, "fallback": 0, "none": 0}

    # ------------------------------------------------------------ RapidOCR 경로
    def _texts(self, img):
        try:
            r = self.ocr(img)
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

    def _rapid(self, img):
        if self.ocr is None:          # 가중치가 없으면 보조 경로만으로 간다
            return None
        small = downscale_max(img, self.MAX_SIDE)
        got = self._from_texts(self._texts(small))
        if got is not None:
            self.stats["rapid"] += 1
            return got
        # adaptive threshold 재시도 (팀원2 원본 2단계)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11)
        got = self._from_texts(self._texts(adaptive))
        if got is not None:
            self.stats["adaptive"] += 1
        return got

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
