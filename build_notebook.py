"""
predict.ipynb 생성기.

대회 규정상 채점 대상은 predict.ipynb **단 하나**다. src/pipeline.py 를 import 하는
구조로 두면 저장소 체크아웃 상태에 의존하게 되므로, 코드를 노트북 안에 직접 넣어
**노트북 단독으로 Run All 이 되도록** 만든다.

여기서 생성하므로 로직의 원본은 언제나 src/pipeline.py 하나뿐이다 (중복 관리 없음).

    python build_notebook.py
"""
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src", "pipeline.py")          # 보조 경로 (PP-OCR + YOLO)
SRC_RULES = os.path.join(ROOT, "src", "team2_rules.py")  # 주력 규칙 (팀원2 원문)
SRC_COMB = os.path.join(ROOT, "src", "combined.py")      # 두 경로를 잇는 층
SRC_DSEG = os.path.join(ROOT, "src", "dotseg.py")        # 도트 인쇄 줄 분리
SRC_DCRNN = os.path.join(ROOT, "src", "dotcrnn.py")      # 도트 판독기 (CTC)
OUT = os.path.join(ROOT, "predict.ipynb")

# 규정 §7: 첫 코드셀은 반드시 이 형태. os.environ.get 을 그대로 유지해야 한다.
CELL_CONFIG = '''# ===== CONFIG =====
import os
INPUT_DIR   = os.environ.get("ITDA_INPUT_DIR",  "./val_images")
OUTPUT_PATH = os.environ.get("ITDA_OUTPUT_PATH", "./submission.csv")
# ==================
'''

CELL_SETUP = '''import os, sys, time, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import cv2

print("input :", INPUT_DIR)
print("output:", OUTPUT_PATH)
'''

CELL_MODEL = '''# 모델 로딩 — 오프라인 전용
#
# ⚠️ 채점 서버는 인터넷이 차단되어 있다. 이 셀은 **weights/ 에 이미 있는 파일만 읽는다.**
#    어떤 경로로도 네트워크에 나가지 않는다 — 실행 중 가중치를 받으려 하면
#    그 자리에서 예외가 나 노트북이 멈추고 정량 0점이 된다.
#    가중치는 운영진이 채점 전에 `bash download_weights.sh` 를 1회 실행해 채운다.
#
# weights/ 는 이 노트북이 있는 폴더 기준이다. nbconvert 는 커널의 작업 디렉터리를
# 노트북 위치로 잡아 주므로, 어느 경로에서 실행하든 저장소의 weights/ 를 가리킨다.
LOCAL = os.path.join(os.getcwd(), "weights")

def _need(*parts):
    """weights/ 아래 경로. 없으면 None 을 돌려주고 호출 쪽에서 건너뛴다."""
    p = os.path.join(LOCAL, *parts)
    return p if os.path.exists(p) else None


# PaddleOCR 의 C++ 백엔드는 **비ASCII 경로의 모델 파일을 열지 못한다.**
# 같은 파일을 두 경로에 두고 확인했다.
#     C:\\Users\\Public\\ocr_models        로딩 성공
#     ...\\연합 학술제\\weights            (NotFound) Cannot open file ...
# 그리고 조용히 실패한다 — 아래 try/except 가 보조 경로만 끄고 넘어가므로
# 에러 없이 정확도만 내려간다(248장에서 6장). 저장소를 한글 폴더에 풀면
# 이 일이 그대로 일어난다.
#
# 그래서 모델 경로가 비ASCII 면 ASCII 임시 폴더로 복사해 그 경로를 넘긴다.
# **ASCII 환경에서는 아무 일도 하지 않으므로 동작이 바뀌지 않는다.**
def _ascii_safe(path):
    if path is None:
        return None
    try:
        path.encode("ascii")
        return path                      # ASCII — 그대로 쓴다
    except UnicodeEncodeError:
        pass
    import shutil
    import tempfile
    global _ASCII_TMP
    try:
        _ASCII_TMP
    except NameError:
        # tempfile 기본 경로도 사용자명이 비ASCII 면 함께 막힌다
        # (Windows 의 %TEMP% 는 사용자 폴더 아래에 있다). ASCII 인 후보를 찾아 쓴다.
        _ASCII_TMP = None
        # 경로를 박아 쓰지 않는다. PUBLIC 은 Windows 가 설정하는 표준 변수다.
        for cand in (tempfile.gettempdir(), os.environ.get("PUBLIC"),
                     "/tmp", os.getcwd()):
            if not cand:
                continue
            try:
                cand.encode("ascii")
            except (UnicodeEncodeError, AttributeError):
                continue
            if os.path.isdir(cand) and os.access(cand, os.W_OK):
                _ASCII_TMP = tempfile.mkdtemp(prefix="itda_w_", dir=cand)
                break
    if _ASCII_TMP is None:
        print("[보조] ASCII 임시 경로를 못 찾았다 — 보조 경로를 건너뛴다")
        return None
    dst = os.path.join(_ASCII_TMP, os.path.basename(path.rstrip("\\/")))
    if not os.path.exists(dst):
        try:
            if os.path.isdir(path):
                shutil.copytree(path, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(path, dst)
        except Exception as e:
            print("[보조] ASCII 경로 복사 실패: %s" % e)
            return None
    return dst


# ---- 주력 엔진: RapidOCR (PP-OCRv6 det/rec, ONNX) ----
# 홀드아웃 198장 실측(EDA 63~65): 보조 단독 66.8% / 주력 단독 79.8% / 합쳐서 82.3%
# 모델 세대 차이가 크다. 보조 경로는 v3 det + v4 rec, 이쪽은 v6 det/rec 다.
# onnxruntime 기반이라 torch 를 싣지 않는다 — 날짜 검출기와 같은 런타임이다.
t0 = time.time()
RAPID, _rapid_src = build_rapid_ocr(os.path.join(LOCAL, "rapidocr"))
print("[주력] RapidOCR PP-OCRv6 det/rec small : %s (%.1fs)" % (_rapid_src, time.time() - t0))

# ---- 재시도 단계: 더 큰 검출기(PP-OCRv6 det medium) ----
# 주력이 아무것도 못 읽은 이미지에만 한 번 더 돈다.
#
# 근거 (EDA 74~76 실측):
#   실패 이미지 252장에 설정을 바꿔 가며 돌려 보니, 검출기를 medium 으로 키운 것이
#   69장(27.4%)을 회수했다. 인식기 쪽 교체는 전부 합쳐도 그보다 작았다.
#   홀드아웃2 198장에서 83.5% → 87.2% (+3.7%p, 개선 9 / 악화 **0**, p=0.004).
#
# 왜 전면 교체가 아니라 재시도인가: medium 을 처음부터 쓰면 라벨 179장에서
# 맞히던 2장을 망가뜨렸다. 실패분에만 걸면 그 회귀가 구조적으로 0 이 된다.
t0 = time.time()
RAPID_RETRY, _retry_src = build_rapid_retry(os.path.join(LOCAL, "rapidocr"))
print("[재시도] RapidOCR det medium : %s (%.1fs)" % (_retry_src, time.time() - t0))

# ---- 재시도 단계: 인식기를 바꾼다 (한국어 → 영문) ----
# 주력 인식기 PP-OCRv6_rec_small 은 중국어 모델이라 **한글을 제대로 읽지 못한다**.
# 날짜 자체는 숫자라 읽지만, 한글이 날짜에 바로 붙어 있으면 그 글자를 엉뚱한
# 한자로 읽으면서 날짜까지 함께 깨뜨린다 (EDA 101~102).
#   `나 2026.03.11 까지 홍병현`  → NONE-03-11 에 머물렀다
#   `유통2023.01.05까지`         → 아무것도 못 냈다
# 이 칸들도 앞 칸이 실패한 이미지에만 돈다. 580장에서 개선 6 / 악화 1.
# (REC_KOREAN·REC_EN 은 combined.py 가 정의한 **파일명** 상수다. 엔진 변수는
#  이름이 겹치지 않도록 _ENG 를 붙인다.)
t0 = time.time()
REC_KO_ENG, _ko_src = build_rapid_rec(os.path.join(LOCAL, "rapidocr"), REC_KOREAN)
REC_EN_ENG, _en_src = build_rapid_rec(os.path.join(LOCAL, "rapidocr"), REC_EN)
print("[재시도] rec 한국어 : %s / rec 영문 : %s (%.1fs)"
      % (_ko_src, _en_src, time.time() - t0))


# ---- 보조 엔진: PP-OCRv3 det + 한국어 PP-OCRv4 rec ----
# 주력이 아무것도 읽지 못한 이미지에만 돈다 (전체의 약 5%).
# 세 모델 디렉터리가 모두 있어야 만든다. 하나라도 없으면 PaddleOCR 이 자동
# 다운로드를 시도하므로 **아예 만들지 않는다**.
OCR = None
# _ascii_safe 는 경로가 비ASCII 일 때만 ASCII 임시 폴더로 옮긴다 (위 주석 참고).
_det = _ascii_safe(_need("det", "ml", "Multilingual_PP-OCRv3_det_infer"))
_rec = _ascii_safe(_need("rec", "korean", "korean_PP-OCRv4_rec_infer"))
_cls = _ascii_safe(_need("cls", "ch_ppocr_mobile_v2.0_cls_infer"))
if _det and _rec and _cls:
    try:
        from paddleocr import PaddleOCR
        t0 = time.time()
        # det_limit_side_len 은 사다리에서 쓸 최대값(640). det_limit_type="max" 라
        # 그보다 작은 입력은 그대로 쓰이므로 실제 해상도는 pipe.long_side 로 제어한다.
        OCR = PaddleOCR(use_angle_cls=False, lang="korean", show_log=False,
                        use_gpu=False,            # 채점 환경 GPU 미제공
                        det_limit_side_len=640, det_limit_type="max",
                        det_model_dir=_det, rec_model_dir=_rec, cls_model_dir=_cls)
        print("[보조] PP-OCRv3 det + PP-OCRv4 rec : bundled (%.1fs)" % (time.time() - t0))
    except Exception as e:
        OCR = None
        print("[보조] 로딩 실패 (%s) — 주력 경로만으로 진행" % e)
else:
    print("[보조] 가중치 없음 — 주력 경로만으로 진행")


# ---- 날짜 영역 전용 검출기 (YOLOv8n, ONNX, 직접 학습) ----
# PP-OCRv3 det 가 날짜 영역을 아예 텍스트로 보지 않는 것이 보조 경로의 최대 손실원이었다
# (EDA 44: 정답 박스를 det 가 찾은 비율 — 성공 59% / 실패 13% / 못읽음 5%).
#
# torch 가 아니라 ONNX Runtime 을 쓰는 이유 (EDA 52~53 실측): torch 를 임포트하기만 해도
# 장당 2.76 → 5.88 초가 됐다. paddle 과 각자 OpenMP 런타임을 싣고 스레드를 다투기 때문이다.
DETECTOR = None
_yolo = _need("yolo", "date_v1.onnx")
if _yolo:
    try:
        DETECTOR = OnnxDateDetector(_yolo)
        print("[보조] 날짜 검출기 : bundled")
    except Exception as e:
        print("[보조] 날짜 검출기 로딩 실패 (%s) — det 단독으로 진행" % e)
else:
    print("[보조] 날짜 검출기 가중치 없음 — det 단독으로 진행")

if RAPID is None and OCR is None:
    print("\\n[경고] 두 엔진 모두 로딩되지 않았다. `bash download_weights.sh` 를 먼저 실행할 것.")
    print("       제출 형식은 유지되지만 결과는 전부 NONE 이 된다.")
elif RAPID_RETRY is None:
    print("\\n[참고] 재시도용 det medium 이 없다. 동작은 하지만 정확도가 낮아진다")
    print("       (홀드아웃 기준 87.2% → 83.5%). download_weights.sh 로 받을 수 있다.")
'''

CELL_RUN = '''# 추론
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
files = sorted(f for f in os.listdir(INPUT_DIR)
               if f.lower().endswith(EXTS) and not f.startswith("."))
print(len(files), "images")

# 운영진 확정(2026-09-12): 평가셋 500장, 제한 40분(2400초) 일괄. 장당 4.8초.
#
# 단일 프로세스 실측(EDA 66, 홀드아웃2 표본):
#   보조 경로 단독      2.19 초/장   500장 1,096초 (제한의 46%)
#   RapidOCR 단독       0.37 초/장   500장   184초 (제한의  8%)
#   합친 것             0.53 초/장   500장   267초 (제한의 11%)
# 비싼 보조 경로가 RapidOCR 이 실패한 소수 이미지에만 도는 구조라, 정확도가
# 66.8% → 82.3% 로 오르면서 속도는 오히려 4배 빨라졌다.
#
# ⚠️ 과거 문서의 '0.99초/장, 제한의 21%' 는 4프로세스 병렬 처리량을 잘못 쓴 값이었다.
#    노트북은 순차 루프다. 평가셋이 예상보다 클 경우를 대비해 예산 제어기는 그대로 둔다.
BUDGET_SEC = float(os.environ.get("ITDA_BUDGET_SEC", "2000"))
budget = TimeBudget(len(files), budget_sec=BUDGET_SEC)

# 보조 경로 — RapidOCR 이 아무것도 못 냈을 때만 돈다.
# 보조 가중치가 없으면 만들지 않는다 (None 이면 주력 경로만으로 동작한다).
fallback = None
if OCR is not None:
    fallback = DatePipeline(OCR, long_side=640, rotations=(0, 90),
                            adaptive=True, keep=None)
    if DETECTOR is not None:
        fallback.yolo = DETECTOR
        fallback.yolo_conf = 0.25
        fallback.yolo_mode = "always"
    budget.apply(fallback)

# 주력 + 재시도 + 보조를 합친 파이프라인 (설계 근거는 combined.py 주석)
# ---- 마지막 칸: 도트 인쇄 판독기 ----
# 도트 매트릭스 날짜는 일반 인식기가 글꼴을 모른다. 합성 데이터로 학습한
# CTC 모델이 줄 단위로 통째로 읽는다. 영역은 YOLO 가 잡는다.
# 두 가중치가 다 있어야 돈다. 없으면 이 칸을 건너뛴다.
DOT, _dot_src = build_crnn(LOCAL)
print("[재시도] 도트 판독기 : %s" % _dot_src)

pipe = CombinedPipeline(RAPID, TEAM2_RULES, fallback, rapid_retry=RAPID_RETRY,
                        rec_korean=REC_KO_ENG, rec_en=REC_EN_ENG,
                        dot=DOT, dot_boxes=DETECTOR)

rows, t0 = [], time.time()
for i, fname in enumerate(files, 1):
    # image_id 는 확장자를 뺀 파일명 문자열 그대로.
    # '000001' 을 int 로 바꾸면 '1' 이 되어 전량 불일치한다.
    image_id = os.path.splitext(fname)[0]
    try:
        fields = pipe.predict(os.path.join(INPUT_DIR, fname))
    except Exception:
        fields = {"year": "NONE", "month": "NONE", "day": "NONE"}
    rows.append(combined_to_row(image_id, fields))

    budget.tick()
    if fallback is not None and budget.apply(fallback):
        print("  [budget] lv%d side=%s keep=%s sp=%.2f at %d/%d"
              % (budget.level, fallback.long_side, fallback.keep,
                 fallback.second_pass_frac, i, len(files)), flush=True)
    if i % 200 == 0:
        el = time.time() - t0
        print("  %d/%d  %.0fs  (%.3fs/img)  경로별 %s"
              % (i, len(files), el, el / i, pipe.stats), flush=True)

print("경로별 처리:", pipe.stats,
      "  (rapid=1차, adaptive=재시도, fallback=보조 경로, none=실패)")

el = time.time() - t0
print("done: %d rows in %.0fs (%.3fs/img)" % (len(rows), el, el / max(len(rows), 1)))
'''

CELL_SAVE = '''# 제출 파일 저장
df = pd.DataFrame(rows, columns=["image_id", "year", "month", "day", "final_date"])

# 규정 §8 방어: month/day 는 2자리 문자열, 미인식은 문자열 "NONE"
for c in ("year", "month", "day", "final_date"):
    df[c] = df[c].fillna("NONE").astype(str)

assert not df["image_id"].duplicated().any(), "image_id 중복"
assert df["month"].map(lambda v: v == "NONE" or len(v) == 2).all(), "month 자릿수"
assert df["day"].map(lambda v: v == "NONE" or len(v) == 2).all(), "day 자릿수"
assert list(df.columns) == ["image_id", "year", "month", "day", "final_date"]

outdir = os.path.dirname(os.path.abspath(OUTPUT_PATH))
if outdir:
    os.makedirs(outdir, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)   # index=False 필수 - 켜면 컬럼이 밀려 0점

print("date extracted:", round(float(df["final_date"].ne("NONE").mean()), 4))
print(df.head())
print("saved:", OUTPUT_PATH, os.path.getsize(OUTPUT_PATH), "bytes")
'''


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text}


def main():
    src = open(SRC, encoding="utf-8").read()
    rules = open(SRC_RULES, encoding="utf-8").read()
    comb = open(SRC_COMB, encoding="utf-8").read()
    dseg = open(SRC_DSEG, encoding="utf-8").read()
    dcrnn = open(SRC_DCRNN, encoding="utf-8").read()
    # 노트북에는 모듈이 없다. dotcrnn 이 `import dotseg` 로 쓰는 이름들을
    # 같은 네임스페이스에서 찾도록, import 줄을 지우고 얇은 대역을 세운다.
    dseg = dseg.replace("import cv2\nimport numpy as np\n", "")
    dcrnn = dcrnn.replace("import dotseg\n", "").replace(
        "import numpy as np\n", "")
    dcrnn = dcrnn.replace("dotseg.", "_dotseg.")
    dseg_glue = (
        "class _DotSeg:\n"
        "    _binarize = staticmethod(_binarize)\n"
        "    _stroke_width = staticmethod(_stroke_width)\n"
        "    _rows = staticmethod(_rows)\n"
        "    segment = staticmethod(segment)\n"
        "\n_dotseg = _DotSeg()\n")
    # 노트북 안에서는 모듈이 아니라 한 네임스페이스에 다 올라오므로,
    # combined.py 가 기대하는 이름들을 노트북 문맥에 맞춰 연결한다.
    # (combined.py 의 to_row 가 pipeline.py 의 동명 함수를 가린다 — 의도한 것이다.
    #  추론 루프는 combined_to_row 를 쓴다.)
    glue = (
        "# combined.py 는 규칙을 모듈로 받도록 되어 있다. 노트북에서는 모든 함수가\n"
        "# 한 네임스페이스에 있으므로, 같은 인터페이스를 갖는 얇은 객체로 감싼다.\n"
        "class _Team2Rules:\n"
        "    choose_expiration_date = staticmethod(choose_expiration_date)\n"
        "    extract_partial_date = staticmethod(extract_partial_date)\n"
        "\n"
        "TEAM2_RULES = _Team2Rules()\n"
        "combined_to_row = to_row          # combined.py 쪽 to_row (제출 행 생성)\n"
    )
    cells = [
        md("# 소비기한 추출 — 추론 노트북\n\n"
           "제3회 ITDA 연합학술제. `Run All` 만으로 `submission.csv` 를 생성한다.\n\n"
           "**구조**: RapidOCR(PP-OCRv6)을 주력으로 쓰고, 그것이 아무것도 읽지 못한\n"
           "이미지에만 PP-OCRv3/v4 + YOLO 날짜검출기 경로를 보조로 돌린다.\n"
           "홀드아웃 198장 실측 — 보조 단독 66.8% / 주력 단독 79.8% / **합쳐서 82.3%**.\n\n"
           "설계 근거는 저장소의 `METHOD.md` 와 `eda/out/` 에 실측과 함께 기록되어 있다."),
        code(CELL_CONFIG),
        code(CELL_SETUP),
        md("## 보조 경로 — PP-OCR + 날짜 검출기\n\n"
           "아래 셀은 `src/pipeline.py` 에서 자동 생성된다 (`python build_notebook.py`). "
           "노트북 단독 실행을 보장하기 위해 코드를 내장한다."),
        code(src),
        md("## 주력 규칙 — 날짜 판별\n\n"
           "`src/team2_rules.py` (팀원2 박안젤라 원문). OCR 조각을 이어 붙여 후보를 만들고, "
           "소비기한/유통기한 가점·제조일자 감점으로 고른다."),
        code(rules),
        md("## 도트 인쇄 판독기\n\n"
           "`src/dotseg.py` + `src/dotcrnn.py`. 도트 매트릭스로 찍힌 날짜는 일반 "
           "인식기가 글꼴을 학습한 적이 없어 `26.08.13` 을 `2:88:13` 으로 읽는다. "
           "그래서 도트 날짜를 합성해 CTC 시퀀스 모델을 따로 학습했다. "
           "앞 칸들이 전부 실패한 이미지에만 돈다."),
        code(dseg + "\n\n" + dseg_glue + "\n\n" + dcrnn),
        md("## 두 경로를 잇는 층\n\n"
           "`src/combined.py`. RapidOCR 이 실패한 이미지에만 보조 경로를 부른다."),
        code(comb + "\n\n" + glue),
        md("## 모델"),
        code(CELL_MODEL),
        md("## 추론"),
        code(CELL_RUN),
        md("## 제출 파일 저장"),
        code(CELL_SAVE),
    ]
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print("wrote %s (%d cells)" % (OUT, len(cells)))


if __name__ == "__main__":
    main()

