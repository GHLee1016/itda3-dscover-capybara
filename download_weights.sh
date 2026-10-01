#!/usr/bin/env bash
# 모델 가중치 준비 — 채점 실행 **전에** 인터넷이 되는 상태에서 1회 실행한다.
#
#   bash download_weights.sh
#
# predict.ipynb 는 여기서 받아 놓은 weights/ 만 읽는다. 노트북 안에는 다운로드
# 코드가 전혀 없다 — 채점 서버는 오프라인이라 실행 중 내려받으려 하면 그 자리에서
# 예외가 나 정량 0점이 되기 때문이다.
#
# 받는 것 (합계 약 71MB)
#   weights/rapidocr/*.onnx   주력 det/rec small + cls, 재시도 det medium/rec 한국어/rec 영문  110MB  [필수]
#   weights/det|rec|cls/...   보조 엔진 PP-OCRv3 det + v4 rec  30MB  [권장]
#   weights/yolo/date_v1.onnx 직접 학습한 날짜 검출기          11MB  [권장]
#   weights/dot/dotcrnn_v1.onnx 직접 학습한 도트 인쇄 판독기    3MB  [권장]
#
# 주력만 있어도 동작한다(부분점수 79.8%). 셋 다 있으면 82.3%.

set -uo pipefail   # -e 는 쓰지 않는다: 하나가 실패해도 나머지는 받아야 한다

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/weights"
mkdir -p "$DIR"
FAIL=0

say() { printf '%s\n' "$*"; }

# --------------------------------------------------------------------------
# 1) 주력 엔진 — RapidOCR (PP-OCRv6 det/rec + cls)  [필수]
#
# rapidocr 패키지가 최초 실행 시 자신의 site-packages 안으로 모델을 받는다.
# 그 파일을 weights/rapidocr/ 로 복사해 두면 노트북이 오프라인에서도 읽을 수 있다.
# --------------------------------------------------------------------------
RAPID_DST="$DIR/rapidocr"
if [ -f "$RAPID_DST/PP-OCRv6_det_small.onnx" ] && \
   [ -f "$RAPID_DST/PP-OCRv6_rec_small.onnx" ] && \
   [ -f "$RAPID_DST/PP-OCRv6_det_medium.onnx" ]; then
  say "skip (exists): $RAPID_DST"
else
  mkdir -p "$RAPID_DST"
  python - "$RAPID_DST" <<'PYEOF'
import os, sys, glob, shutil
dst = sys.argv[1]
try:
    import rapidocr
    from rapidocr import RapidOCR
    from rapidocr.utils.typings import (ModelType as MT, LangDet as LD,
                                        LangRec as LR, OCRVersion as OV)
except ImportError:
    print("!! rapidocr 가 없다. pip install -r requirements.txt 를 먼저 실행할 것.")
    raise SystemExit(1)

# 인스턴스를 만드는 행위가 모델 다운로드를 유발한다.
#   1) 기본(det/rec small + cls) — 주력 경로
#   2) det medium — 재시도 3·4칸. 없으면 미검출이 늘어 정확도가 내려간다
#   3) rec 한국어·영문 — 재시도 5·6칸.
#      주력 인식기 rec_small 은 중국어 모델이라 **한글을 제대로 읽지 못한다**.
#      한글이 날짜에 붙어 있으면(`나 2026.03.11 까지 홍병현`) 그 글자를 엉뚱한
#      한자로 읽으면서 날짜까지 깨뜨린다. 없으면 홀드아웃4 기준 93.0% → 91.9%.
try:
    RapidOCR()
except Exception as e:
    print("!! RapidOCR 기본 모델 초기화 실패:", e)
    raise SystemExit(1)
try:
    RapidOCR(params={"Det.lang_type": LD.CH, "Det.model_type": MT.MEDIUM,
                     "Det.ocr_version": OV.PPOCRV6})
except Exception as e:
    print("!! det medium 내려받기 실패(계속 진행):", e)
for _lang in (LR.KOREAN, LR.EN):
    try:
        RapidOCR(params={"Rec.lang_type": _lang})
    except Exception as e:
        print("!! rec %s 내려받기 실패(계속 진행): %s" % (_lang, e))

src = os.path.join(os.path.dirname(rapidocr.__file__), "models")
KEEP = ("PP-OCRv6_det_small.onnx", "PP-OCRv6_rec_small.onnx",
        "ch_ppocr_mobile_v2.0_cls_mobile.onnx", "PP-OCRv6_det_medium.onnx",
        "korean_PP-OCRv5_rec_mobile.onnx", "en_PP-OCRv5_rec_mobile.onnx")
n = 0
for p in glob.glob(os.path.join(src, "*.onnx")):
    if os.path.basename(p) in KEEP:      # 실험용으로 받아둔 다른 모델은 제외
        shutil.copy2(p, dst)
        n += 1
print("copied %d onnx -> %s" % (n, dst))
raise SystemExit(0 if n >= 2 else 1)
PYEOF
  [ $? -ne 0 ] && FAIL=1
fi

# --------------------------------------------------------------------------
# 2) 보조 엔진 — PaddleOCR PP-OCRv3 det + 한국어 PP-OCRv4 rec  [권장]
#    주력이 아무것도 읽지 못한 이미지(약 5%)에만 돈다.
# --------------------------------------------------------------------------
fetch_tar() {  # $1 = URL, $2 = 풀어 넣을 디렉터리
  local url="$1" dest="$2" tmp
  if [ -f "$dest/inference.pdmodel" ]; then
    say "skip (exists): $dest"
    return 0
  fi
  mkdir -p "$dest"
  tmp="$(mktemp)"
  say "downloading $url"
  if curl -fL --retry 3 "$url" -o "$tmp" && tar -xf "$tmp" -C "$dest" --strip-components=1; then
    rm -f "$tmp"
    return 0
  fi
  rm -f "$tmp"
  say "!! 실패: $url"
  return 1
}

BASE="https://paddleocr.bj.bcebos.com"
fetch_tar "$BASE/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_infer.tar" \
          "$DIR/det/ml/Multilingual_PP-OCRv3_det_infer" || FAIL=1
fetch_tar "$BASE/PP-OCRv4/multilingual/korean_PP-OCRv4_rec_infer.tar" \
          "$DIR/rec/korean/korean_PP-OCRv4_rec_infer" || FAIL=1
# 방향 분류기는 use_angle_cls=False 라 쓰지 않지만, PaddleOCR 생성 시 경로를 요구한다
fetch_tar "$BASE/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar" \
          "$DIR/cls/ch_ppocr_mobile_v2.0_cls_infer" || FAIL=1

# --------------------------------------------------------------------------
# 3) 날짜 영역 검출기 — 직접 학습한 YOLOv8n (ONNX)  [권장]
#    규정상 가중치는 Git 에 직접 커밋할 수 없어 Release Assets 로 배포한다.
# --------------------------------------------------------------------------
YOLO_URL="${ITDA_YOLO_URL:-https://github.com/GHLee1016/itda3-dscover-capybara/releases/download/v1.0/date_v1.onnx}"
YOLO_DST="$DIR/yolo/date_v1.onnx"

if [ -f "$YOLO_DST" ]; then
  say "skip (exists): $YOLO_DST"
elif printf '%s' "$YOLO_URL" | grep -q '<ORG>'; then
  say "!! 날짜 검출기 URL 이 설정되지 않았다 (download_weights.sh 의 YOLO_URL)."
  say "   없어도 추론은 동작한다 — 보조 경로가 det 단독으로 내려갈 뿐이다."
else
  mkdir -p "$(dirname "$YOLO_DST")"
  say "downloading $YOLO_URL"
  curl -fL --retry 3 "$YOLO_URL" -o "$YOLO_DST" || { say "!! 실패: $YOLO_URL"; rm -f "$YOLO_DST"; }
fi

# --------------------------------------------------------------------------
# 4) 도트 인쇄 판독기 — 직접 학습한 CTC 시퀀스 모델 (ONNX)  [권장]
#    도트 매트릭스 날짜는 일반 인식기가 글꼴을 학습한 적이 없어
#    `26.08.13` 을 `2:88:13` 으로 읽는다. 도트 날짜를 합성해 따로 학습했다.
#    없으면 그 칸만 건너뛴다(미검출이 조금 는다).
# --------------------------------------------------------------------------
DOT_URL="${ITDA_DOT_URL:-https://github.com/GHLee1016/itda3-dscover-capybara/releases/download/v1.0/dotcrnn_v1.onnx}"
DOT_DST="$DIR/dot/dotcrnn_v1.onnx"

if [ -f "$DOT_DST" ]; then
  say "skip (exists): $DOT_DST"
elif printf '%s' "$DOT_URL" | grep -q '<ORG>'; then
  say "!! 도트 판독기 URL 이 설정되지 않았다 (download_weights.sh 의 DOT_URL)."
  say "   없어도 추론은 동작한다 — 도트 칸만 건너뛴다."
else
  mkdir -p "$(dirname "$DOT_DST")"
  say "downloading $DOT_URL"
  curl -fL --retry 3 "$DOT_URL" -o "$DOT_DST" || { say "!! 실패: $DOT_URL"; rm -f "$DOT_DST"; }
  # ONNX 가 외부 데이터 파일을 함께 쓰는 경우 그것도 받는다
  if [ -f "$DOT_DST" ] && [ ! -f "${DOT_DST}.data" ]; then
    curl -fsL --retry 2 "${DOT_URL}.data" -o "${DOT_DST}.data" || true
    [ -s "${DOT_DST}.data" ] || rm -f "${DOT_DST}.data"
  fi
fi

# --------------------------------------------------------------------------
say ""
say "=== weights/ 점검"
python - "$DIR" <<'PYEOF'
import os, sys
d = sys.argv[1]
need = [
    ("[필수] 주력 det", "rapidocr/PP-OCRv6_det_small.onnx"),
    ("[필수] 주력 rec", "rapidocr/PP-OCRv6_rec_small.onnx"),
    ("[권장] 주력 cls", "rapidocr/ch_ppocr_mobile_v2.0_cls_mobile.onnx"),
    ("[권장] 재시도 det", "rapidocr/PP-OCRv6_det_medium.onnx"),
    ("[권장] 재시도 rec 한국어", "rapidocr/korean_PP-OCRv5_rec_mobile.onnx"),
    ("[권장] 재시도 rec 영문", "rapidocr/en_PP-OCRv5_rec_mobile.onnx"),
    ("[권장] 도트 판독기", "dot/dotcrnn_v1.onnx"),
    ("[권장] 보조 det", "det/ml/Multilingual_PP-OCRv3_det_infer/inference.pdmodel"),
    ("[권장] 보조 rec", "rec/korean/korean_PP-OCRv4_rec_infer/inference.pdmodel"),
    ("[권장] 보조 cls", "cls/ch_ppocr_mobile_v2.0_cls_infer/inference.pdmodel"),
    ("[권장] 날짜검출기", "yolo/date_v1.onnx"),
]
bad = 0
for label, rel in need:
    p = os.path.join(d, rel)
    e = os.path.exists(p)
    mb = (" %6.1f MB" % (os.path.getsize(p) / 1e6)) if e else "          -"
    print("  %s %-6s %s  %s" % ("OK " if e else "없음", label, mb, rel))
    if not e and label.startswith("[필수]"):
        bad += 1
print()
if bad:
    print("!! 필수 가중치가 없다. predict.ipynb 가 전부 NONE 을 출력한다.")
    raise SystemExit(1)
print("주력 엔진 준비 완료. predict.ipynb 를 실행할 수 있다.")
PYEOF
RC=$?

say ""
if [ "$RC" -ne 0 ]; then
  say "결과: 실패"
  exit 1
fi
[ "$FAIL" -ne 0 ] && say "결과: 일부 [권장] 항목을 받지 못했다 (실행에는 지장 없음)"
say "done. weights at: $DIR"
exit 0
