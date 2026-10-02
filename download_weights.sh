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
# 파이썬 실행기를 찾는다.
#
# ⚠️ 예전에는 `python` 을 그냥 불렀다. Git Bash 에서는 PATH 에 `python` 이
#    없는 경우가 흔하고, 그러면 아래 주력 엔진 블록이
#        download_weights.sh: line 39: python: command not found
#    로 조용히 넘어가 **RapidOCR 가중치 110MB 를 받지 못한 채** 끝났다.
#    curl 로 받는 보조 엔진만 채워져서 겉보기에는 성공처럼 보였다.
#    새 가상환경 검증에서 발견했다(logs/fresh_run_*.log).
#
# 우선순위: 활성화된 venv → PYTHON 환경변수 → python3 → python → py -3
# --------------------------------------------------------------------------
PY=""
for cand in \
    "${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}" \
    "${VIRTUAL_ENV:+$VIRTUAL_ENV/Scripts/python.exe}" \
    "${PYTHON:-}" \
    ".venv/bin/python" ".venv/Scripts/python.exe" \
    python3 python; do
  [ -z "$cand" ] && continue
  if command -v "$cand" >/dev/null 2>&1; then PY="$cand"; break; fi
done
if [ -z "$PY" ] && command -v py >/dev/null 2>&1; then PY="py -3"; fi

# --------------------------------------------------------------------------
# WSL bash + Windows python 조합 보정.
#
# ⚠️ 이것 때문에 주력 엔진 가중치가 조용히 복사되지 않았다.
#    WSL bash 에서 $DIR 은 `/mnt/c/...` 인데 `.venv/Scripts/python.exe` 는
#    **Windows** 실행 파일이다. 그 python 에게 `/mnt/c/...` 를 목적지로 주면
#    쓸 수 없어 복사가 실패한다. 로컬에서는 파일이 이미 있어 `skip (exists)` 로
#    빠져 이 버그가 보이지 않았다 — 새 가상환경 검증에서 드러났다
#    (logs/fresh_run_20261002_161451.log: weights/rapidocr 가 빈 채로 끝났다).
#
# 리눅스·macOS 에서는 아무 일도 하지 않는다. 채점 환경이 그쪽이면 무관하다.
# --------------------------------------------------------------------------
winpath() {   # $1 = POSIX 경로 → python 이 이해할 수 있는 경로
  case "$PY" in
    *.exe|*.exe\ *|py\ -3)
      if command -v wslpath >/dev/null 2>&1; then
        wslpath -w "$1" 2>/dev/null || printf '%s' "$1"
      elif command -v cygpath >/dev/null 2>&1; then
        cygpath -w "$1" 2>/dev/null || printf '%s' "$1"
      else
        printf '%s' "$1"
      fi ;;
    *) printf '%s' "$1" ;;
  esac
}

if [ -z "$PY" ]; then
  say "!! 파이썬을 찾지 못했다. 주력 엔진(RapidOCR) 가중치를 받을 수 없다."
  say "   가상환경을 활성화한 뒤 다시 실행하거나, PYTHON 으로 경로를 지정할 것:"
  say "     PYTHON=/c/Python310/python.exe bash download_weights.sh"
  say "   이 가중치가 없으면 주력 경로가 비활성화되고 정확도가 크게 내려간다."
  FAIL=1
else
  say "python: $PY"
fi

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
  if [ -z "$PY" ]; then
    say "!! 파이썬이 없어 주력 엔진 가중치를 건너뛴다"
  else
  $PY - "$(winpath "$RAPID_DST")" <<'PYEOF'
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
# ⚠️ 버전을 명시해야 한다. 기본값이 PP-OCRv6 인데 그쪽에는 한국어 rec 가 없어
#    `Unsupported rec.lang_type='korean' for PP-OCRv6 small model.` 로 실패한다.
#    우리가 쓰는 파일은 korean_PP-OCRv5_rec_mobile.onnx / en_PP-OCRv5_rec_mobile.onnx
#    즉 **v5 mobile** 이다. 개발 환경에는 예전에 받아 둔 파일이 있어 이 실패가
#    가려져 있었고, 새 가상환경 검증에서 5칸이 빠지는 것으로 드러났다.
for _lang in (LR.KOREAN, LR.EN):
    _ok = False
    _last = None
    for _p in ({"Rec.lang_type": _lang, "Rec.ocr_version": OV.PPOCRV5,
                "Rec.model_type": MT.MOBILE},
               {"Rec.lang_type": _lang, "Rec.ocr_version": OV.PPOCRV5},
               {"Rec.lang_type": _lang}):
        try:
            RapidOCR(params=_p)
            _ok = True
            break
        except Exception as e:
            _last = e
    if not _ok:
        print("!! rec %s 내려받기 실패(계속 진행): %s" % (_lang, _last))

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
  fi    # [ -z "$PY" ] 분기 닫기
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
