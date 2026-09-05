#!/usr/bin/env bash
# ITDA 3rd 학술제 - EasyOCR 가중치 사전 다운로드 스크립트
#
# 채점 서버는 오프라인이므로, 이 스크립트는 predict.ipynb 실행 전에 운영진이
# (인터넷이 되는 상태에서) 1회 실행합니다. predict.ipynb 는 이 스크립트가 받아둔
# ./weights 폴더의 로컬 가중치만 사용하며 (download_enabled=False), 실행 중 어떤
# 네트워크 다운로드도 시도하지 않습니다.
#
# 사용법:
#   bash download_weights.sh

set -euo pipefail

WEIGHTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/weights"
mkdir -p "$WEIGHTS_DIR"

DET_URL="https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip"
REC_URL="https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/korean_g2.zip"

download_and_extract () {
  local url="$1"
  local out_zip="$2"
  echo "Downloading: $url"
  curl -L --fail -o "$out_zip" "$url"
  echo "Extracting: $out_zip"
  python3 - "$out_zip" "$WEIGHTS_DIR" <<'PYEOF'
import sys, zipfile
zip_path, out_dir = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(zip_path) as z:
    z.extractall(out_dir)
PYEOF
  rm -f "$out_zip"
}

if [ ! -f "$WEIGHTS_DIR/craft_mlt_25k.pth" ]; then
  download_and_extract "$DET_URL" "$WEIGHTS_DIR/craft_mlt_25k.zip"
else
  echo "craft_mlt_25k.pth already present, skipping."
fi

if [ ! -f "$WEIGHTS_DIR/korean_g2.pth" ]; then
  download_and_extract "$REC_URL" "$WEIGHTS_DIR/korean_g2.zip"
else
  echo "korean_g2.pth already present, skipping."
fi

echo "Done. Weights in: $WEIGHTS_DIR"
ls -la "$WEIGHTS_DIR"
