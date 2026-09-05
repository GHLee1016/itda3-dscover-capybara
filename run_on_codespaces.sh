#!/usr/bin/env bash
# ITDA 3rd 학술제 - GitHub Codespaces(4-core 옵션)에서 predict.ipynb 처리량/정확도 검증용 스크립트
#
# 사용법:
#   1) github.com 에서 아무 저장소(빈 저장소라도 무방)에 들어가서
#      Code ▸ Codespaces ▸ "..." ▸ "New with options..." 를 선택
#   2) Machine type 을 "4-core · 16GB RAM" (또는 그 이상)으로 선택해서 Codespace 생성
#      (개인 계정 무료 사용량: 월 120 core-hour → 4-core 기준 약 30시간, 이번 검증엔 충분)
#   3) Codespace가 열리면, 이 스크립트와 itda3-dscover-capybara.zip 을
#      VS Code 파일 탐색기에 드래그 앤 드롭으로 올려두기 (같은 폴더, 보통 /workspaces/<repo명>)
#   4) 터미널에서:  bash run_on_codespaces.sh
#
# 실제 채점 서버 사양(Standard 4-Core vCPU, 오프라인)과 가장 비슷한 조건에서
# predict.ipynb 의 실제 처리 속도/NONE 비율을 확인하기 위한 스크립트입니다.

set -e

echo "=== 0. 환경 확인 ==="
python3 --version
echo "CPU 코어 수: $(nproc)"

echo ""
echo "=== 1. 저장소 압축 해제 ==="
if [ ! -f itda3-dscover-capybara.zip ]; then
  echo "오류: itda3-dscover-capybara.zip 파일이 현재 폴더에 없습니다."
  echo "VS Code 파일 탐색기에 zip 파일을 먼저 드래그 앤 드롭으로 올려주세요."
  exit 1
fi
unzip -q -o itda3-dscover-capybara.zip
cd itda3-dscover-capybara

echo ""
echo "=== 2. 파이썬 패키지 설치 (requirements.txt, 버전 고정 그대로) ==="
pip install -q -r requirements.txt

echo ""
echo "=== 3. OCR 가중치 다운로드 ==="
bash download_weights.sh

echo ""
echo "=== 4. 대회 이미지 데이터셋 다운로드 (3,352장) ==="
curl -L --fail -o images.zip "https://github.com/b9511242000-blip/itda-ocr-template/releases/download/v.1.0.0/images.zip"
mkdir -p val_images_raw val_images
unzip -q images.zip -d val_images_raw
find val_images_raw -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) -exec mv {} val_images/ \;
rm -rf val_images_raw images.zip
echo "이미지 개수: $(ls val_images | wc -l)   # 3352 가 나와야 정상"

echo ""
echo "=== 5. 채점 재현 명령 실행 (운영진 표준 nbconvert 명령, 2400초 타임아웃) ==="
export ITDA_INPUT_DIR=./val_images
export ITDA_OUTPUT_PATH=./submission.csv
time jupyter nbconvert --to notebook --execute predict.ipynb \
    --ExecutePreprocessor.timeout=2400 \
    --output executed.ipynb

echo ""
echo "=== 6. 결과 요약 ==="
python3 - <<'PYEOF'
import pandas as pd
df = pd.read_csv("submission.csv")
print(len(df), "rows")
print((df["final_date"] == "NONE").mean(), "= NONE 비율")
ok = df[df["final_date"] != "NONE"]
print("성공 개수:", len(ok))
if len(ok):
    print("image_id 범위:", ok["image_id"].min(), "~", ok["image_id"].max())
PYEOF

echo ""
echo "=== 7. 처리 속도 로그 (N_WORKERS / 진행상황 / 경과시간) ==="
python3 - <<'PYEOF'
import json
nb = json.load(open("executed.ipynb"))
for cell in nb["cells"]:
    for out in cell.get("outputs", []):
        text = "".join(out.get("text", []))
        if any(k in text for k in ("N_WORKERS", "processed", "elapsed", "input images", "done.")):
            print(text.strip())
PYEOF

echo ""
echo "완료. submission.csv / executed.ipynb 가 itda3-dscover-capybara/ 폴더에 저장되었습니다."
