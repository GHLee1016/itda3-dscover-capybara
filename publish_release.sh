#!/usr/bin/env bash
# GitHub 저장소 생성 + 가중치 Release 업로드
#
# 규정 §7: .pt/.pth 등 대용량 가중치는 Git 에 직접 커밋 금지.
#           Release Assets + download_weights.sh 로 배포해야 한다.
#           직접 학습한 weights/yolo/date_v1.onnx (10.7MB) 가 여기 해당한다.
#
# 사전 준비 (한 번만)
#   1) GitHub CLI 설치:  brew install gh
#   2) 로그인:           gh auth login
#   3) git 신원 설정:    git config --global user.name  "이름"
#                        git config --global user.email "메일주소"
#
# 실행
#   bash publish_release.sh <owner/repo>          # 비공개
#   bash publish_release.sh <owner/repo> --public  # 공개
#
# 끝나면 download_weights.sh 의 YOLO_URL 이 실제 주소로 자동 갱신된다.

set -euo pipefail

# ── 인자 파싱 ─────────────────────────────────────────────────────────────────
REPO="${1:-}"
TAG="v1.0"
PUBLIC=false
for arg in "$@"; do
  [[ "$arg" == "--public" ]] && PUBLIC=true
  [[ "$arg" == "--tag="* ]] && TAG="${arg#--tag=}"
done

if [[ -z "$REPO" ]]; then
  echo "사용법: bash publish_release.sh <owner/repo> [--public] [--tag=v1.0]"
  echo "  예:   bash publish_release.sh GHLee1016/itda3-dscover-capybara --public"
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# ── 사전 확인 ─────────────────────────────────────────────────────────────────
ASSET="$ROOT/weights/yolo/date_v1.onnx"
if [[ ! -f "$ASSET" ]]; then
  echo "오류: 가중치가 없다: $ASSET"
  echo "      → eda/54_yolo_onnx.py 로 먼저 내보낼 것"
  exit 1
fi
if ! command -v gh &>/dev/null; then
  echo "오류: gh 가 없다. 'brew install gh' 후 'gh auth login' 할 것."
  exit 1
fi
if [[ -z "$(git config --global user.email 2>/dev/null)" ]]; then
  echo "오류: git 신원이 없다."
  echo "      → git config --global user.name '이름'"
  echo "      → git config --global user.email '메일주소'"
  exit 1
fi

# ── 1. 로컬 저장소 ────────────────────────────────────────────────────────────
echo "[1/5] git 저장소 준비"
if [[ ! -d ".git" ]]; then
  git init -q
  git branch -M main
  echo "      초기화 완료"
else
  echo "      기존 저장소 사용"
fi

git add -A
if git diff --cached --quiet; then
  echo "      커밋할 변경 없음"
else
  git commit -q -m "소비기한 추출 파이프라인 — RapidOCR(PP-OCRv6) 주력 + PP-OCRv3/v4+YOLOv8n(ONNX) 보조"
  echo "      커밋 완료: $(git rev-parse --short HEAD)"
fi

# ── 2. 원격 저장소 ────────────────────────────────────────────────────────────
echo "[2/5] GitHub 저장소 준비: $REPO"
VIS="--private"
$PUBLIC && VIS="--public"

if gh repo view "$REPO" &>/dev/null; then
  echo "      저장소 이미 있음"
  if ! git remote get-url origin &>/dev/null; then
    git remote add origin "https://github.com/$REPO.git"
  fi
  git push -u origin main
else
  gh repo create "$REPO" $VIS --source=. --remote=origin --push
  echo "      저장소 생성 완료"
fi

# ── 3. Release + 가중치 업로드 ────────────────────────────────────────────────
echo "[3/5] Release $TAG 에 가중치 업로드"
if gh release view "$TAG" --repo "$REPO" &>/dev/null; then
  echo "      Release $TAG 이미 있음 — 가중치만 덮어씀"
  gh release upload "$TAG" "$ASSET" --repo "$REPO" --clobber
else
  gh release create "$TAG" "$ASSET" --repo "$REPO" \
    --title "날짜 영역 검출기 가중치 $TAG" \
    --notes "YOLOv8n 파인튜닝 → ONNX (10.7 MB).
download_weights.sh 가 weights/yolo/date_v1.onnx 로 내려받는다.
없어도 추론은 동작하며 보조 경로가 det 단독으로 내려간다 (82.3% → ~80%)."
fi

# ── 4. download_weights.sh 의 YOLO_URL 갱신 ──────────────────────────────────
echo "[4/5] download_weights.sh 의 YOLO_URL 갱신"
URL="https://github.com/$REPO/releases/download/$TAG/date_v1.onnx"
DW="$ROOT/download_weights.sh"

# sed -i 는 macOS 에서 -i '' 가 필요하다
if sed -i '' \
    "s|YOLO_URL=\"\${ITDA_YOLO_URL:-[^}]*}\"|YOLO_URL=\"\${ITDA_YOLO_URL:-$URL}\"|" \
    "$DW"; then
  echo "      YOLO_URL → $URL"
  git add download_weights.sh
  if git diff --cached --quiet; then
    echo "      (변경 없음 — 이미 같은 URL)"
  else
    git commit -q -m "download_weights.sh: YOLO Release 주소 반영 ($TAG)"
    git push -q origin main
    echo "      커밋·푸시 완료: $(git rev-parse --short HEAD)"
  fi
else
  echo "      경고: sed 실패 — download_weights.sh 를 직접 확인할 것"
fi

# ── 5. 불필요한 협력자 제거 ───────────────────────────────────────────────────
echo "[5/5] 불필요한 협력자 제거 (claude)"
# HTTP 204 = 삭제 성공, 404 = 원래 없음 — 둘 다 정상
if gh api -X DELETE "repos/$REPO/collaborators/claude" 2>/dev/null; then
  echo "      'claude' 협력자 제거 완료 (또는 이미 없었음)"
else
  echo "      경고: 협력자 제거 실패 — GitHub 웹 > Settings > Collaborators 에서 직접 제거할 것"
fi

# ── 완료 ──────────────────────────────────────────────────────────────────────
echo ""
echo "완료"
echo "  저장소     : https://github.com/$REPO"
echo "  가중치 URL : $URL"
echo "  커밋 해시  : $(git rev-parse HEAD)"
echo ""
echo "제출 메일에 위 저장소 URL 과 커밋 해시를 기재할 것."
if ! $PUBLIC; then
  echo "비공개 저장소입니다. 제출 전 공개로 바꾸거나 운영진 계정(b9511242000-blip)을 협력자로 추가할 것."
fi
