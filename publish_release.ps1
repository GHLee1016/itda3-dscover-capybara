# GitHub 저장소 생성 + 가중치 Release 업로드
#
# 규정 §7: .pt/.pth 등 대용량 가중치는 Git 에 직접 커밋 금지.
#           Release Assets 나 외부 링크 + download_weights.sh 로 배포해야 한다.
#           직접 학습한 weights/yolo/date_v1.onnx (10.7MB) 가 여기 해당한다.
#
# 사전 준비 (한 번만)
#   1) GitHub CLI 설치:  winget install --id GitHub.cli
#   2) 로그인:           gh auth login          ← 브라우저가 열린다
#   3) git 신원 설정:    git config --global user.name  "이름"
#                        git config --global user.email "메일주소"
#
# 실행
#   powershell -ExecutionPolicy Bypass -File publish_release.ps1 -Repo "myid/itda-expiry"
#
# 끝나면 download_weights.sh 의 YOLO_URL 이 실제 주소로 자동 갱신된다.

param(
    [Parameter(Mandatory = $true)]
    [string]$Repo,                       # 예: "myid/itda-expiry"
    [string]$Tag = "v1.0",
    [switch]$Public                      # 기본은 비공개. 제출 전 공개로 바꿀 것
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$asset = Join-Path $root "weights\yolo\date_v1.onnx"
if (-not (Test-Path -LiteralPath $asset)) {
    throw "가중치가 없다: $asset  (eda/54_yolo_onnx.py 로 먼저 내보낼 것)"
}
if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    throw "gh 가 없다. 'winget install --id GitHub.cli' 후 'gh auth login' 할 것."
}
if (-not (git config --global user.email)) {
    throw "git 신원이 없다. git config --global user.name/user.email 을 먼저 설정할 것."
}

# --- 1. 로컬 저장소 -------------------------------------------------------
if (-not (Test-Path ".git")) {
    Write-Output "[1/4] git 저장소 초기화"
    git init -q
    git branch -M main
} else {
    Write-Output "[1/4] 기존 git 저장소 사용"
}

git add -A
if (git diff --cached --quiet) {
    Write-Output "      커밋할 변경 없음"
} else {
    git commit -q -m "소비기한 추출 파이프라인 — PP-OCR + 날짜 영역 검출기(YOLOv8n/ONNX)"
    Write-Output "      커밋 완료"
}

# --- 2. 원격 저장소 -------------------------------------------------------
Write-Output "[2/4] GitHub 저장소 준비: $Repo"
$vis = if ($Public) { "--public" } else { "--private" }
gh repo view $Repo > $null 2>&1
if ($LASTEXITCODE -ne 0) {
    gh repo create $Repo $vis --source=. --remote=origin --push
} else {
    if (-not (git remote get-url origin 2>$null)) {
        git remote add origin "https://github.com/$Repo.git"
    }
    git push -u origin main
}

# --- 3. Release + 가중치 업로드 -------------------------------------------
Write-Output "[3/4] Release $Tag 에 가중치 업로드"
gh release view $Tag --repo $Repo > $null 2>&1
if ($LASTEXITCODE -ne 0) {
    gh release create $Tag $asset --repo $Repo `
        --title "날짜 영역 검출기 가중치 $Tag" `
        --notes "YOLOv8n 파인튜닝 → ONNX (10.7MB). weights/yolo/date_v1.onnx 로 내려받는다. 없어도 추론은 동작하며 부분점수가 66.8% → 64.5% 로 낮아진다."
} else {
    gh release upload $Tag $asset --repo $Repo --clobber
}

# --- 4. download_weights.sh 의 URL 갱신 -----------------------------------
Write-Output "[4/4] download_weights.sh 의 YOLO_URL 갱신"
$url = "https://github.com/$Repo/releases/download/$Tag/date_v1.onnx"
$dw  = Join-Path $root "download_weights.sh"
$txt = [IO.File]::ReadAllText($dw)
$txt = [Text.RegularExpressions.Regex]::Replace(
    $txt, 'YOLO_URL="\$\{ITDA_YOLO_URL:-[^}]*\}"', "YOLO_URL=`"`${ITDA_YOLO_URL:-$url}`"")
[IO.File]::WriteAllText($dw, $txt, (New-Object Text.UTF8Encoding $false))

git add download_weights.sh
if (-not (git diff --cached --quiet)) {
    git commit -q -m "download_weights.sh: 가중치 Release 주소 반영"
    git push -q origin main
}

Write-Output ""
Write-Output "완료"
Write-Output "  저장소     : https://github.com/$Repo"
Write-Output "  가중치 URL : $url"
Write-Output "  커밋 해시  : $(git rev-parse HEAD)"
Write-Output ""
Write-Output "제출 메일에 위 저장소 URL 과 커밋 해시를 기재할 것."
Write-Output "비공개로 만들었다면 제출 전 공개로 바꾸거나 운영진 계정을 협력자로 추가할 것."
