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
    [string]$Repo,                       # 예: "GHLee1016/itda3-dscover-capybara"
    [string]$Tag = "v1.0",
    [switch]$Public                      # 기본은 비공개. 제출 전 공개로 바꿀 것
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

# ── 사전 확인 ─────────────────────────────────────────────────────────────────
$asset = Join-Path $root "weights\yolo\date_v1.onnx"
if (-not (Test-Path -LiteralPath $asset)) {
    throw "가중치가 없다: $asset`n  → eda/54_yolo_onnx.py 로 먼저 내보낼 것"
}
if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    throw "gh 가 없다.`n  → 'winget install --id GitHub.cli' 후 'gh auth login' 할 것."
}
if (-not (git config --global user.email 2>$null)) {
    throw "git 신원이 없다.`n  → git config --global user.name/user.email 을 먼저 설정할 것."
}

# --- 1. 로컬 저장소 ──────────────────────────────────────────────────────────
if (-not (Test-Path ".git")) {
    Write-Output "[1/5] git 저장소 초기화"
    git init -q
    git branch -M main
} else {
    Write-Output "[1/5] 기존 git 저장소 사용"
}

git add -A
# git diff --cached --quiet 는 변경 없으면 exit 0, 있으면 exit 1
$staged = (git diff --cached --name-only)
if (-not $staged) {
    Write-Output "      커밋할 변경 없음"
} else {
    git commit -q -m "소비기한 추출 파이프라인 — RapidOCR(PP-OCRv6) 주력 + PP-OCRv3/v4+YOLOv8n(ONNX) 보조"
    Write-Output "      커밋 완료: $(git rev-parse --short HEAD)"
}

# --- 2. 원격 저장소 ──────────────────────────────────────────────────────────
Write-Output "[2/5] GitHub 저장소 준비: $Repo"
$vis = if ($Public) { "--public" } else { "--private" }

$repoExists = $false
gh repo view $Repo 2>&1 | Out-Null
if ($LASTEXITCODE -eq 0) { $repoExists = $true }

if (-not $repoExists) {
    gh repo create $Repo $vis --source=. --remote=origin --push
} else {
    # 원격이 아직 없으면 추가
    $remoteUrl = git remote get-url origin 2>$null
    if (-not $remoteUrl) {
        git remote add origin "https://github.com/$Repo.git"
    }
    git push -u origin main
}

# --- 3. Release + 가중치 업로드 ──────────────────────────────────────────────
Write-Output "[3/5] Release $Tag 에 가중치 업로드"
gh release view $Tag --repo $Repo 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    gh release create $Tag $asset --repo $Repo `
        --title "날짜 영역 검출기 가중치 $Tag" `
        --notes "YOLOv8n 파인튜닝 → ONNX (10.7 MB).``ndownload_weights.sh 가 weights/yolo/date_v1.onnx 로 내려받는다.``n없어도 추론은 동작하며 보조 경로가 det 단독으로 내려간다 (82.3% → ~80%)."
} else {
    Write-Output "      Release $Tag 이미 있음 — 가중치만 덮어씀"
    gh release upload $Tag $asset --repo $Repo --clobber
}

# --- 4. download_weights.sh 의 YOLO_URL 갱신 ─────────────────────────────────
# ※ 버그 주의: PowerShell 문자열 보간으로 ${...} 를 쓰면 변수로 해석된다.
#              단순 문자열 연결(+)로 조립해야 bash 변수 구문이 그대로 유지된다.
Write-Output "[4/5] download_weights.sh 의 YOLO_URL 갱신"
$url  = "https://github.com/$Repo/releases/download/$Tag/date_v1.onnx"
$dw   = Join-Path $root "download_weights.sh"
$txt  = [IO.File]::ReadAllText($dw)

# 교체 대상: YOLO_URL="${ITDA_YOLO_URL:-<이전값>}"
# 교체 결과: YOLO_URL="${ITDA_YOLO_URL:-<실제URL>}"
$repl = 'YOLO_URL="${ITDA_YOLO_URL:-' + $url + '}"'
$pat  = 'YOLO_URL="\$\{ITDA_YOLO_URL:-[^}]*\}"'
$new  = [Text.RegularExpressions.Regex]::Replace($txt, $pat, $repl)

if ($new -eq $txt) {
    Write-Output "      경고: YOLO_URL 패턴을 찾지 못함 — download_weights.sh 를 직접 확인할 것"
} else {
    [IO.File]::WriteAllText($dw, $new, (New-Object Text.UTF8Encoding $false))
    Write-Output "      YOLO_URL → $url"

    git add download_weights.sh
    $staged2 = (git diff --cached --name-only)
    if ($staged2) {
        git commit -q -m "download_weights.sh: YOLO Release 주소 반영 ($Tag)"
        git push -q origin main
        Write-Output "      커밋·푸시 완료: $(git rev-parse --short HEAD)"
    }
}

# --- 5. 불필요한 협력자 제거 ─────────────────────────────────────────────────
Write-Output "[5/5] 불필요한 협력자 제거 (claude)"
# 204 = 삭제 성공, 404 = 원래 없음 — 둘 다 정상
gh api -X DELETE "repos/$Repo/collaborators/claude" 2>&1 | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-Output "      'claude' 협력자 제거 완료 (또는 이미 없었음)"
} else {
    Write-Output "      경고: 협력자 제거 실패 (exit $LASTEXITCODE) — GitHub 웹 > Settings > Collaborators 에서 직접 제거할 것"
}

# ── 완료 출력 ─────────────────────────────────────────────────────────────────
$hash = git rev-parse HEAD
Write-Output ""
Write-Output "완료"
Write-Output "  저장소     : https://github.com/$Repo"
Write-Output "  가중치 URL : $url"
Write-Output "  커밋 해시  : $hash"
Write-Output ""
Write-Output "제출 메일에 위 저장소 URL 과 커밋 해시를 기재할 것."
if (-not $Public) {
    Write-Output "비공개 저장소입니다. 제출 전 공개로 바꾸거나 운영진 계정(b9511242000-blip)을 협력자로 추가할 것."
}
