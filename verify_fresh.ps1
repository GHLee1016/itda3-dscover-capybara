#Requires -Version 5.1
<#
    새 가상환경에서 제출 절차를 처음부터 수행하고 로그를 남긴다.

    규정: "제출 전 아래 절차를 새 가상환경에서 반드시 수행하고,
           실행 로그를 저장소에 포함해 주세요."

        powershell -ExecutionPolicy Bypass -File verify_fresh.ps1
        powershell -ExecutionPolicy Bypass -File verify_fresh.ps1 -N 500

    로그는 logs/ 에 남고 저장소에 포함된다.

    ⚠️ 이 스크립트는 ASCII 경로에서 돌려야 한다. PaddleOCR 의 C++ 백엔드가
       비ASCII 경로의 모델을 열지 못해 보조 경로가 **에러 없이** 꺼진다.
       -WorkDir 기본값을 %PUBLIC% 아래로 잡아 둔 이유다.
#>
param(
    [int]$N = 500,
    [string]$WorkDir = (Join-Path $env:PUBLIC "itda_fresh")
)

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$logDir = Join-Path $root "logs"
New-Item -ItemType Directory $logDir -Force | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$log = Join-Path $logDir ("fresh_run_{0}.log" -f $stamp)

function Say($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $msg
    Write-Output $line
    Add-Content -LiteralPath $log -Value $line -Encoding UTF8
}

function Run($label, $exe, $argList) {
    Say "---- $label"
    Say "     > $exe $($argList -join ' ')"
    $out = & $exe @argList 2>&1 | Out-String
    Add-Content -LiteralPath $log -Value $out -Encoding UTF8
    Say "     exit=$LASTEXITCODE"
    return $out
}

Say "=============================================================="
Say " 새 가상환경 재현성 검증 — [DScover]_카피바라"
Say "=============================================================="
Say "작업 경로 : $WorkDir"
Say "저장소     : $root"
Say "OS         : $([Environment]::OSVersion.VersionString)"
Say "CPU        : $((Get-CimInstance Win32_Processor).Name)"
Say "코어       : $([Environment]::ProcessorCount)"
Say "처리 장수  : $N"
Say ""

# 1) 저장소 사본 — _github_upload/ 가 '올릴 것만' 담은 폴더다
$stage = Join-Path $root "_github_upload"
if (-not (Test-Path $stage)) { Say "!! _github_upload 가 없다. python eda/98_build_upload.py 먼저"; exit 1 }
if (Test-Path $WorkDir) { Remove-Item $WorkDir -Recurse -Force }
New-Item -ItemType Directory $WorkDir -Force | Out-Null
Copy-Item (Join-Path $stage "*") $WorkDir -Recurse -Force
Say "저장소 사본 복사 완료 ($((Get-ChildItem $WorkDir -Recurse -File).Count) 파일)"

Push-Location $WorkDir
try {
    # 2) 새 가상환경
    Run "새 가상환경 생성" "python" @("-m", "venv", ".venv") | Out-Null
    $py = Join-Path $WorkDir ".venv\Scripts\python.exe"
    if (-not (Test-Path $py)) { Say "!! venv 생성 실패"; exit 1 }
    Run "python 버전" $py @("--version") | Out-Null
    Run "의존성 설치" $py @("-m", "pip", "install", "-q", "-r", "requirements.txt") | Out-Null
    Run "설치된 패키지" $py @("-m", "pip", "freeze") | Out-Null

    # 3) 가중치 — 인터넷 필요. Release URL 이 비어 있으면 여기서 경고가 난다
    if (Get-Command bash -ErrorAction SilentlyContinue) {
        Run "가중치 내려받기" "bash" @("download_weights.sh") | Out-Null
    } else {
        Say "!! bash 가 없어 download_weights.sh 를 건너뛴다 (Git Bash 설치 필요)"
        Say "   저장소의 weights/ 를 그대로 복사해 진행한다"
        Copy-Item (Join-Path $root "weights\*") (Join-Path $WorkDir "weights") -Recurse -Force
    }
    Say "weights/ 내용:"
    Get-ChildItem (Join-Path $WorkDir "weights") -Recurse -File |
        ForEach-Object { Say ("     {0,-46} {1,8:N1} MB" -f $_.Name, ($_.Length/1MB)) }

    # 4) 형식 점검
    Run "제출 형식 점검" $py @("check_submission.py") | Out-Null
    Run "노트북 네트워크 감사" $py @("audit_notebook.py") | Out-Null

    # 5) 추론 — 채점과 같은 방식
    $imgSrc = Join-Path $root "images\상품사진입니다"
    $evalDir = Join-Path $WorkDir "eval_input"
    New-Item -ItemType Directory $evalDir -Force | Out-Null
    $rand = New-Object System.Random 11
    Get-ChildItem $imgSrc -File |
        Where-Object { $_.Extension -match '\.(jpe?g|png)$' } |
        Sort-Object { $rand.Next() } | Select-Object -First $N |
        ForEach-Object { New-Item -ItemType HardLink -Path (Join-Path $evalDir $_.Name) -Target $_.FullName | Out-Null }
    Say "입력 $N 장 준비: $evalDir"

    $env:ITDA_INPUT_DIR = $evalDir
    $env:ITDA_OUTPUT_PATH = Join-Path $WorkDir "submission.csv"
    Say "ITDA_INPUT_DIR  = $env:ITDA_INPUT_DIR"
    Say "ITDA_OUTPUT_PATH= $env:ITDA_OUTPUT_PATH"

    $sw = [Diagnostics.Stopwatch]::StartNew()
    Run "노트북 실행 (nbconvert)" $py @(
        "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
        "predict.ipynb", "--ExecutePreprocessor.timeout=2400",
        "--output", (Join-Path $WorkDir "executed.ipynb")) | Out-Null
    $sw.Stop()

    Say ""
    Say "=============================================================="
    Say (" 전체 실행 시간 : {0:N0}초 ({1:N1}분)" -f $sw.Elapsed.TotalSeconds, $sw.Elapsed.TotalMinutes)
    Say (" 제한 2,400초 대비: {0:N0}%" -f ($sw.Elapsed.TotalSeconds / 2400 * 100))
    if (Test-Path -LiteralPath $env:ITDA_OUTPUT_PATH) {
        $rows = (Import-Csv -LiteralPath $env:ITDA_OUTPUT_PATH).Count
        Say (" submission.csv : {0}행" -f $rows)
        Say " 판정: " + $(if ($sw.Elapsed.TotalSeconds -le 2400 -and $rows -eq $N) { "통과" } else { "확인 필요" })
    } else {
        Say " !! submission.csv 가 생기지 않았다"
    }
    Say "=============================================================="
} finally {
    Pop-Location
}

Write-Output ""
Write-Output "로그: $log"
