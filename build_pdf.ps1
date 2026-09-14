# 요약서.html → PDF (A4 2장)
#
# 헤드리스 Edge 로 인쇄한다. CSS @page 와 한글 폰트(맑은 고딕)를 그대로 쓰므로
# 브라우저에서 Ctrl+P 로 뽑은 것과 결과가 같다.
#
#   powershell -ExecutionPolicy Bypass -File build_pdf.ps1
#
# 파일명 형식은 규정 그대로다: [학회명_팀명]_아키텍처구조도.pdf

$ErrorActionPreference = "Stop"
$edge = "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
if (-not (Test-Path $edge)) { $edge = "C:\Program Files\Microsoft\Edge\Application\msedge.exe" }
if (-not (Test-Path $edge)) { throw "Edge 를 찾지 못했다. 브라우저에서 요약서.html 을 열고 Ctrl+P → PDF 로 저장할 것." }

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$src  = Join-Path $root "요약서.html"
$out  = Join-Path $root "[DScover]_카피바라_아키텍처구조도.pdf"
$uri  = "file:///" + $src.Replace('\', '/')

if (Test-Path -LiteralPath $out) { Remove-Item -LiteralPath $out -Force -Confirm:$false }

# --no-pdf-header-footer: 페이지 상하단에 URL·날짜가 찍히지 않게 한다
#
# ⚠️ Edge 는 정상 동작 중에도 stderr 로 경고를 뱉는다. PowerShell 5.1 에서
#    `2>&1` 로 받으면 NativeCommandError 가 되어 스크립트가 죽으므로 쓰지 않는다.
#
# --virtual-time-budget: 머메이드가 SVG 를 **비동기로** 그린다. 이 옵션 없이 인쇄하면
#    다이어그램 자리가 빈 채로 찍힌다. 가상 시계를 20초까지 빨리 감아 렌더를 끝낸다.
$ErrorActionPreference = "Continue"
& $edge --headless --disable-gpu --no-pdf-header-footer --virtual-time-budget=20000 --print-to-pdf="$out" $uri
$ErrorActionPreference = "Stop"

# Edge 는 프로세스가 끝난 뒤에도 파일을 마저 쓴다. 나타날 때까지 기다린다.
# 대괄호가 든 경로는 -LiteralPath 로 다뤄야 한다 (와일드카드로 해석됨)
for ($i = 0; $i -lt 30; $i++) {
    if (Test-Path -LiteralPath $out) { break }
    Start-Sleep -Milliseconds 500
}
if (-not (Test-Path -LiteralPath $out)) { throw "PDF 생성 실패" }
Start-Sleep -Seconds 1
$kb = [math]::Round((Get-Item -LiteralPath $out).Length / 1KB, 0)
Write-Output "생성: $out ($kb KB)"
