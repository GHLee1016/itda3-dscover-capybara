#Requires -Version 5.1
<#
    본선보고서.html -> [DScover]_카피바라_본선보고서.pdf

    규정: PDF · A4 · 본문 11pt 고정 · 표지·목차 제외 최대 10페이지
          파일명 [학회명]_팀명_본선보고서.pdf

        powershell -ExecutionPolicy Bypass -File docs\build_report_pdf.ps1

    왜 Word 가 아니라 헤드리스 Edge 인가
      docx -> PDF 를 Word COM 으로 시도했으나 폰트 대체 처리에서 멈췄다
      (CPU 86초 소비 후에도 무응답, 두 번 재현). 마감이 임박해 이 프로젝트에서
      이미 검증된 Edge 인쇄 경로로 바꿨다. 여백 25.4mm·줄간격은 CSS 로 고정했다.

    ⚠️ 이 파일은 **UTF-8 BOM** 으로 저장해야 한다. PowerShell 5.1 은 BOM 이
       없는 .ps1 을 ANSI 로 읽어 한글 경로를 깨뜨린다(실제로 겪었다).
#>
$ErrorActionPreference = "Stop"

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$src  = Join-Path $here "본선보고서.html"
$out  = Join-Path $here "[DScover]_카피바라_본선보고서.pdf"

$edge = "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
if (-not (Test-Path $edge)) { $edge = "C:\Program Files\Microsoft\Edge\Application\msedge.exe" }
if (-not (Test-Path $edge)) { throw "Edge 를 찾지 못했다. 브라우저에서 본선보고서.html 을 열고 Ctrl+P → PDF 로 저장할 것." }
if (-not (Test-Path -LiteralPath $src)) { throw "원본이 없다: $src" }

if (Test-Path -LiteralPath $out) { Remove-Item -LiteralPath $out -Force -Confirm:$false }

# Edge 는 정상 동작 중에도 stderr 로 경고를 뱉는다. PowerShell 5.1 에서 2>&1 로
# 받으면 NativeCommandError 가 되어 스크립트가 죽으므로 쓰지 않는다.
$uri = "file:///" + (Resolve-Path -LiteralPath $src).Path.Replace('\', '/')
$ErrorActionPreference = "Continue"
& $edge --headless --disable-gpu --no-pdf-header-footer `
        --virtual-time-budget=20000 --print-to-pdf="$out" $uri
$ErrorActionPreference = "Stop"

# Edge 는 프로세스가 끝난 뒤에도 파일을 마저 쓴다
for ($i = 0; $i -lt 40; $i++) {
    if (Test-Path -LiteralPath $out) { break }
    Start-Sleep -Milliseconds 500
}
if (-not (Test-Path -LiteralPath $out)) { throw "PDF 생성 실패" }
Start-Sleep -Seconds 1

Write-Output ("생성: {0}" -f $out)
Write-Output ("크기: {0:N0} KB" -f ((Get-Item -LiteralPath $out).Length / 1KB))
Write-Output "분량·규정 점검: python docs\check_report.py"
