"""
제출물 최종 점검 — 규정 §7/§8 형식 요건을 코드로 확인한다.

    python check_submission.py

체크리스트를 눈으로 훑는 대신 기계가 보게 한다. 형식 실수는 정량 전량 0점으로
이어지는데(to_csv 에서 index=False 를 빠뜨리면 컬럼이 통째로 밀린다) 사람 눈으로는
잘 걸러지지 않는다. 제출 직전과 코드를 고칠 때마다 돌린다.

가중치·PDF 는 저장소에 없을 수 있다(규정상 Release 배포). 그 항목이 실패하면
download_weights.sh / build_pdf.ps1 을 먼저 실행할 것.
"""
import json, io, os, re, sys

sys.stdout.reconfigure(encoding="utf-8")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
ok = []


def chk(name, cond, detail=""):
    ok.append(cond)
    print("  %s %-46s %s" % ("OK " if cond else "!! ", name, detail))


print("=== predict.ipynb")
nb = json.load(io.open("predict.ipynb", encoding="utf-8"))
cells = nb["cells"]
src = lambda c: "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
all_src = "\n".join(src(c) for c in cells)
code = [c for c in cells if c["cell_type"] == "code"]

chk("첫 코드셀이 os.environ.get 형태",
    'os.environ.get("ITDA_INPUT_DIR"' in src(code[0])
    and 'os.environ.get("ITDA_OUTPUT_PATH"' in src(code[0]))
chk("index=False 로 저장", "to_csv(OUTPUT_PATH, index=False)" in all_src)
chk("torch/ultralytics 임포트 없음",
    not re.search(r"^\s*(?:import|from)\s+(?:torch|ultralytics)\b", all_src, re.M))
chk("input() 등 입력 대기 없음",
    not re.search(r"\binput\s*\(|getpass|Read-Host", all_src))
chk("셀 출력이 비어 있음",
    all(not c.get("outputs") for c in code))
chk("로컬 절대경로 하드코딩 없음",
    not re.search(r"[Cc]:\\\\Users|/content/drive|\./data\b", all_src))
chk("주력 엔진(RapidOCR) 포함", "build_rapid_ocr" in all_src and "RapidOCR" in all_src)
chk("보조 경로(DatePipeline) 포함", "class DatePipeline" in all_src)
chk("날짜 검출기(ONNX) 포함", "class OnnxDateDetector" in all_src)
chk("결합 층 포함", "class CombinedPipeline" in all_src)
chk("가중치 없을 때 강하 처리", "det 단독으로 진행" in all_src)
chk("셀 수", len(cells) == 15, f"{len(cells)}셀")

print("\n=== requirements.txt")
req = io.open("requirements.txt", encoding="utf-8").read()
for pkg in ("rapidocr==3.9.2", "onnxruntime==1.19.2", "paddleocr==2.7.3",
            "numpy==1.26.4", "opencv-python==4.10.0.84", "nbconvert", "ipykernel"):
    chk(f"{pkg} 포함", pkg.split("==")[0] in req and (pkg in req or "==" not in pkg))
chk("torch/ultralytics 없음",
    not re.search(r"^\s*(?:torch|ultralytics)\b", req, re.M))

print("\n=== 가중치")
for p, mb in (("weights/rapidocr/PP-OCRv6_det_small.onnx", 9),
              ("weights/rapidocr/PP-OCRv6_rec_small.onnx", 20),
              ("weights/rapidocr/ch_ppocr_mobile_v2.0_cls_mobile.onnx", 0.5),
              ("weights/yolo/date_v1.onnx", 10)):
    e = os.path.exists(p)
    chk(os.path.basename(p), e,
        f"{os.path.getsize(p)/1e6:.1f} MB" if e else "없음")

print("\n=== 요약서 PDF")
import pymupdf
d = pymupdf.open("[DScover]_카피바라_아키텍처구조도.pdf")
chk("A4 2장", d.page_count == 2 and abs(d[0].rect.width / 72 * 25.4 - 210) < 1,
    f"{d.page_count}장")
t = "\n".join(p.get_text() for p in d)
chk("텍스트 추출 가능 (스캔본 아님)", len(t) > 3000, f"{len(t)}자")
chk("팀명 기재", "카피바라" in t)
chk("최종 수치 반영", "82.3%" in t and "0.53" in t)
chk("자리표시자 없음", "팀명_" not in t and "<ORG>" not in t)

print("\n=== 문서")
for f in ("README.md", "METHOD.md", "JOURNAL.md", "WHERE.md",
          "SUBMISSION_CHECKLIST.md", "RULES_QNA.md", "download_weights.sh",
          "requirements-train.txt", "build_notebook.py", "build_pdf.ps1",
          "publish_release.ps1", "요약서.html"):
    chk(f, os.path.exists(f))

print("\n=== 소스")
for f in ("src/pipeline.py", "src/team2_rules.py", "src/combined.py"):
    chk(f, os.path.exists(f))
import ast
for f in ("src/pipeline.py", "src/team2_rules.py", "src/combined.py",
          "build_notebook.py"):
    try:
        ast.parse(io.open(f, encoding="utf-8").read())
        chk(f"{f} 구문", True)
    except SyntaxError as e:
        chk(f"{f} 구문", False, str(e))

print("\n" + "=" * 62)
print("통과 %d / %d" % (sum(ok), len(ok)))
if all(ok):
    print("모든 점검 통과. 남은 것은 Release 업로드와 커밋 해시뿐이다.")
