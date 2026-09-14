"""
predict.ipynb 0점 유발 요인 감사.

운영진이 공지한 네 가지 원인을 코드로 확인한다.
  ① 실행 중 모델 가중치를 자동 다운로드하는 코드
  ② requirements.txt 에 누락된 패키지
  ③ 노트북 내 하드코딩된 로컬 경로
  ④ GPU 전용 코드 (cuda 설정)

    python audit_notebook.py

주석은 검사에서 뺀다 — 설명문에 'https://' 가 나오는 것은 문제가 아니다.
"""
import ast
import io
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

nb = json.load(io.open("predict.ipynb", encoding="utf-8"))
cells = nb["cells"]


def src(c):
    s = c["source"]
    return "".join(s) if isinstance(s, list) else s


ALL = "\n".join(src(c) for c in cells)
CODE_CELLS = [src(c) for c in cells if c["cell_type"] == "code"]
CODE = "\n".join(CODE_CELLS)
# 주석·독스트링을 걷어낸 '실행되는 코드'만 남긴다
BARE = "\n".join(l for l in CODE.splitlines() if not l.lstrip().startswith("#"))

bad = []


def check(no, name, pattern, text=BARE, invert=False):
    hits = [l.strip() for l in text.splitlines() if re.search(pattern, l)]
    fail = (not hits) if invert else bool(hits)
    print("  %s %s" % ("!!" if fail else "OK", name))
    for h in hits[:5]:
        print("       ", h[:110])
    if fail:
        bad.append(name)


print("=== ① 실행 중 자동 다운로드")
check(1, "URL / 다운로드 호출이 실행 코드에 없음",
      r"https?://|urlretrieve|urlopen|requests\.(get|post)|wget|curl|"
      r"download_enabled\s*=\s*True|hf_hub|snapshot_download|from_pretrained")

# 아래 두 검사는 정규식으로는 정확하지 않다. 호출이 여러 줄에 걸치고,
# 독스트링 안의 같은 문자열이 걸린다. 구문 트리에서 실제 호출만 본다.
def calls_of(func_name):
    """노트북 코드 셀에서 func_name(...) 호출 노드를 전부 모은다."""
    out = []
    for cell in CODE_CELLS:
        try:
            tree = ast.parse(cell)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                f = node.func
                nm = getattr(f, "id", None) or getattr(f, "attr", None)
                if nm == func_name:
                    out.append(node)
    return out


def check_ctor(no, name, func, required_kw):
    """func(...) 호출이 required_kw 중 하나라도 갖추고 있는지 본다."""
    calls = calls_of(func)
    offenders = [c for c in calls
                 if not any(k.arg and k.arg in required_kw for k in c.keywords)
                 and not any(isinstance(k.arg, type(None)) for k in c.keywords)]
    # **kwargs 로 넘기는 형태도 허용 (keywords 에 arg=None 으로 들어온다)
    offenders = [c for c in offenders
                 if not any(k.arg is None for k in c.keywords)]
    fail = bool(offenders)
    print("  %s %s  (호출 %d건)" % ("!!" if fail else "OK", name, len(calls)))
    for c in offenders[:3]:
        print("        line %d: 인자에 %s 가 없다" % (c.lineno, "/".join(required_kw)))
    if fail:
        bad.append(name)


check_ctor(1, "RapidOCR 을 모델 경로 없이 만들지 않음(만들면 받으러 나간다)",
           "RapidOCR", {"params"})
check_ctor(1, "PaddleOCR 을 model_dir 없이 만들지 않음",
           "PaddleOCR", {"det_model_dir", "rec_model_dir", "cls_model_dir"})

print("\n=== ③ 하드코딩된 로컬 경로")
check(3, "절대경로 / 고정 데이터 경로 없음",
      r"[A-Za-z]:[\\/]{1,2}Users|/content/|/home/|~/|\./data\b|\./val_images(?!\"\))")
check(3, "INPUT_DIR / OUTPUT_PATH 를 환경변수에서 읽음",
      r'os\.environ\.get\("ITDA_(INPUT_DIR|OUTPUT_PATH)"', invert=True)

print("\n=== ④ GPU 전용 코드")
check(4, "cuda / GPU 지정 없음",
      r"\.cuda\(|cuda:|device\s*=\s*[\"']cuda|use_gpu\s*=\s*True|gpu\s*=\s*True|"
      r"CUDAExecutionProvider|torch\.cuda")
check(4, "use_gpu=False 를 명시",
      r"use_gpu\s*=\s*False", invert=True)
check(4, "onnxruntime 를 CPUExecutionProvider 로 고정",
      r'providers=\["CPUExecutionProvider"\]', invert=True)

print("\n=== ② requirements.txt 대 실제 import")
req = io.open("requirements.txt", encoding="utf-8").read()
declared = set()
for line in req.splitlines():
    line = line.split("#")[0].strip()
    if not line or line.startswith("-"):
        continue
    declared.add(re.split(r"[=<>!~\[]", line)[0].strip().lower().replace("-", "_"))

# 노트북이 실제로 import 하는 최상위 모듈
imported = set()
for cell in CODE_CELLS:
    try:
        tree = ast.parse(cell)
    except SyntaxError:
        continue
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                imported.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])

STDLIB = set(sys.stdlib_module_names)
PKG2DIST = {"cv2": "opencv_python", "PIL": "pillow", "sklearn": "scikit_learn",
            "yaml": "pyyaml", "paddleocr": "paddleocr", "paddle": "paddlepaddle"}
third = sorted(m for m in imported if m not in STDLIB)
missing = []
for m in third:
    dist = PKG2DIST.get(m, m).lower().replace("-", "_")
    if dist not in declared:
        missing.append(m)
print("  노트북이 import 하는 외부 패키지:", ", ".join(third) or "없음")
print("  %s requirements.txt 에 모두 선언됨%s" % (
    "OK" if not missing else "!!", "" if not missing else " — 누락: " + ", ".join(missing)))
if missing:
    bad.append("requirements 누락")

for must in ("nbconvert", "ipykernel"):
    ok = must in declared
    print("  %s %s 포함 (노트북 실행에 필수)" % ("OK" if ok else "!!", must))
    if not ok:
        bad.append(f"{must} 누락")

pinned = [l.split("#")[0].strip() for l in req.splitlines()
          if l.split("#")[0].strip() and not l.strip().startswith("-")]
unpinned = [p for p in pinned if "==" not in p]
print("  %s 모든 패키지 버전 고정%s" % (
    "OK" if not unpinned else "!!", "" if not unpinned else " — " + ", ".join(unpinned)))
if unpinned:
    bad.append("버전 미고정")

print("\n=== 그 밖의 형식 요건")
check(0, "input() 등 입력 대기 없음", r"\binput\s*\(|getpass")
check(0, "index=False 로 저장", r"to_csv\([^)]*index=False", invert=True)
check(0, "첫 코드셀이 CONFIG 셀", r"ITDA_INPUT_DIR", CODE_CELLS[0], invert=True)
print("  %s 셀 출력이 비어 있음" % (
    "OK" if all(not c.get("outputs") for c in cells if c["cell_type"] == "code") else "!!"))

print("\n" + "=" * 60)
if bad:
    print("문제 %d건: %s" % (len(bad), ", ".join(bad)))
    sys.exit(1)
print("0점 유발 요인 없음.")
