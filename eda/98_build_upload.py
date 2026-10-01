"""
EDA 98단계 — `_github_upload/` 를 다시 만든다

팀원이 그대로 저장소에 올릴 수 있도록, `.gitignore` 를 실제로 적용해
커밋될 파일만 골라 담는다. 규칙을 직접 해석하지 않고 **git 에게 물어본다** —
부정 패턴(`!`)과 디렉터리 규칙까지 정확히 맞추려면 그게 안전하다.

이 저장소는 git repo 가 아니므로 임시 폴더에 빈 repo 를 만들고 `.gitignore`
만 옮겨 `git check-ignore` 로 판정한다. 경로만으로 판정하므로 파일을
복사하지 않아도 된다.

⚠️ 두 가지를 틀리기 쉽다.
   · `-z` 를 주면 **입력도 NUL 로 구분**해야 한다. 개행으로 넘기면 전부
     한 경로로 취급되어 판정이 통째로 실패한다(실제로 그렇게 됐었다).
   · Windows 에서는 **바이너리 파이프**로 써야 한다. 텍스트 모드는
     `\\n` 을 `\\r\\n` 으로 바꿔 경로를 깨뜨린다.

    python eda/98_build_upload.py
"""
import os
import sys
import shutil
import subprocess
import tempfile

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(ROOT, "_github_upload")

# 용량이 크거나 규정상 Git 에 올리지 않는 것 (.gitignore 와 별개로 한 번 더 막는다)
HARD_SKIP = {".git", "_github_upload", "images", "yolo_ds", "yolo_runs",
             "라벨대조", "라벨대조2", "오답모음", "오답사진", "실패샘플_P6",
             "yolo_시각화", "팀원 기록", "label",
             ".venv", ".venv-team", ".venv-team2"}
HARD_SKIP_FILE = {"images.zip", "_github_upload.zip", "yolov8n.pt"}
MAX_MB = 90          # GitHub 단일 파일 100MB 제한


def collect():
    out = []
    for cur, dirs, files in os.walk(ROOT):
        rel = os.path.relpath(cur, ROOT)
        if rel == ".":
            rel = ""
        parts = rel.split(os.sep) if rel else []
        if any(p in HARD_SKIP for p in parts):
            dirs[:] = []
            continue
        dirs[:] = [d for d in dirs if d not in HARD_SKIP]
        for f in files:
            if f in HARD_SKIP_FILE or f.endswith(".bak"):
                continue
            out.append(os.path.join(rel, f) if rel else f)
    return sorted(out)


def ignored(paths):
    """git 에게 무시 대상인지 물어본다. 무시되는 경로의 집합을 낸다."""
    tmp = tempfile.mkdtemp(prefix="ignorecheck-")
    try:
        subprocess.run(["git", "init", "-q"], cwd=tmp, check=True,
                       capture_output=True)
        shutil.copy2(os.path.join(ROOT, ".gitignore"),
                     os.path.join(tmp, ".gitignore"))
        # -z 를 주면 입력도 NUL 구분이어야 한다
        blob = "\0".join(p.replace(os.sep, "/") for p in paths).encode("utf-8")
        r = subprocess.run(["git", "check-ignore", "-z", "--stdin"],
                           cwd=tmp, input=blob, capture_output=True)
        got = r.stdout.decode("utf-8").split("\0")
        return {g.replace("/", os.sep) for g in got if g}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    if not shutil.which("git"):
        print("git 이 없다. 이 스크립트는 .gitignore 판정에 git 을 사용한다.")
        return

    allp = collect()
    skip = ignored(allp)
    keep = [p for p in allp if p not in skip]
    print(f"후보 {len(allp)}개 · .gitignore 로 제외 {len(skip)}개 · 담을 것 {len(keep)}개")

    big = []
    for p in keep:
        mb = os.path.getsize(os.path.join(ROOT, p)) / 1024 / 1024
        if mb > MAX_MB:
            big.append((p, mb))
    if big:
        print("\n⚠️ 단일 파일이 너무 크다 (GitHub 100MB 제한):")
        for p, mb in big:
            print(f"   {p}  {mb:.0f}MB")
        print("   Release Assets 로 올리고 download_weights.sh 에 URL 을 넣을 것.")
        keep = [p for p in keep if p not in {b[0] for b in big}]

    if os.path.exists(DEST):
        shutil.rmtree(DEST)
    total = 0
    for p in keep:
        src = os.path.join(ROOT, p)
        dst = os.path.join(DEST, p)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        total += os.path.getsize(src)

    print(f"\n{len(keep)}개 / {total/1024/1024:.1f}MB → _github_upload/")
    from collections import Counter
    top = Counter(p.split(os.sep)[0] if os.sep in p else "(루트)" for p in keep)
    for k, v in top.most_common():
        print(f"   {k:16s} {v}개")

    # 제출에 꼭 있어야 하는 것들이 실제로 담겼는지 확인한다
    must = ["predict.ipynb", "requirements.txt", "download_weights.sh",
            "README.md", "METHOD.md", "JOURNAL.md",
            "check_submission.py", "audit_notebook.py", "build_notebook.py",
            "run_local.py",
            os.path.join("src", "combined.py"),
            os.path.join("src", "team2_rules.py"),
            os.path.join("src", "pipeline.py"),
            os.path.join("src", "dotcrnn.py"),
            os.path.join("src", "rejected_rungs.py"),
            os.path.join("custom_data", "labels.csv"),
            "[DScover]_카피바라_아키텍처구조도.pdf"]
    miss = [m for m in must if not os.path.exists(os.path.join(DEST, m))]
    print("\n필수 파일:", "모두 포함" if not miss else f"⚠️ 빠짐 {miss}")

    # 들어가면 안 되는 것 — 규정(원본 이미지 재배포 금지)과 용량
    bad = []
    for cur, _, files in os.walk(DEST):
        for f in files:
            p = os.path.join(cur, f)
            rel = os.path.relpath(p, DEST)
            if os.path.splitext(f)[1].lower() in (
                    ".jpg", ".jpeg", ".png", ".onnx", ".pt", ".pdmodel",
                    ".pdiparams", ".zip"):
                bad.append(rel)
    print("금지 확장자:", "없음" if not bad else f"⚠️ {len(bad)}개 {bad[:5]}")
    print("용량 상위:")
    sz = sorted(((os.path.getsize(os.path.join(DEST, p)), p) for p in keep),
                reverse=True)[:5]
    for n, p in sz:
        print(f"   {n/1024/1024:6.1f}MB  {p}")


if __name__ == "__main__":
    main()
