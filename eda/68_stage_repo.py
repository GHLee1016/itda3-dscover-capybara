"""
EDA 68단계 — GitHub 에 올릴 파일만 따로 모은다

팀원이 저장소에 올리기로 했으므로, `.gitignore` 규칙을 실제로 적용해서
**커밋될 파일만** 별도 폴더로 복사한다. 팀원은 그 폴더 내용을 저장소 루트에
그대로 붙여 넣으면 된다.

`.gitignore` 판정은 눈으로 하지 않고 git 에게 직접 물어본다 —
임시 폴더에 빈 저장소를 만들고 우리 .gitignore 를 넣은 뒤
`git check-ignore` 로 경로를 하나씩 판정시킨다. 프로젝트에는 .git 을 만들지 않는다
(팀원이 저장소를 관리하므로 로컬에 .git 이 생기면 헷갈린다).

    python eda/68_stage_repo.py
"""
import os
import sys
import shutil
import subprocess
import tempfile
import argparse

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DST = os.path.join(ROOT, "_github_upload")

# 걸어도 의미 없는 대용량 디렉터리는 아예 훑지 않는다 (.gitignore 로도 걸린다)
SKIP_WALK = {".venv", ".venv-team", ".venv-team2", "images", ".git",
             "__pycache__", "_github_upload"}


def list_files():
    out = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        rel = os.path.relpath(dirpath, ROOT)
        parts = set(rel.split(os.sep))
        if parts & SKIP_WALK:
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames if d not in SKIP_WALK]
        for f in filenames:
            p = f if rel == "." else os.path.join(rel, f)
            out.append(p.replace(os.sep, "/"))
    return sorted(out)


def ignored_set(paths):
    """
    우리 .gitignore 로 무시되는 경로 집합을 git 에게 직접 물어본다.

    ⚠️ 반드시 -z (NUL 구분) + 바이너리 파이프로 주고받아야 한다.
       text=True 로 하면 Windows 에서 파이썬이 입력의 \\n 을 \\r\\n 으로 바꿔
       git 이 경로를 `images.zip\\r` 로 읽어 규칙이 안 걸린다. 출력 쪽도
       한글 경로가 core.quotepath 때문에 8진수로 escape 되어 돌아온다.
       (둘 다 실제로 겪은 오작동이다 — 무시 목록이 통째로 비어 나왔다)
    """
    tmp = tempfile.mkdtemp(prefix="itda_ignore_")
    try:
        shutil.copyfile(os.path.join(ROOT, ".gitignore"),
                        os.path.join(tmp, ".gitignore"))
        subprocess.run(["git", "init", "-q"], cwd=tmp, check=True)
        # check-ignore 는 파일이 실제로 없어도 규칙만으로 판정한다
        data = b"\0".join(p.encode("utf-8") for p in paths)
        r = subprocess.run(["git", "-c", "core.quotepath=false",
                            "check-ignore", "-z", "--stdin"],
                           cwd=tmp, input=data, capture_output=True)
        return {b.decode("utf-8") for b in r.stdout.split(b"\0") if b}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst", default=DST)
    a = ap.parse_args()
    dst = os.path.abspath(a.dst)

    paths = list_files()
    ign = ignored_set(paths)
    keep = [p for p in paths if p not in ign]

    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)

    total = 0
    for p in keep:
        src = os.path.join(ROOT, p.replace("/", os.sep))
        tgt = os.path.join(dst, p.replace("/", os.sep))
        os.makedirs(os.path.dirname(tgt), exist_ok=True)
        shutil.copy2(src, tgt)
        total += os.path.getsize(src)

    # 규정 구조상 weights/ 폴더는 저장소에 있어야 한다 (안은 비운다)
    wdir = os.path.join(dst, "weights")
    os.makedirs(wdir, exist_ok=True)
    with open(os.path.join(wdir, ".gitkeep"), "w", encoding="utf-8") as f:
        f.write("# 가중치 저장 경로. bash download_weights.sh 가 이 안을 채운다.\n"
                "# 가중치 파일 자체는 규정상 커밋하지 않는다.\n")

    by_dir = {}
    for p in keep:
        d = p.split("/")[0] if "/" in p else "(루트)"
        by_dir.setdefault(d, []).append(p)

    print(f"→ {dst}")
    print(f"  파일 {len(keep)}개, {total/1e6:.2f} MB\n")
    for d in sorted(by_dir):
        sz = sum(os.path.getsize(os.path.join(ROOT, p.replace('/', os.sep)))
                 for p in by_dir[d])
        print("  %-14s %4d개  %7.2f MB" % (d, len(by_dir[d]), sz / 1e6))

    print("\n  루트 파일:")
    for p in by_dir.get("(루트)", []):
        print("    ", p)

    big = [(p, os.path.getsize(os.path.join(ROOT, p.replace("/", os.sep))))
           for p in keep]
    big = [x for x in big if x[1] > 5_000_000]
    print("\n  5MB 초과:", big if big else "없음 (GitHub 단일 파일 100MB 제한 여유)")

    print("\n  제외된 것 중 큰 것:")
    excl = [(p, os.path.getsize(os.path.join(ROOT, p.replace("/", os.sep))))
            for p in paths if p in ign]
    for p, s in sorted(excl, key=lambda x: -x[1])[:6]:
        print("    %-58s %8.1f MB" % (p, s / 1e6))


if __name__ == "__main__":
    main()
