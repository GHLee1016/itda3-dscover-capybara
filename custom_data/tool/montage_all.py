"""남은 라벨링 대상 전체를 4장씩 묶어 몽타주로 만든다."""
import os
import sys
import json
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "eda", "out")
sys.stdout.reconfigure(encoding="utf-8")

todo = json.load(open(os.path.join(OUT_DIR, "label_remaining.json"), encoding="utf-8"))
files = [t["file"] for t in todo]
py = sys.executable
script = os.path.join(ROOT, "label", "montage.py")

plan = []
for i in range(0, len(files), 4):
    grp = files[i:i + 4]
    name = "m%02d.jpg" % (i // 4 + 1)
    subprocess.run([py, script, "--files", ",".join(grp), "--out", name],
                   check=True, capture_output=True)
    plan.append({"montage": name, "files": grp})

json.dump(plan, open(os.path.join(OUT_DIR, "montage_plan.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print(f"몽타주 {len(plan)}개 생성 (총 {len(files)}장)")
for p in plan:
    print(f"  {p['montage']}: {', '.join(os.path.splitext(f)[0] for f in p['files'])}")
