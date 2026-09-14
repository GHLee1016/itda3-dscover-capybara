"""
EDA 36단계 — 홀드아웃 표본 생성

지금까지의 모든 설계 결정(해상도 사다리, 전처리 선택, 부터/까지 재시도)을
동일한 정답 137장으로 골랐다. 개선폭이 +0.5~+1.5%p 수준이라 **표본에 맞춰
고른 것일 가능성**을 배제할 수 없다.

한 번도 쓰지 않은 무작위 표본을 새로 뽑아, 튜닝에 전혀 관여하지 않은 상태에서
같은 파이프라인을 측정한다. 여기서 성능이 유지되면 과적합이 아니다.
"""
import os
import sys
import json
import argparse
import subprocess

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "eda", "out")
IMG_DIR = os.path.join(ROOT, "images", "상품사진입니다")
DST = os.path.join(ROOT, "label", "imgs")

import cv2


def save(path, img, q=88):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    if ok:
        buf.tofile(path)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=48)
    a = ap.parse_args()

    t = json.load(open(os.path.join(OUT_DIR, "truth_v4.json"), encoding="utf-8"))
    used = set(t["labels"]) | set(t.get("ambiguous", {}))
    # 액티브 풀도 제외 — 선발 과정에서 통계를 본 적이 있다
    pool_seen = set()
    ap_csv = os.path.join(OUT_DIR, "active_pool.csv")
    if os.path.exists(ap_csv):
        pool_seen = set(pd.read_csv(ap_csv)["file"])

    meta = pd.read_csv(os.path.join(OUT_DIR, "metadata.csv"))
    fresh = meta[~meta["file"].isin(used | pool_seen)]
    print(f"후보 풀 {len(fresh)}장 (전체 {len(meta)} − 사용됨 {len(used | pool_seen)})")

    rng = np.random.default_rng(31337)
    pick = fresh.sample(min(a.n, len(fresh)), random_state=31337)["file"].tolist()

    os.makedirs(DST, exist_ok=True)
    made = 0
    for f in pick:
        dst = os.path.join(DST, os.path.splitext(f)[0] + ".jpg")
        if os.path.exists(dst):
            continue
        img = cv2.imdecode(np.fromfile(os.path.join(IMG_DIR, f), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        s = 1500 / max(h, w)
        if s < 1.0:
            img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        if save(dst, img):
            made += 1
    json.dump(pick, open(os.path.join(OUT_DIR, "holdout_files.json"), "w",
                         encoding="utf-8"), ensure_ascii=False, indent=0)
    print(f"홀드아웃 {len(pick)}장 선정 (축소본 {made}장 신규 생성)")

    # 몽타주 생성
    py = sys.executable
    script = os.path.join(ROOT, "label", "montage.py")
    plan = []
    for i in range(0, len(pick), 4):
        grp = pick[i:i + 4]
        name = "h%02d.jpg" % (i // 4 + 1)
        subprocess.run([py, script, "--files", ",".join(grp), "--out", name],
                       check=True, capture_output=True)
        plan.append({"montage": name, "files": grp})
    json.dump(plan, open(os.path.join(OUT_DIR, "holdout_montage.json"), "w",
                         encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"몽타주 {len(plan)}개")
    for p in plan:
        print("  %s: %s" % (p["montage"],
                            ", ".join(os.path.splitext(f)[0] for f in p["files"])))


if __name__ == "__main__":
    main()
