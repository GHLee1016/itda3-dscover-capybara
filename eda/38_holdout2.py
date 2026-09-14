"""
EDA 38단계 — 2차 홀드아웃 200장 생성

1차 홀드아웃 48장은 연도 후행형 버그를 찾아 고치는 데 썼다. 그 순간부터
튜닝셋이 되었으므로 더 이상 과적합 검증에 쓸 수 없다.

설계·디버깅에 한 번도 쓰지 않은 200장을 새로 뽑는다.
표본 200장이면 95% 신뢰구간이 ±7%p 로 좁아져, 다음 개선(±5%p 수준)을
유의하게 검증할 수 있다.
"""
import os
import sys
import json
import argparse
import subprocess

import numpy as np
import pandas as pd
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "eda", "out")
IMG_DIR = os.path.join(ROOT, "images", "상품사진입니다")
DST = os.path.join(ROOT, "label", "imgs")


def save(path, img, q=88):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    if ok:
        buf.tofile(path)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()

    used = set()
    for fn in ("truth_v4.json", "truth_holdout.json"):
        p = os.path.join(OUT_DIR, fn)
        if os.path.exists(p):
            d = json.load(open(p, encoding="utf-8"))
            used |= set(d.get("labels", {}))
            used |= set(d.get("ambiguous", {}))
    ap_csv = os.path.join(OUT_DIR, "active_pool.csv")
    if os.path.exists(ap_csv):
        used |= set(pd.read_csv(ap_csv)["file"])

    meta = pd.read_csv(os.path.join(OUT_DIR, "metadata.csv"))
    fresh = meta[~meta["file"].isin(used)]
    print(f"후보 풀 {len(fresh)}장 (전체 {len(meta)} − 사용됨 {len(used)})")

    pick = fresh.sample(min(a.n, len(fresh)), random_state=20260913)["file"].tolist()

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
    json.dump(pick, open(os.path.join(OUT_DIR, "holdout2_files.json"), "w",
                         encoding="utf-8"), ensure_ascii=False, indent=0)
    print(f"2차 홀드아웃 {len(pick)}장 (축소본 {made}장 생성)")

    py = sys.executable
    script = os.path.join(ROOT, "label", "montage.py")
    plan = []
    for i in range(0, len(pick), 4):
        grp = pick[i:i + 4]
        name = "g%02d.jpg" % (i // 4 + 1)
        subprocess.run([py, script, "--files", ",".join(grp), "--out", name],
                       check=True, capture_output=True)
        plan.append({"montage": name, "files": grp})
    json.dump(plan, open(os.path.join(OUT_DIR, "holdout2_montage.json"), "w",
                         encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"몽타주 {len(plan)}개 (g01~g{len(plan):02d})")
    for p in plan:
        print("  %s: %s" % (p["montage"],
                            ", ".join(os.path.splitext(f)[0] for f in p["files"])))


if __name__ == "__main__":
    main()
