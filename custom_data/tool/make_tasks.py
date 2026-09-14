"""
라벨링 과제 생성기

이미 돌려둔 OCR 결과(eda/out/ocr_sample.jsonl)를 재활용해
 - 라벨링용 축소 이미지 (label/imgs/)
 - 후보 날짜가 채워진 과제 파일 (label/tasks.json)
를 만든다. 라벨러는 대부분 버튼만 누르면 되고, OCR 이 놓친 것만 직접 입력한다.

사용:
    python label/make_tasks.py --n 300 --shards 3
    → label/index.html 을 브라우저로 열어 라벨링
"""
import os
import sys
import json
import argparse
from collections import Counter

import numpy as np
import pandas as pd
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "eda"))
from datex import find_dates, find_anchors, POSITIVE, NEGATIVE

IMG_DIR = os.path.join(ROOT, "images", "상품사진입니다")
EDA_OUT = os.path.join(ROOT, "eda", "out")
LAB = os.path.join(ROOT, "label")
IMGS = os.path.join(LAB, "imgs")


def save_jpg(path, img, q=88):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    if ok:
        buf.tofile(path)          # 한글 경로 대응 (cv2.imwrite 는 실패한다)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=250, help="대표 표본 (정확도 측정용)")
    ap.add_argument("--hard", type=int, default=50, help="어려운 케이스 추가 표본 (디버깅용, 통계 제외)")
    ap.add_argument("--shards", type=int, default=1, help="팀원 수만큼 분할")
    ap.add_argument("--long-side", type=int, default=1500)
    a = ap.parse_args()

    os.makedirs(IMGS, exist_ok=True)

    # --- OCR 결과에서 후보 날짜 생성 ---
    jl = os.path.join(EDA_OUT, "ocr_sample.jsonl")
    recs = [json.loads(l) for l in open(jl, encoding="utf-8")]
    recs = [r for r in recs if r.get("ok")]
    print(f"OCR 결과 {len(recs)}장 로드")

    meta = pd.read_csv(os.path.join(EDA_OUT, "metadata.csv")).set_index("file")

    tasks = []
    for r in recs:
        lines = r["lines_0"] + r["lines_90"]
        text = "\n".join(l["t"] for l in lines)
        anc = find_anchors(text)
        seen, cands = set(), []
        for d in find_dates(text):
            if d["in_long_digits"] or d["day_missing"]:
                continue
            key = (d["y"], d["m"], d["d"])
            if key in seen:
                continue
            seen.add(key)
            cands.append({
                "date": "%04d-%02d-%02d" % key,
                "raw": d["raw"].strip()[:24],
                "kind": d["kind"],
            })
        # 후보가 많으면 연도 타당한 것 우선
        cands.sort(key=lambda c: (not (2015 <= int(c["date"][:4]) <= 2035), c["date"]))
        m = meta.loc[r["file"]] if r["file"] in meta.index else None
        tasks.append({
            "file": r["file"],
            "image_id": r["image_id"],
            "cands": cands[:8],
            "anchors": sorted(anc),
            "pos": bool(anc & set(POSITIVE)),
            "neg": bool(anc & set(NEGATIVE)),
            "wh": [int(m["w"]), int(m["h"])] if m is not None else None,
            # 난이도: 후보 0개(=OCR 실패) 와 후보 2개+(=선택 필요) 를 먼저 라벨링
            "prio": 0 if len(cands) >= 2 else (1 if len(cands) == 0 else 2),
        })

    # --- 표본 선정 ---
    # 정확도를 정직하게 재려면 대표 표본이어야 한다. 난이도순으로 뽑으면
    # 어려운 케이스만 모여 실제 성능보다 나쁘게 측정된다.
    rng = np.random.default_rng(20260915)
    idx = rng.permutation(len(tasks))
    main_set = [tasks[k] for k in idx[:a.n]]
    for t in main_set:
        t["extra"] = False          # 정확도 측정에 포함

    # 디버깅용으로 어려운 케이스를 추가 표집 (측정에서는 제외)
    chosen = {t["file"] for t in main_set}
    hard_pool = [t for t in tasks if t["file"] not in chosen and t["prio"] == 0]
    extra = hard_pool[:a.hard]
    for t in extra:
        t["extra"] = True           # 난이도 편향이므로 통계에서 뺀다

    tasks = main_set + extra
    print(f"대표표본 {len(main_set)}장 + 난이도 추가표본 {len(extra)}장")

    # --- 축소 이미지 생성 ---
    print(f"이미지 {len(tasks)}장 축소 중 ...")
    made = 0
    for t in tasks:
        dst = os.path.join(IMGS, os.path.splitext(t["file"])[0] + ".jpg")
        if os.path.exists(dst):
            made += 1
            continue
        src = os.path.join(IMG_DIR, t["file"])
        img = cv2.imdecode(np.fromfile(src, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        s = a.long_side / max(h, w)
        if s < 1.0:
            img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        if save_jpg(dst, img):
            made += 1
    print(f"  → {made}장 준비 ({IMGS})")

    # --- 샤드 분할 ---
    for i in range(a.shards):
        shard = tasks[i::a.shards]
        name = "tasks.json" if a.shards == 1 else f"tasks_{i+1}.json"
        with open(os.path.join(LAB, name), "w", encoding="utf-8") as f:
            json.dump({"shard": i + 1, "of": a.shards, "tasks": shard},
                      f, ensure_ascii=False, indent=1)
        print(f"  → {name}: {len(shard)}장")

    c = Counter(t["prio"] for t in tasks if not t["extra"])
    print(f"\n대표표본 난이도 분포: 후보2개+ {c[0]}장 / 후보0개 {c[1]}장 / 후보1개 {c[2]}장")
    print("→ 이 분포가 전체 데이터의 실제 분포를 반영해야 정확도 측정이 유효하다.")


if __name__ == "__main__":
    main()
