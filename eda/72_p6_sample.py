"""
EDA 72단계 — P6(날짜 없음) 실패 이미지를 사람이 직접 보기 위한 표본 추출

69·71단계에서 두 번 연속으로 전제가 뒤집혔다. 둘 다 이미지를 보지 않고
수치만으로 추론한 것이 원인이었다. 라벨링 예산을 걸기 전에 눈으로 확인한다.

확인할 질문은 하나다.
    **P6 이미지에 소비기한이 실제로 인쇄되어 있는가?**
  - 인쇄돼 있는데 OCR 이 못 읽었다  → 라벨링해서 검출기를 학습시킬 값이 있다
  - 애초에 안 찍혀 있거나 판독 불가  → 라벨링해도 배울 게 없다. 예산을 딴 데 쓴다

실패 유형과 해상도로 층화해 뽑는다. 저해상도가 실패의 33.6%를 차지하므로
그 층을 따로 봐야 한다.

산출물
    실패샘플_P6/        읽을 수 있는 크기(긴 변 1400px)로 저장, 파일명에 유형 표기
    실패샘플_P6/_요약.csv  판정을 적어 넣을 수 있는 표

    python eda/72_p6_sample.py --n 40
"""
import os
import sys
import json
import random
import argparse
import warnings
from collections import Counter, defaultdict

warnings.filterwarnings("ignore")

import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(ROOT, "images", "상품사진입니다")
OUT_DIR = os.path.join(ROOT, "eda", "out")
DST = os.path.join(ROOT, "실패샘플_P6")
SEED = 42

KO = {"no_text": "글자못찾음", "text_no_digits": "숫자줄없음",
      "digits_no_date": "숫자있음날짜없음"}


def imread(path):
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)


def imwrite(path, img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if ok:
        buf.tofile(path)          # cv2.imwrite 는 한글 경로에서 조용히 실패한다
    return ok


def label_bar(img, lines):
    """이미지 위에 한글 설명 띠를 얹는다. cv2.putText 는 한글을 못 쓰므로 PIL 사용."""
    from PIL import Image, ImageDraw, ImageFont
    h, w = img.shape[:2]
    bar = 26 * len(lines) + 12
    canvas = np.full((h + bar, w, 3), 255, np.uint8)
    canvas[bar:] = img
    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 19)
    except Exception:
        font = ImageFont.load_default()
    for i, t in enumerate(lines):
        d.text((10, 6 + i * 26), t, fill=(20, 20, 20), font=font)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--side", type=int, default=1400)
    a = ap.parse_args()

    inv = {r["file"]: r for r in
           json.load(open(os.path.join(OUT_DIR, "69_failures.json"), encoding="utf-8"))}
    diag = {r["file"]: r for r in
            json.load(open(os.path.join(OUT_DIR, "71_failure_texts.json"),
                           encoding="utf-8"))}

    p6 = [f for f, r in diag.items() if r["probe"] == "P6_날짜없음"]
    print(f"P6 모집단 {len(p6)}장")

    # 층: (69단계 원인) × (저해상도 여부)
    strata = defaultdict(list)
    for f in p6:
        r = inv[f]
        lo = max(r["w"], r["h"]) <= 800
        strata[(r["cause"], lo)].append(f)

    rng = random.Random(SEED)
    # 각 층에서 비례 배분하되, 작은 층도 최소 2장은 본다
    total = len(p6)
    picked = []
    for key, fs in sorted(strata.items(), key=lambda kv: -len(kv[1])):
        want = max(2, round(a.n * len(fs) / total))
        want = min(want, len(fs))
        picked += [(f, key) for f in rng.sample(fs, want)]
    picked = picked[:a.n]

    if os.path.isdir(DST):
        for f in os.listdir(DST):
            try:
                os.remove(os.path.join(DST, f))
            except OSError:
                pass
    os.makedirs(DST, exist_ok=True)

    import csv
    rowsout = []
    for i, (f, (cause, lo)) in enumerate(sorted(picked, key=lambda x: (x[1], x[0])), 1):
        img = imread(os.path.join(IMG_DIR, f))
        if img is None:
            continue
        h, w = img.shape[:2]
        s = a.side / max(h, w)
        if s < 1.0:
            img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        d = diag[f]
        frags = [t for t in d["texts"] if any(c.isdigit() for c in t)][:6]
        lines = [
            f"{i:02d}  {f}   {w}x{h}" + ("  [저해상도]" if lo else ""),
            f"유형: {KO.get(cause, cause)}   OCR 조각 {d['n_frag']}개"
            + ("   소비기한 키워드 있음" if d["has_anchor"] else ""),
            "숫자 조각: " + (" / ".join(frags)[:90] if frags else "없음"),
        ]
        img = label_bar(img, lines)
        name = f"{i:02d}_[{KO.get(cause,cause)}]{'_저해상도' if lo else ''}_{f}"
        imwrite(os.path.join(DST, name), img)
        rowsout.append({"no": i, "file": f, "유형": KO.get(cause, cause),
                        "해상도": f"{w}x{h}", "저해상도": "Y" if lo else "",
                        "OCR조각수": d["n_frag"],
                        "키워드": "Y" if d["has_anchor"] else "",
                        "판정(직접 적기)": "", "메모": ""})

    with open(os.path.join(DST, "_요약.csv"), "w", encoding="utf-8-sig",
              newline="") as fo:
        w_ = csv.DictWriter(fo, fieldnames=list(rowsout[0].keys()))
        w_.writeheader()
        w_.writerows(rowsout)

    cnt = Counter(r["유형"] for r in rowsout)
    lo_n = sum(1 for r in rowsout if r["저해상도"] == "Y")
    kw_n = sum(1 for r in rowsout if r["키워드"] == "Y")
    print(f"\n→ {DST}")
    print(f"  {len(rowsout)}장 저장 (긴 변 {a.side}px, 설명 띠 포함)")
    for k, v in cnt.most_common():
        print(f"    {k}: {v}장")
    print(f"    저해상도 {lo_n}장 / 소비기한 키워드가 읽힌 것 {kw_n}장")
    print("\n  _요약.csv 의 '판정' 칸에 아래 중 하나를 적어 주세요:")
    print("    인쇄됨   — 날짜가 보이는데 OCR 이 못 읽음  (라벨링 값 있음)")
    print("    없음     — 사진에 소비기한이 안 찍혀 있음   (라벨링 무의미)")
    print("    판독불가 — 찍혔지만 사람도 못 읽음          (라벨링 무의미)")


if __name__ == "__main__":
    main()
