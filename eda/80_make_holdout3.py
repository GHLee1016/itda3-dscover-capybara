"""
EDA 80단계 — 본선용 새 홀드아웃(홀드아웃3) 표본 추출

왜 필요한가
  홀드아웃2 198장은 1차 예선에서 설정 선택에 여러 번 쓰였고, 이번 라운드에서도
  76·78·79단계에 썼다. 특히 79단계에서 **오답 17장을 직접 들여다보고 원인을
  찾았다.** 그 지식으로 규칙을 고친 뒤 같은 표본으로 재면, 그 17장에 맞춘 정도가
  성능으로 보인다. 이제 홀드아웃2 로는 개선폭을 주장할 수 없다.

두 집합을 따로 만든다. 섞으면 대표성이 깨진다.
  A. 대표 표본 150장  — 무작위. **전체 성능 수치는 여기서만 낸다**
  B. 표적 표본  50장  — 2자리 연도가 검출된 이미지. 이번 수정의 효과를 보는 용도.
                        난이도가 편향돼 있으므로 **헤드라인 수치에 쓰지 않는다**

제외: 홀드아웃2(198) + 튜닝셋(137) + 1차 홀드아웃(48) = 이미 라벨링한 383장

⚠️ 선발에 정답을 쓰지 않는다. B 집합의 '2자리 연도' 판정은 파이프라인 출력으로만
   한다(문서 §3.5 정보 누수 차단).

    python eda/80_make_holdout3.py --n 150 --target 50
"""
import os
import re
import sys
import json
import argparse
import warnings
from collections import Counter

warnings.filterwarnings("ignore")

import numpy as np
import cv2

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(ROOT, "images", "상품사진입니다")
OUT_DIR = os.path.join(ROOT, "eda", "out")
TOOL = os.path.join(ROOT, "custom_data", "tool")
IMGS = os.path.join(TOOL, "imgs")
SEED = 20260925

# 2자리 연도로 읽힌 흔적: 연도가 20xx 인데 일이 20~35 범위라 뒤집힐 수 있는 경우
AMBIG = re.compile(r"^(20[2-3]\d)-(\d{2})-(\d{2})$")


def save_jpg(path, img, q=88):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    if ok:
        buf.tofile(path)          # cv2.imwrite 는 한글 경로에서 조용히 실패한다
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150, help="대표 표본")
    ap.add_argument("--target", type=int, default=50, help="2자리 연도 표적 표본")
    ap.add_argument("--long-side", type=int, default=1500)
    a = ap.parse_args()

    # 이미 라벨링한 것 제외
    used = set()
    for fn in ("truth_v4.json", "truth_holdout.json", "truth_holdout2.json"):
        p = os.path.join(OUT_DIR, fn)
        if os.path.exists(p):
            used |= set(json.load(open(p, encoding="utf-8"))["labels"])

    EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
    allf = sorted(f for f in os.listdir(IMG_DIR)
                  if f.lower().endswith(EXTS) and not f.startswith("."))
    pool = [f for f in allf if f not in used]
    print(f"전체 {len(allf)}장 − 라벨 완료 {len(used)}장 → 후보 {len(pool)}장")

    # 69단계 전량 실행 결과를 재사용해 '2자리 연도로 뒤집힐 수 있는' 것을 찾는다.
    # 정답이 아니라 **파이프라인 출력**만 본다.
    amb = set()
    p69 = os.path.join(OUT_DIR, "69_failures.json")
    if os.path.exists(p69):
        for r in json.load(open(p69, encoding="utf-8")):
            if r["stage"] == "none":
                continue
            y, m, d = r.get("year"), r.get("month"), r.get("day")
            if y and y != "NONE" and d and d != "NONE":
                try:
                    yy, dd = int(y) % 100, int(d)
                except ValueError:
                    continue
                # 연도 뒷 두 자리와 일이 둘 다 20~35 면 YY.MM.DD / DD.MM.YY 가 모두 성립
                if 20 <= yy <= 35 and 20 <= dd <= 35:
                    amb.add(r["file"])
    amb &= set(pool)
    print(f"2자리 연도로 뒤집힐 수 있는 후보 {len(amb)}장")

    rng = np.random.default_rng(SEED)

    # A. 대표 표본 — 순수 무작위
    main_pool = sorted(pool)
    idx = rng.permutation(len(main_pool))
    rep = [main_pool[i] for i in idx[:a.n]]

    # B. 표적 표본 — 대표 표본과 겹치지 않게
    tgt_pool = sorted(amb - set(rep))
    if tgt_pool:
        j = rng.permutation(len(tgt_pool))
        tgt = [tgt_pool[i] for i in j[:a.target]]
    else:
        tgt = []

    tasks = ([{"file": f, "extra": False, "prio": 1} for f in rep] +
             [{"file": f, "extra": True, "prio": 0} for f in tgt])
    print(f"대표 {len(rep)}장 + 표적 {len(tgt)}장 = {len(tasks)}장")

    # 축소 이미지 생성
    os.makedirs(IMGS, exist_ok=True)
    made = 0
    for t in tasks:
        dst = os.path.join(IMGS, os.path.splitext(t["file"])[0] + ".jpg")
        if os.path.exists(dst):
            made += 1
            continue
        img = cv2.imdecode(np.fromfile(os.path.join(IMG_DIR, t["file"]), np.uint8),
                           cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        s = a.long_side / max(h, w)
        if s < 1.0:
            img = cv2.resize(img, (int(w * s), int(h * s)),
                             interpolation=cv2.INTER_AREA)
        made += bool(save_jpg(dst, img))
    print(f"축소 이미지 {made}장 준비 → {IMGS}")

    with open(os.path.join(TOOL, "tasks.json"), "w", encoding="utf-8") as f:
        json.dump({"shard": 1, "of": 1, "tasks": tasks}, f,
                  ensure_ascii=False, indent=1)

    # 명단을 따로 남겨 나중에 대표/표적을 갈라 집계할 수 있게 한다
    json.dump({"seed": SEED, "representative": rep, "targeted": tgt},
              open(os.path.join(OUT_DIR, "80_holdout3_split.json"), "w",
                   encoding="utf-8"), ensure_ascii=False, indent=1)

    print()
    print("=== 라벨링 방법")
    print("  cd custom_data/tool && python -m http.server 8000")
    print("  → http://localhost:8000/index.html")
    print()
    print("  숫자키로 후보 선택, 0 은 NONE, ← → 이동, R 90° 회전")
    print("  다 하면 우측 상단 'JSON 내려받기' → truth_labeled.json")
    print("  받은 파일을 eda/out/truth_holdout3.json 으로 옮기면 된다")
    print()
    print("=== 라벨링 기준 (기존과 동일하게 유지할 것)")
    print("  · 소비기한과 제조일자가 함께 있으면 늦은 쪽")
    print("  · 일부만 판독되면 그 필드만 채우고 나머지는 NONE")
    print("  · 판독 불가·중의적이면 건너뛴다 (억지로 채우지 않는다)")
    print()
    print(f"  대표 {len(rep)}장이 성능 수치의 근거가 된다. 표적 {len(tgt)}장은")
    print("  2자리 연도 수정의 효과를 보는 용도이므로 헤드라인 수치에 넣지 않는다.")


if __name__ == "__main__":
    main()
