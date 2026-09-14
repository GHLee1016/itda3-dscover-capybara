"""
EDA 43단계 — 오답 원본 사진을 유형별 폴더로 모은다

사람이 직접 넘겨보며 확인할 수 있도록, 오답 이미지를 유형별 하위 폴더에
**원본 그대로** 복사한다. 파일명 앞에 정답과 출력을 붙여 탐색기에서
바로 알아볼 수 있게 한다.

    오답사진/
      1_미검출_인식불가/   [GT]2021-07-01__[예측]NONE__000876.jpg
      2_완전히다름/
      3_미검출_파싱실패/
      4_월일오독/
      5_연도오독/
      _오답목록.csv
"""
import os
import re
import shutil
import argparse

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "eda", "out")
IMG_DIR = os.path.join(ROOT, "images", "상품사진입니다")
DST_ROOT = os.path.join(ROOT, "오답사진")

FOLDER = {
    "미검출-못읽음": "1_미검출_인식불가",
    "완전히 다름": "2_완전히다름",
    "미검출-읽긴읽음": "3_미검출_파싱실패",
    "월/일 오독": "4_월일오독",
    "연도만 오독": "5_연도오독",
}


def safe(s):
    return re.sub(r'[\\/:*?"<>|]', "-", str(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(OUT_DIR, "holdout2_errors.csv"))
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    if os.path.isdir(DST_ROOT):
        shutil.rmtree(DST_ROOT)
    os.makedirs(DST_ROOT, exist_ok=True)

    n = 0
    for _, r in df.iterrows():
        sub = FOLDER.get(r["kind"], "9_기타")
        d = os.path.join(DST_ROOT, sub)
        os.makedirs(d, exist_ok=True)
        src = os.path.join(IMG_DIR, r["file"])
        if not os.path.exists(src):
            continue
        stem, ext = os.path.splitext(r["file"])
        name = "[정답]%s__[출력]%s__%s%s" % (safe(r["truth"]), safe(r["pred"]), stem, ext)
        shutil.copy2(src, os.path.join(d, name))
        n += 1

    # 목록 CSV 도 함께 (엑셀로 열어볼 수 있게)
    cols = [c for c in ("file", "truth", "pred", "kind", "anchors", "cands")
            if c in df.columns]
    df[cols].sort_values(["kind", "file"]).to_csv(
        os.path.join(DST_ROOT, "_오답목록.csv"), index=False, encoding="utf-8-sig")

    print(f"오답 원본 {n}장 복사 → {DST_ROOT}\n")
    for sub in sorted(os.listdir(DST_ROOT)):
        p = os.path.join(DST_ROOT, sub)
        if os.path.isdir(p):
            print("  %-22s %d장" % (sub, len(os.listdir(p))))
    print("\n파일명 형식: [정답]YYYY-MM-DD__[출력]YYYY-MM-DD__원본파일명")


if __name__ == "__main__":
    main()
