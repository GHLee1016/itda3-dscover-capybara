"""
EDA 86단계 — 미검출 7장이 어디서 막혔는지

85단계에서 대표 147장의 오답 17장 중 7장(4.8%)이 미검출이었다. 가장 큰 덩어리인데
원인을 모른다. 69단계는 "텍스트를 하나라도 읽었나"로만 나눠서, 영양성분표 글자가
날짜 구역 실패를 가려 버렸고 그 때문에 방향을 두 번 잘못 잡았다.

여기서는 세 지점을 분리해서 본다.
    ① OCR 이 조각을 하나도 못 냈나
    ② 조각은 냈는데 숫자가 4개 이상인 조각이 없나   ← 날짜 구역 자체를 놓친 것
    ③ 숫자 조각은 있는데 규칙이 날짜를 못 만들었나   ← 규칙 문제
②면 검출기, ③이면 규칙이다. 고칠 곳이 완전히 다르다.

    python eda/86_miss_cause.py
"""
import os
import re
import sys
import json
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

IMG_DIR = os.path.join(ROOT, "images", "상품사진입니다")
OUT = os.path.join(ROOT, "eda", "out")
RAPID_DIR = os.path.join(ROOT, "weights", "rapidocr")


def main():
    import combined
    import team2_rules as R

    lab = json.load(open(os.path.join(OUT, "truth_holdout3.json"),
                        encoding="utf-8"))["labels"]
    pred = json.load(open(os.path.join(OUT, "83_holdout3.json"),
                          encoding="utf-8"))["C_최종"]

    miss = [f for f in lab
            if pred.get(f) == "NONE-NONE-NONE" and lab[f] != "NONE-NONE-NONE"]
    print(f"미검출 {len(miss)}장 추적\n", flush=True)

    rapid, _ = combined.build_rapid_ocr(RAPID_DIR)
    retry, _ = combined.build_rapid_retry(RAPID_DIR)

    # ⚠️ 처음에는 "숫자가 4개 이상인 조각"이 있으면 규칙 문제로 쳤다. 그러면
    #    영양성분표(`85mL×67H(510mL/510kcal)`)나 바코드가 날짜로 잡혀서
    #    7장 전부가 규칙 문제로 나왔다. 69단계와 같은 실수다.
    #    여기서는 **날짜처럼 생긴 조각**만 센다.
    DATEISH = re.compile(
        r"(?<!\d)(?:20\d{2}|\d{1,2})\s*[.\-/:]\s*\d{1,2}\s*[.\-/:]\s*\d{1,4}"
        r"|(?<!\d)\d{1,2}\s*(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)"
        r"|(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s*\.?\s*\d{1,2}",
        re.I)

    def texts_of(engine, arr):
        try:
            r = engine(arr)
            return list(r.txts) if getattr(r, "txts", None) else []
        except Exception:
            return []

    L = ["# EDA 86 — 미검출 7장이 어디서 막혔는지\n",
         "69단계의 분류가 거칠어 두 번 방향을 잘못 잡았다. 여기서는",
         "**조각 없음 / 숫자 조각 없음 / 규칙 실패** 를 분리한다.",
         "앞의 둘은 검출기 문제, 마지막은 규칙 문제다.",
         ""]

    tally = {"조각 없음": 0, "날짜꼴 조각 없음": 0, "규칙 실패": 0}

    for f in miss:
        img = combined.imread_any(os.path.join(IMG_DIR, f))
        if img is None:
            L.append(f"### `{f}` — 이미지를 못 읽음\n")
            continue
        small = combined.downscale_max(img, 1024)

        rows = []
        for name, eng in (("주력 det small", rapid), ("재시도 det medium", retry)):
            if eng is None:
                continue
            txt = texts_of(eng, small)
            dig = [t for t in txt if sum(c.isdigit() for c in t) >= 4]
            hit = [t for t in txt if DATEISH.search(t)]
            got = R.choose_expiration_date(txt) if txt else None
            rows.append((name, txt, dig, hit, got))

        # 어느 단계든 날짜꼴 조각을 냈으면 그걸로 판정한다
        best = max(rows, key=lambda r: (len(r[3]), len(r[2]), len(r[1])))
        if not best[1]:
            why = "조각 없음"
        elif not best[3]:
            why = "날짜꼴 조각 없음"
        else:
            why = "규칙 실패"
        tally[why] += 1

        L.append(f"### `{f}` — 정답 {lab[f]} · **{why}**\n")
        for name, txt, dig, hit, got in rows:
            L.append(f"- {name}: 조각 {len(txt)}개 / 숫자 조각 {len(dig)}개 "
                     f"/ **날짜꼴 {len(hit)}개** / 규칙 결과 `{got}`")
            if hit:
                L.append(f"  - 날짜꼴: `{' | '.join(hit[:6])[:200]}`")
            if dig:
                L.append(f"  - 숫자 조각: `{' | '.join(dig[:8])[:200]}`")
        L.append("")
        print(f"  {f}  {why}", flush=True)

    L.append("## 집계\n")
    L.append("| 막힌 지점 | 장수 | 고칠 곳 |")
    L.append("| --- | ---: | --- |")
    KO = {"조각 없음": "검출기",
          "날짜꼴 조각 없음": "검출기 또는 인식기(문자 오독)",
          "규칙 실패": "규칙(team2_rules)"}
    for k, v in tally.items():
        L.append(f"| {k} | {v} | {KO[k]} |")
    L.append("")
    det = tally["조각 없음"] + tally["날짜꼴 조각 없음"]
    L.append(f"OCR {det}장 / 규칙 {tally['규칙 실패']}장")
    L.append("")
    L.append("> 「날짜꼴 조각 없음」은 검출을 놓친 경우와 읽었지만 글자를 틀리게")
    L.append("> 읽어 날짜 모양이 깨진 경우가 섞여 있다. 위의 조각 목록으로 구분한다.")

    with open(os.path.join(OUT, "86_miss_cause.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n" + "\n".join(L[-8:]))
    print("\n→ eda/out/86_miss_cause.md")


if __name__ == "__main__":
    main()
