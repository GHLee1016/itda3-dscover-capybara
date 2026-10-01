"""
EDA 71단계 — 실패 271장이 '무엇을 읽고 왜 버렸는가' (능동학습 방향 재설정)

69단계 결과가 전제를 뒤집었다.
    글자를 못 찾음        10장 (3.7%)   ← 검출 문제. YOLO 재학습의 표적
    숫자 줄이 없음        29장 (10.7%)
    숫자는 있으나 날짜 아님 232장 (85.6%)  ← 여기가 본체

박스 라벨링은 앞의 39장만 겨냥한다. 232장은 OCR 이 이미 숫자를 읽어 놓은 상태라
사람이 박스를 그릴 필요가 없다. 대신 **왜 날짜로 인정되지 않았는지**를 갈라야 한다.

진단 방식: 실패 이미지의 OCR 텍스트에 점점 느슨한 탐침을 걸어, 어느 단계까지
완화하면 날짜가 잡히는지 본다. 가장 약한 탐침에서 잡히면 그만큼 싼 수정이다.

    P1 4자리 연도 + 구분자      2026.03.15
    P2 4자리 연도 무구분자       20260315
    P3 2자리 연도 + 구분자       26.03.15
    P4 OCR 혼동 문자 정규화 후    2O26.O3.I5 → 2026.03.15
    P5 월/일만                  03.15
    P6 어느 탐침에도 안 걸림      날짜가 실제로 없거나 심하게 깨짐

⚠️ 라벨은 '진단'에만 쓴다. 여기서 얻은 규칙 수정안은 반드시 홀드아웃2(동결)로
   따로 검증해야 한다. 진단에 쓴 이미지로 성능을 재면 의미가 없다.

    python eda/71_failure_text_diag.py --workers 4
"""
import os
import re
import sys
import json
import time
import argparse
import warnings
from collections import Counter

warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "eda"))

IMG_DIR = os.path.join(ROOT, "images", "상품사진입니다")
OUT_DIR = os.path.join(ROOT, "eda", "out")
RAPID_DIR = os.path.join(ROOT, "weights", "rapidocr")

# OCR 이 자주 혼동하는 문자. 숫자 문맥에서만 되돌린다.
CONFUSE = str.maketrans({"O": "0", "o": "0", "D": "0", "Q": "0",
                         "I": "1", "l": "1", "i": "1", "|": "1",
                         "Z": "2", "z": "2", "S": "5", "s": "5",
                         "B": "8", "G": "6", "b": "6", "T": "7"})

SEP = r"[.\-/\s,:]"
P1 = re.compile(r"(?<!\d)(20[0-4]\d)\s*" + SEP + r"\s*(0?[1-9]|1[0-2])\s*" + SEP +
                r"\s*(0?[1-9]|[12]\d|3[01])(?!\d)")
P2 = re.compile(r"(?<!\d)(20[0-4]\d)(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(?!\d)")
P3 = re.compile(r"(?<!\d)([0-4]\d)\s*" + SEP + r"\s*(0?[1-9]|1[0-2])\s*" + SEP +
                r"\s*(0?[1-9]|[12]\d|3[01])(?!\d)")
P5 = re.compile(r"(?<!\d)(0?[1-9]|1[0-2])\s*" + SEP + r"\s*(0?[1-9]|[12]\d|3[01])(?!\d)")

ANCHOR = ("소비기한", "유통기한", "품질유지기한", "까지", "EXP", "exp", "BB",
          "BEST BEFORE", "Best before", "USE BY")
NEG = ("제조", "생산", "포장", "MFG", "PROD", "LOT", "로트")

_S = None


def setup():
    global _S
    if _S is None:
        import combined
        rapid, _ = combined.build_rapid_ocr(RAPID_DIR)
        _S = (combined, rapid)
    return _S


def probe(texts):
    """어느 탐침에서 처음 잡히는지. 가장 싼 수정이 무엇인지를 말해 준다."""
    blob = " ".join(texts)
    joined = blob
    # 인접 조각 이어붙인 형태도 함께 본다 (팀원2 규칙과 같은 착상)
    for i in range(len(texts) - 1):
        joined += " | " + texts[i] + texts[i + 1]

    if P1.search(joined):
        return "P1_4자리_구분자"
    if P2.search(joined):
        return "P2_4자리_무구분자"
    if P3.search(joined):
        return "P3_2자리_연도"
    fixed = joined.translate(CONFUSE)
    if P1.search(fixed) or P2.search(fixed) or P3.search(fixed):
        return "P4_문자혼동_정규화"
    if P5.search(joined):
        return "P5_월일만"
    return "P6_날짜없음"


def work(fname):
    combined, rapid = setup()
    img = combined.imread_any(os.path.join(IMG_DIR, fname))
    if img is None:
        return None
    small = combined.downscale_max(img, 1024)
    try:
        r = rapid(small)
        texts = list(r.txts) if getattr(r, "txts", None) else []
    except Exception:
        texts = []
    blob = " ".join(texts)
    return {
        "file": fname,
        "probe": probe(texts),
        "n_frag": len(texts),
        "has_anchor": any(k in blob for k in ANCHOR),
        "has_neg": any(k in blob for k in NEG),
        "texts": texts[:40],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    rows = json.load(open(os.path.join(OUT_DIR, "69_failures.json"), encoding="utf-8"))
    fails = [r for r in rows if r["stage"] == "none" and r["cause"] != "load_fail"]
    cause = {r["file"]: r["cause"] for r in fails}
    files = [r["file"] for r in fails]
    print(f"실패 {len(files)}장 진단", flush=True)

    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    res = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(work, files, chunksize=4), 1):
            if r:
                res.append(r)
            if i % 100 == 0:
                print(f"  {i}/{len(files)}  {time.time()-t0:.0f}s", flush=True)

    json.dump(res, open(os.path.join(OUT_DIR, "71_failure_texts.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)

    # 라벨이 있는 실패는 진단 품질 확인에 쓴다 (선발 아님)
    lab = {}
    for fn in ("truth_v4.json", "truth_holdout.json"):
        p = os.path.join(OUT_DIR, fn)
        if os.path.exists(p):
            lab.update(json.load(open(p, encoding="utf-8"))["labels"])

    pc = Counter(r["probe"] for r in res)
    n = len(res)
    anchor_n = sum(1 for r in res if r["has_anchor"])

    DESC = {
        "P1_4자리_구분자": ("`2026.03.15` 형태가 실제로 있다", "**규칙 버그**. 가장 싼 수정"),
        "P2_4자리_무구분자": ("`20260315` 처럼 구분자가 소실됐다", "정규식 추가로 회수 가능"),
        "P3_2자리_연도": ("`26.03.15` 2자리 연도다", "연도 판별 규칙 보강"),
        "P4_문자혼동_정규화": ("`2O26.O3.I5` 처럼 문자로 오독됐다", "숫자 문맥 문자 정규화"),
        "P5_월일만": ("월/일만 있고 연도가 없다", "부분 출력으로 회수 (연=NONE)"),
        "P6_날짜없음": ("어떤 탐침에도 안 걸린다", "인식 품질 또는 검출 문제. 라벨링 대상"),
    }

    L = ["# EDA 71 — 실패 271장이 무엇을 읽고 왜 버렸는가\n",
         "69단계가 전제를 뒤집었다. 실패의 85.6%는 검출 실패가 아니라",
         "**숫자는 읽었는데 날짜로 인정되지 않은** 경우였다. 박스 라벨링으로는",
         "39장(14.4%)밖에 겨냥하지 못한다. 그래서 232장이 왜 버려졌는지 먼저 갈랐다.",
         "",
         "진단은 점점 느슨한 탐침을 걸어 어느 단계에서 날짜가 잡히는지 보는 방식이다.",
         "약한 탐침에서 잡힐수록 싼 수정으로 회수할 수 있다는 뜻이다.",
         "",
         f"- 대상 {n}장, {time.time()-t0:.0f}초",
         f"- 소비기한 계열 키워드가 함께 읽힌 이미지 {anchor_n}장 ({anchor_n/n*100:.1f}%)",
         "",
         "## 어느 탐침에서 잡히는가\n",
         "| 탐침 | 장수 | 비율 | 의미 | 대응 |",
         "| --- | ---: | ---: | --- | --- |"]
    for k in ("P1_4자리_구분자", "P2_4자리_무구분자", "P3_2자리_연도",
              "P4_문자혼동_정규화", "P5_월일만", "P6_날짜없음"):
        v = pc.get(k, 0)
        d, plan = DESC[k]
        L.append(f"| `{k}` | {v} | {v/n*100:.1f}% | {d} | {plan} |")
    L.append("")

    recover = sum(pc.get(k, 0) for k in
                  ("P1_4자리_구분자", "P2_4자리_무구분자", "P3_2자리_연도",
                   "P4_문자혼동_정규화"))
    L.append(f"> **규칙 수정으로 접근 가능한 상한: {recover}장 "
             f"({recover/n*100:.1f}%)**. 라벨링 없이 코드만 고쳐서 닿는 범위다.")
    L.append(f"> 나머지 {n-recover}장은 인식·검출 품질 문제라 라벨링이나 엔진 쪽 손질이 필요하다.")
    L.append("")

    # 원인×탐침 교차
    L.append("## 69단계 원인 분류와의 교차\n")
    KO = {"no_text": "글자 못 찾음", "text_no_digits": "숫자 줄 없음",
          "digits_no_date": "숫자는 있으나 날짜 아님"}
    keys = sorted({r["probe"] for r in res})
    L.append("| 69단계 원인 | " + " | ".join(f"`{k.split('_')[0]}`" for k in keys) + " |")
    L.append("| --- | " + " | ".join("---:" for _ in keys) + " |")
    for c in ("digits_no_date", "text_no_digits", "no_text"):
        sub = [r for r in res if cause.get(r["file"]) == c]
        if not sub:
            continue
        cc = Counter(r["probe"] for r in sub)
        L.append(f"| {KO[c]} ({len(sub)}) | " +
                 " | ".join(str(cc.get(k, 0)) for k in keys) + " |")
    L.append("")

    # 사례 — 탐침별로 몇 장씩 실제 OCR 텍스트를 보여 준다
    L.append("## 사례 (OCR 이 실제로 읽은 것)\n")
    for k in ("P1_4자리_구분자", "P2_4자리_무구분자", "P3_2자리_연도",
              "P4_문자혼동_정규화"):
        sub = [r for r in res if r["probe"] == k][:3]
        if not sub:
            continue
        L.append(f"### `{k}` — {DESC[k][0]}\n")
        for r in sub:
            gt = lab.get(r["file"])
            joined = " / ".join(t for t in r["texts"] if any(ch.isdigit() for ch in t))
            L.append(f"- `{r['file']}`" + (f" (정답 {gt})" if gt else ""))
            L.append(f"  - 숫자 포함 조각: `{joined[:200]}`")
        L.append("")

    L.append("## 다음 판단\n")
    L.append("1. `P1`~`P4` 가 많으면 **규칙 수정이 라벨링보다 싸다.** 코드만 고치면 된다")
    L.append("2. `P6` 가 많으면 인식 품질 문제라 **엔진 쪽 또는 박스 라벨링**으로 간다")
    L.append("3. 어느 쪽이든 수정 후 검증은 **동결한 홀드아웃2**로만 한다")
    L.append("")
    L.append("> 전체 텍스트 덤프는 `eda/out/71_failure_texts.json` 에 있다.")

    with open(os.path.join(OUT_DIR, "71_failure_text_diag.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L))
    print()
    print("\n".join(L[10:26]))
    print("\n→ eda/out/71_failure_text_diag.md")


if __name__ == "__main__":
    main()
