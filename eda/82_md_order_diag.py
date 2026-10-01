"""
EDA 82단계 — 남은 월/일 오독과 연도 자리바꿈의 원인 추적

81단계에서 2자리 연도 기본값을 한국식으로 바꿔 `other` 가 9→5장으로 줄었다.
그런데 남은 5장 중 `001558` 은 여전히 유럽식이다.
    001558  정답 2021-12-24  출력 2024-12-21
`team2_rules` 의 pattern6(F 구역)은 고쳤으니, **다른 분기가 날짜를 만들고 있다**.

team2_rules 에는 2자리 연도를 다루는 곳이 여럿이다.
    F  pattern6  `DD.MM.YY` / `YY.MM.DD`   ← 81단계에서 고친 곳
    G  pattern7  공백 구분 `09 11 22`
    I  pattern9  무구분 `YYMMDD`
고치지 않은 분기가 같은 실수를 반복하고 있을 가능성이 크다.

여기서는 원인만 확인한다. 대상은 78단계가 남긴 오답 중 월/일 또는 연도가 틀린 것.

    python eda/82_md_order_diag.py --workers 4
"""
import os
import re
import sys
import json
import argparse
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "eda"))

IMG_DIR = os.path.join(ROOT, "images", "상품사진입니다")
OUT_DIR = os.path.join(ROOT, "eda", "out")
RAPID_DIR = os.path.join(ROOT, "weights", "rapidocr")

# team2_rules 의 각 2자리 연도 분기가 잡는 모양
PATS = [
    ("F 구분자", re.compile(r"(?<!\d)(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{2})(?!\d)")),
    ("G 공백",   re.compile(r"(?<!\d)(\d{2})\s+(\d{2})\s+(\d{2})(?!\d)")),
    ("I 무구분", re.compile(r"(?<!\d)(\d{2})(\d{2})(\d{2})(?!\d)")),
    ("A 4자리",  re.compile(r"(?<!\d)(20\d{2})[.\-/](\d{1,2})[.\-/](\d{1,2})(?!\d)")),
]

_S = None


def setup():
    global _S
    if _S is None:
        import combined
        import team2_rules
        rapid, _ = combined.build_rapid_ocr(RAPID_DIR)
        retry, _ = combined.build_rapid_retry(RAPID_DIR)
        _S = (combined, team2_rules, rapid, retry)
    return _S


def work(fname):
    combined, rules, rapid, retry = setup()
    img = combined.imread_any(os.path.join(IMG_DIR, fname))
    if img is None:
        return None
    small = combined.downscale_max(img, 1024)

    def texts_of(engine, a):
        try:
            r = engine(a)
            return list(r.txts) if getattr(r, "txts", None) else []
        except Exception:
            return []

    texts = texts_of(rapid, small)
    stage = "rapid"
    if not texts or rules.choose_expiration_date(texts) is None:
        if retry is not None:
            t2 = texts_of(retry, small)
            if t2 and rules.choose_expiration_date(t2) is not None:
                texts, stage = t2, "retry"

    hits = []
    for t in texts:
        up = t.upper()
        for name, rx in PATS:
            for m in rx.finditer(up):
                hits.append({"branch": name, "raw": m.group(0), "frag": t})
    return {"file": fname, "stage": stage, "hits": hits,
            "digit_frags": [t for t in texts if sum(c.isdigit() for c in t) >= 4][:10],
            "final": rules.choose_expiration_date(texts)}


def s3(v):
    if not v or v == "NONE":
        return ("NONE",) * 3
    q = str(v).split("-")
    return tuple(q) if len(q) == 3 else ("NONE",) * 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    lab = json.load(open(os.path.join(OUT_DIR, "truth_holdout2.json"),
                         encoding="utf-8"))["labels"]
    preds = json.load(open(os.path.join(OUT_DIR, "76_escalation.json"),
                           encoding="utf-8"))["escalation"]

    targets = []
    for f in lab:
        if f not in preds:
            continue
        g, p = s3(lab[f]), s3(preds[f])
        if g == p:
            continue
        if p == ("NONE",) * 3:
            continue                      # 미검출은 여기 대상이 아니다
        targets.append(f)
    print(f"값을 냈는데 틀린 {len(targets)}장 추적", flush=True)

    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        res = {r["file"]: r for r in ex.map(work, targets, chunksize=1) if r}

    L = ["# EDA 82 — 남은 월/일 오독과 연도 자리바꿈의 원인\n",
         "81단계에서 2자리 연도 기본값을 한국식으로 바꿔 `other` 가 9→5장으로 줄었다.",
         "그런데 `001558` 은 여전히 유럽식이다(정답 2021-12-24, 출력 2024-12-21).",
         "고친 곳은 pattern6(F 구역)뿐이고, team2_rules 에는 2자리 연도를 다루는",
         "분기가 여럿이다. 고치지 않은 분기가 같은 실수를 반복하는지 본다.",
         "",
         f"- 대상 {len(targets)}장 (미검출 제외)",
         "",
         "## 이미지별 추적\n"]
    for f in targets:
        r = res.get(f)
        if not r:
            continue
        L.append(f"### `{f}` — 정답 {lab[f]} / 출력 {preds[f]}\n")
        L.append(f"- 어느 단계에서 읽었나: **{r['stage']}**")
        L.append(f"- 규칙 재실행 결과: `{r['final']}`")
        if r["hits"]:
            L.append("- 날짜 형태로 잡히는 문자열:")
            seen = set()
            for h in r["hits"]:
                key = (h["branch"], h["raw"])
                if key in seen:
                    continue
                seen.add(key)
                L.append(f"  - `{h['branch']}` → `{h['raw']}`  (조각: `{h['frag'][:50]}`)")
        else:
            L.append("- 날짜 형태 문자열 없음")
        L.append(f"- 숫자 조각: `{' / '.join(r['digit_frags'])[:160]}`")
        L.append("")

    # 분기별 집계
    from collections import Counter
    br = Counter(h["branch"] for r in res.values() for h in r["hits"])
    L.append("## 분기별 등장 횟수\n")
    L.append("| 분기 | 건수 |")
    L.append("| --- | ---: |")
    for k, v in br.most_common():
        L.append(f"| {k} | {v} |")
    L.append("")
    L.append("> `F 구분자` 외의 분기가 많이 나오면, 81단계 수정이 그 경로에는")
    L.append("> 적용되지 않았다는 뜻이다. 같은 기본값 문제를 거기에도 적용해야 한다.")

    with open(os.path.join(OUT_DIR, "82_md_order_diag.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L[-10:]))
    print("\n→ eda/out/82_md_order_diag.md")


if __name__ == "__main__":
    main()
