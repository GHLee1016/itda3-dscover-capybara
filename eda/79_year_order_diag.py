"""
EDA 79단계 — 연/일 자리바꿈 오독의 원인 추적

78단계에서 뚜렷한 패턴이 나왔다.
    002172  정답 2020-06-24  출력 2024-06-20
    002340  정답 2021-12-28  출력 2028-12-21
    000810  정답 2026-03-31  출력 2031-03-26
    000771  정답 2019-11-26  출력 2026-11-19
  연도와 일이 서로 자리를 바꿨다. `20.06.24` 를 `YY.MM.DD` 가 아니라
  `DD.MM.YY` 로 읽은 것이다. 손실이 3.7~5.9%p 로 미검출(4.4%p)에 맞먹는다.

의심: 팀원2 규칙의 `_detect_country_bucket` 이다.
  EAN-13 바코드의 국가 접두어로 표기 순서를 추정하는데, **8자리 이상 숫자열 안의
  모든 13자리 슬라이딩 윈도우**를 검사한다. 국내 상품에 거의 항상 있는
  품목보고번호(14자리)의 부분열이 우연히 체크섬을 통과하면 접두어가 200~899 로
  잡혀 "dmy" 가 되고, 한국 날짜가 유럽식으로 뒤집힌다.

  이 버그는 EDA 60~62 에서 우리 pipeline.py 로 이식할 때 발견해 그쪽만 고쳤다
  (숫자열 길이가 정확히 13 일 때만 인정). 주력 경로는 팀원2 원문을 그대로 쓴다.

여기서는 가설을 확인만 한다. 수정과 검증은 그다음이다.

    python eda/79_year_order_diag.py --workers 4
"""
import os
import re
import sys
import json
import argparse
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "eda"))

IMG_DIR = os.path.join(ROOT, "images", "상품사진입니다")
OUT_DIR = os.path.join(ROOT, "eda", "out")
RAPID_DIR = os.path.join(ROOT, "weights", "rapidocr")

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
    if not texts and retry is not None:
        texts = texts_of(retry, small)

    # 규칙이 보는 형태 그대로 (box, text, conf) 삼중항이 필요하다
    triples = [([[0, 0], [1, 0], [1, 1], [0, 1]], t, 1.0) for t in texts]
    bucket = rules._detect_country_bucket(triples)

    # 어떤 숫자열이 바코드로 인정됐는지 찾아낸다
    culprit = []
    for t in texts:
        for run in re.findall(r"\d{8,}", t):
            for i in range(0, len(run) - 12):
                code = run[i:i + 13]
                if rules._ean13_checksum_ok(code):
                    b = rules._barcode_country_bucket(code)
                    culprit.append((run, code, b, len(run)))
    return {"file": fname, "bucket": bucket, "culprit": culprit,
            "digit_frags": [t for t in texts if sum(c.isdigit() for c in t) >= 4][:12]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    lab = json.load(open(os.path.join(OUT_DIR, "truth_holdout2.json"),
                         encoding="utf-8"))["labels"]
    preds = json.load(open(os.path.join(OUT_DIR, "76_escalation.json"),
                           encoding="utf-8"))["escalation"]

    def s3(v):
        if not v or v == "NONE":
            return ("NONE",) * 3
        q = str(v).split("-")
        return tuple(q) if len(q) == 3 else ("NONE",) * 3

    # 연/일 자리바꿈으로 보이는 것 + 월/일 오독을 대상으로
    targets = []
    for f in lab:
        if f not in preds:
            continue
        g, p = s3(lab[f]), s3(preds[f])
        if g == p or "NONE" in (g[0], p[0]):
            continue
        if g[0] != p[0] and g[0][-2:] == p[2] and g[2] == p[0][-2:]:
            targets.append((f, "연/일 자리바꿈"))
        elif g[0] != p[0]:
            targets.append((f, "연도 오독"))
        elif g[1:] != p[1:]:
            targets.append((f, "월/일 오독"))
    print(f"대상 {len(targets)}장", flush=True)

    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        res = {r["file"]: r for r in
               ex.map(work, [t[0] for t in targets], chunksize=1) if r}

    kind = dict(targets)
    swap = [f for f, k in targets if k == "연/일 자리바꿈"]
    dmy = [f for f in res if res[f]["bucket"] == "dmy"]
    swap_dmy = [f for f in swap if res.get(f, {}).get("bucket") == "dmy"]

    L = ["# EDA 79 — 연/일 자리바꿈 오독의 원인 추적\n",
         "78단계에서 `2020-06-24` → `2024-06-20` 형태의 자리바꿈이 반복됐다.",
         "`YY.MM.DD` 를 `DD.MM.YY` 로 읽은 것이다. 손실이 미검출과 맞먹는다.",
         "",
         "의심 대상은 팀원2 규칙의 `_detect_country_bucket` 이다. EAN-13 국가",
         "접두어로 표기 순서를 정하는데, **8자리 이상 숫자열의 모든 13자리 윈도우**를",
         "검사한다. 국내 상품에 거의 항상 있는 품목보고번호(14자리)의 부분열이",
         "우연히 체크섬을 통과하면 `dmy` 로 잡혀 한국 날짜가 뒤집힌다.",
         "",
         f"- 대상 {len(targets)}장 (연/일 자리바꿈 {len(swap)}장 포함)",
         f"- 국가 힌트가 `dmy` 로 잡힌 이미지 **{len(dmy)}장**",
         f"- 자리바꿈 중 `dmy` 로 잡힌 것 **{len(swap_dmy)}/{len(swap)}장**",
         ""]
    if swap:
        L.append(f"> 자리바꿈의 **{len(swap_dmy)/len(swap)*100:.0f}%** 가 가짜 바코드 힌트로 설명된다."
                 if swap_dmy else
                 "> 자리바꿈이 바코드 힌트로 설명되지 않는다. 다른 원인을 봐야 한다.")
        L.append("")

    L.append("## 이미지별 진단\n")
    L.append("| 파일 | 정답 | 출력 | 유형 | 국가힌트 | 바코드로 인정된 숫자열 |")
    L.append("| --- | --- | --- | --- | --- | --- |")
    for f, k in targets:
        r = res.get(f)
        if not r:
            continue
        cul = ""
        if r["culprit"]:
            run, code, b, ln = r["culprit"][0]
            cul = f"`{run}` ({ln}자리) → `{code}` = {b}"
        L.append(f"| `{f}` | {lab[f]} | {preds[f]} | {k} | "
                 f"**{r['bucket']}** | {cul} |")
    L.append("")

    # 가짜 바코드의 원본 길이 분포 — 진짜 바코드는 정확히 13자리다
    lens = {}
    for f in res:
        for run, code, b, ln in res[f]["culprit"]:
            lens[ln] = lens.get(ln, 0) + 1
    if lens:
        L.append("## 바코드로 인정된 숫자열의 원본 길이\n")
        L.append("| 원본 길이 | 건수 | 판정 |")
        L.append("| ---: | ---: | --- |")
        for ln in sorted(lens):
            verdict = "진짜 바코드일 수 있음" if ln == 13 else "**가짜** (더 긴 숫자열의 부분)"
            L.append(f"| {ln}자리 | {lens[ln]} | {verdict} |")
        L.append("")
        L.append("진짜 EAN-13 은 숫자열 길이가 정확히 13 이다. 그보다 긴 숫자열에서")
        L.append("잘라낸 13자리가 체크섬을 통과하는 것은 우연이며, 1/10 확률로 일어난다.")
        L.append("")

    L.append("## 수정안\n")
    L.append("`team2_rules._detect_country_bucket` 에서 **숫자열 길이가 정확히 13 일 때만**")
    L.append("바코드로 인정한다. 우리 `pipeline.py` 에는 이미 그렇게 고쳐 둔 코드가 있다")
    L.append("(EDA 60~62). 주력 경로가 팀원2 원문을 쓰고 있어 반영되지 않았다.")
    L.append("")
    L.append("> ⚠️ 이 진단에 홀드아웃2 를 썼다. 수정 후 **같은 표본으로 재면 안 된다.**")
    L.append("> 새 표본에서 검증해야 한다.")

    with open(os.path.join(OUT_DIR, "79_year_order_diag.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L))
    print()
    print("\n".join(L[8:14]))
    print("\n→ eda/out/79_year_order_diag.md")


if __name__ == "__main__":
    main()
