"""
EDA 89단계 — 오답 63장을 "고칠 수 있는 것"과 "못 읽은 것"으로 가른다

오답 모음을 보고 세 가지가 눈에 띈다는 지적이 있었다.
    ① 흐릿한 사진   ② 점선(도트) 인쇄   ③ 날짜가 두 줄(제조일자 + 소비기한)
셋은 고칠 곳이 다르다. ③ 은 규칙이 고를 때 실수하는 것이고, ①②는 OCR 이
글자를 못 만드는 것이다. 섞어 두면 또 엉뚱한 데를 고치게 된다.

가르는 기준
    정답이 OCR 텍스트 안에 **이미 있는가**
      있다  → 규칙이 못 고른 것.   날짜 후보가 2개 이상이면 그게 ③ 이다
      없다  → OCR 이 못 만든 것.   날짜꼴이 아예 없으면 검출, 있는데 숫자가
                                   다르면 오독(①②가 여기 온다)

같이 재는 것
  · 선명도(라플라시안 분산)를 오답/정답 두 무리에서 비교한다. "흐릿해서 틀린다"가
    맞다면 두 분포가 갈려야 한다. 안 갈리면 그 가설은 버린다.

    python eda/89_error_anatomy.py --workers 4
"""
import os
import re
import sys
import json
import argparse
import warnings
from collections import Counter

warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

IMG_DIR = os.path.join(ROOT, "images", "상품사진입니다")
OUT = os.path.join(ROOT, "eda", "out")
RAPID_DIR = os.path.join(ROOT, "weights", "rapidocr")

MON = {m: i + 1 for i, m in enumerate(
    "JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split())}

SEP = r"[.\-/:\s]"
PATS = [
    re.compile(r"(?<!\d)(\d{4})" + SEP + r"?(\d{1,2})" + SEP + r"?(\d{1,2})(?!\d)"),
    re.compile(r"(?<!\d)(\d{1,2})" + SEP + r"(\d{1,2})" + SEP + r"(\d{4})(?!\d)"),
    re.compile(r"(?<!\d)(\d{2})" + SEP + r"(\d{1,2})" + SEP + r"(\d{1,2})(?!\d)"),
]
PAT_MON = re.compile(
    r"(?<!\d)(\d{1,2})\s*[.\-/]?\s*(" + "|".join(MON) + r")\s*[.\-/]?\s*(\d{2,4})"
    r"|(" + "|".join(MON) + r")\s*[.\-/]?\s*(\d{1,2})\s*[,.\-/]?\s*(\d{2,4})", re.I)


def y4(v):
    v = int(v)
    if v >= 1900:
        return v
    return 2000 + v if v < 70 else 1900 + v


def cands(text):
    """한 조각에서 날짜로 읽힐 수 있는 모든 해석을 낸다. 순서는 정하지 않는다."""
    out = set()
    up = text.upper()
    for m in PAT_MON.finditer(up):
        g = [x for x in m.groups() if x]
        if len(g) != 3:
            continue
        if g[0].upper() in MON:
            mo, d, yy = MON[g[0].upper()], g[1], g[2]
        else:
            d, mo, yy = g[0], MON[g[1].upper()], g[2]
        try:
            out.add((y4(yy), int(mo), int(d)))
        except Exception:
            pass
    for rx in PATS:
        for m in rx.finditer(up):
            a, b, c = m.groups()
            for y, mo, d in ((a, b, c), (c, b, a), (c, a, b)):
                try:
                    Y, M, D = y4(y), int(mo), int(d)
                except Exception:
                    continue
                if 1 <= M <= 12 and 1 <= D <= 31 and 2000 <= Y <= 2099:
                    out.add((Y, M, D))
    return out


def s3(v):
    if not v or v == "NONE":
        return ("NONE",) * 3
    q = str(v).split("-")
    return tuple(q) if len(q) == 3 else ("NONE",) * 3


_S = None


def setup():
    global _S
    if _S is None:
        import combined
        rapid, _ = combined.build_rapid_ocr(RAPID_DIR)
        retry, _ = combined.build_rapid_retry(RAPID_DIR)
        _S = (combined, rapid, retry)
    return _S


def work(f):
    combined, rapid, retry = setup()
    path = os.path.join(IMG_DIR, f)
    img = combined.imread_any(path)
    if img is None:
        return None
    small = combined.downscale_max(img, 1024)

    def texts_of(eng):
        try:
            r = eng(small)
            return list(r.txts) if getattr(r, "txts", None) else []
        except Exception:
            return []

    txt = texts_of(rapid)
    if retry is not None:
        txt = list(dict.fromkeys(txt + texts_of(retry)))

    cs = set()
    for t in txt:
        cs |= cands(t)

    # 선명도 — 원본 기준. 줄이면 값이 달라지므로 같은 크기에서 잰다
    sharp = None
    try:
        import cv2
        g = cv2.cvtColor(combined.downscale_max(img, 1024), cv2.COLOR_BGR2GRAY)
        sharp = float(cv2.Laplacian(g, cv2.CV_64F).var())
    except Exception:
        pass

    return {"file": f, "n_frag": len(txt), "sharp": sharp,
            "cands": sorted(f"{y:04d}-{m:02d}-{d:02d}" for y, m, d in cs),
            "digit_frags": [t for t in txt if sum(c.isdigit() for c in t) >= 4][:12]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    D = json.load(open(os.path.join(OUT, "88_all_pred.json"), encoding="utf-8"))
    lab, pred, split = D["label"], D["pred"], D["split"]
    wrong = [f for f in lab if s3(lab[f]) != s3(pred[f])]
    right = [f for f in lab if s3(lab[f]) == s3(pred[f])]
    print(f"오답 {len(wrong)}장 / 정답 {len(right)}장", flush=True)

    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        W = {r["file"]: r for r in ex.map(work, wrong, chunksize=1) if r}
        # 선명도 비교용으로 정답 쪽도 표본을 본다(전량은 느리다)
        import random
        samp = random.Random(7).sample(right, min(150, len(right)))
        Rs = [r for r in ex.map(work, samp, chunksize=2) if r]

    rows = []
    for f in wrong:
        r = W.get(f)
        if not r:
            continue
        truth = lab[f]
        has = truth in r["cands"]
        multi = len(r["cands"]) >= 2
        if has and multi:
            why = "고를 때 실수 (후보 여럿)"
        elif has:
            why = "후보에 있는데 못 씀"
        elif r["cands"]:
            why = "숫자를 다르게 읽음"
        elif r["n_frag"] == 0:
            why = "글자를 하나도 못 읽음"
        else:
            why = "날짜꼴을 못 만듦"
        rows.append({"f": f, "truth": truth, "pred": pred[f], "why": why,
                     "split": split[f], "sharp": r["sharp"],
                     "cands": r["cands"], "frags": r["digit_frags"]})

    cnt = Counter(x["why"] for x in rows)

    L = ["# EDA 89 — 오답 63장의 해부\n",
         "지적된 세 가지(흐릿·점선·두 줄)는 고칠 곳이 다르다.",
         "정답이 OCR 텍스트에 **이미 있는지**로 가른다.",
         "",
         "| 막힌 지점 | 장수 | 고칠 곳 |",
         "| --- | ---: | --- |"]
    KO = {"고를 때 실수 (후보 여럿)": "**규칙** — 날짜가 둘 이상일 때 고르는 기준",
          "후보에 있는데 못 씀": "**규칙** — 패턴이 그 모양을 안 받음",
          "숫자를 다르게 읽음": "인식기 — 글자 오독 (흐릿·점선)",
          "날짜꼴을 못 만듦": "검출기/인식기",
          "글자를 하나도 못 읽음": "검출기"}
    for k, v in cnt.most_common():
        L.append(f"| {k} | {v} | {KO.get(k, '')} |")
    fix = cnt["고를 때 실수 (후보 여럿)"] + cnt["후보에 있는데 못 씀"]
    L.append("")
    L.append(f"**규칙으로 고칠 수 있는 것 {fix}장 / OCR 문제 {len(rows)-fix}장**")
    L.append("")

    # 선명도 비교 — "흐릿해서 틀린다" 가 맞는지
    def q(v, p):
        v = sorted(x for x in v if x is not None)
        return v[int(len(v) * p)] if v else 0

    ws = [x["sharp"] for x in rows]
    rs = [r["sharp"] for r in Rs]
    L.append("## 흐릿함이 원인인가\n")
    L.append("라플라시안 분산(클수록 또렷). 오답이 정말 흐린 쪽이면 두 분포가 갈려야 한다.")
    L.append("")
    L.append("| 무리 | 장수 | 하위25% | 중앙값 | 상위25% |")
    L.append("| --- | ---: | ---: | ---: | ---: |")
    L.append(f"| 오답 | {len(ws)} | {q(ws,.25):.0f} | {q(ws,.5):.0f} | {q(ws,.75):.0f} |")
    L.append(f"| 정답(표본) | {len(rs)} | {q(rs,.25):.0f} | {q(rs,.5):.0f} | {q(rs,.75):.0f} |")
    L.append("")
    mw, mr = q(ws, .5), q(rs, .5)
    if mw and mr:
        if mw < mr * 0.75:
            L.append(f"오답 쪽 중앙값이 {mr/mw:.1f}배 낮다. 흐릿함이 실제로 관련 있다.")
        elif mw > mr * 1.25:
            L.append("오답 쪽이 오히려 또렷하다. 흐릿함 가설은 맞지 않는다.")
        else:
            L.append("두 무리가 거의 같다. **흐릿함만으로는 오답을 설명하지 못한다.**")
    L.append("")

    for why, _ in cnt.most_common():
        sub = [x for x in rows if x["why"] == why]
        L.append(f"## {why} — {len(sub)}장\n")
        L.append("| 파일 | 정답 | 예측 | OCR 이 만든 날짜 후보 |")
        L.append("| --- | --- | --- | --- |")
        for x in sub:
            c = ", ".join(x["cands"][:6]) or "(없음)"
            L.append(f"| `{x['f']}` | {x['truth']} | {x['pred']} | {c} |")
        L.append("")

    L.append("## 숫자 조각 원문 (OCR 문제로 분류된 것)\n")
    for x in rows:
        if x["why"] in ("고를 때 실수 (후보 여럿)", "후보에 있는데 못 씀"):
            continue
        L.append(f"- `{x['f']}` 정답 {x['truth']} → `{' | '.join(x['frags'])[:180]}`")

    with open(os.path.join(OUT, "89_error_anatomy.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    json.dump(rows, open(os.path.join(OUT, "89_error_anatomy.json"), "w",
                         encoding="utf-8"), ensure_ascii=False, indent=1)
    print()
    print("\n".join(L[3:14]))
    print("\n→ eda/out/89_error_anatomy.md")


if __name__ == "__main__":
    main()
