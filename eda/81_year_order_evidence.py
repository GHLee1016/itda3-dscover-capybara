"""
EDA 81단계 — 2자리 연도 순서를 무엇으로 정할지, 근거를 먼저 모은다

79단계에서 원인을 찾았다. team2_rules 는 `YY.MM.DD` 와 `DD.MM.YY` 가 모두 성립할 때
**유럽식을 기본으로** 고른다(`team2_rules.py` 460-462). 78단계 오답 5장이 전부
한국식이었다.

그런데 근거가 5장뿐이다. 그것만 보고 기본값을 뒤집는 것은 안전하지 않다.
이미 라벨링한 383장에 훨씬 큰 근거가 있으므로 거기서 정한다.

집합 분리 (이게 이 단계의 핵심이다)
    결정용  튜닝셋 137 + 1차 홀드아웃 48 = **185장**   ← 여기서만 규칙을 고른다
    중간확인 홀드아웃2 198장                          ← 79단계 진단에 썼으므로 오염됨
    최종검증 홀드아웃3 (라벨링 중)                     ← 여기서만 개선폭을 주장한다

⚠️ 결정용 185장은 1차 예선 설계에 이미 쓰였다. 그래도 '2자리 연도 순서'라는
   특정 질문에 대해서는 한 번도 보지 않은 축이라, 결정 근거로 쓸 수 있다.
   다만 여기서 나온 수치를 성능으로 주장하지는 않는다.

비교할 규칙
    R0_always_dmy   현재 동작. 둘 다 되면 유럽식
    R1_always_ymd   둘 다 되면 한국식
    R2_latin        한글이 거의 없으면 유럽식, 아니면 한국식 (pipeline.latin_dominant)
    R3_anchor       같은 조각에 부터/까지 가 있으면 한국식, 아니면 latin 신호
    R4_barcode      정확히 13자리 EAN 국가 접두어. 없으면 한국식
    R5_abstain      모호하면 연도를 비운다 (정밀도 우선)

    python eda/81_year_order_evidence.py --workers 4
"""
import os
import re
import sys
import json
import argparse
import warnings
from collections import Counter
from datetime import date

warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "eda"))

IMG_DIR = os.path.join(ROOT, "images", "상품사진입니다")
OUT_DIR = os.path.join(ROOT, "eda", "out")
RAPID_DIR = os.path.join(ROOT, "weights", "rapidocr")

# team2 규칙의 2자리 연도 패턴과 같은 모양
TWO = re.compile(r"(?<!\d)(\d{2})\s*[.\-/]\s*(\d{1,2})\s*[.\-/]\s*(\d{1,2})(?!\d)")
DIGIT_RUN = re.compile(r"\d+")
ANCHOR = ("부터", "까지", "소비기한", "유통기한", "품질유지기한")

_S = None


def setup():
    global _S
    if _S is None:
        import combined
        rapid, _ = combined.build_rapid_ocr(RAPID_DIR)
        _S = (combined, rapid)
    return _S


def valid(y, m, d):
    try:
        if not (2015 <= y <= 2035):
            return False
        date(y, m, d)
        return True
    except ValueError:
        return False


def ean13_ok(code):
    if len(code) != 13 or not code.isdigit():
        return False
    dg = [int(c) for c in code]
    tot = sum(v if i % 2 == 0 else v * 3 for i, v in enumerate(dg[:-1]))
    return (10 - tot % 10) % 10 == dg[-1]


def barcode_hint(blob):
    """숫자열 길이가 **정확히 13** 일 때만 인정한다 (EDA 60~62)."""
    for run in DIGIT_RUN.findall(blob):
        if len(run) != 13 or not ean13_ok(run):
            continue
        p = int(run[:3])
        if p <= 139 or 754 <= p <= 755:
            return "mdy"
        if p == 880 or 450 <= p <= 459 or 490 <= p <= 499 or 690 <= p <= 699:
            return "kr"          # 한국·일본·중국 → 연-월-일
        if 200 <= p <= 899:
            return "dmy"
    return None


def latin_dominant(text):
    han = sum("가" <= c <= "힣" for c in text)
    lat = sum("a" <= c.lower() <= "z" for c in text)
    return lat > han * 2


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

    cands = []
    for t in texts:
        for m in TWO.finditer(t):
            a_, b_, c_ = int(m.group(1)), int(m.group(2)), int(m.group(3))
            ymd = (2000 + a_, b_, c_)
            dmy = (2000 + c_, b_, a_)
            ok_y, ok_d = valid(*ymd), valid(*dmy)
            if not (ok_y or ok_d):
                continue
            cands.append({
                "raw": m.group(0), "frag": t,
                "ymd": ymd if ok_y else None,
                "dmy": dmy if ok_d else None,
                "frag_anchor": any(k in t for k in ANCHOR),
            })
    return {"file": fname, "cands": cands, "latin": latin_dominant(blob),
            "barcode": barcode_hint(blob),
            "blob_anchor": any(k in blob for k in ANCHOR)}


def fmt(t):
    return "%04d-%02d-%02d" % t if t else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    dec = {}
    for fn in ("truth_v4.json", "truth_holdout.json"):
        p = os.path.join(OUT_DIR, fn)
        if os.path.exists(p):
            dec.update(json.load(open(p, encoding="utf-8"))["labels"])
    h2 = json.load(open(os.path.join(OUT_DIR, "truth_holdout2.json"),
                        encoding="utf-8"))["labels"]

    sets = {"결정용 (튜닝+홀드아웃1)": dec, "중간확인 (홀드아웃2)": h2}
    allfiles = sorted(set(dec) | set(h2))
    allfiles = [f for f in allfiles if os.path.exists(os.path.join(IMG_DIR, f))]
    print(f"라벨 {len(allfiles)}장에서 2자리 연도 후보 추출", flush=True)

    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        res = {r["file"]: r for r in ex.map(work, allfiles, chunksize=2) if r}

    RULES = ["R0_always_dmy", "R1_always_ymd", "R2_latin", "R3_anchor",
             "R4_barcode", "R5_abstain"]

    def decide(rule, c, r):
        """-> 'ymd' | 'dmy' | None(기권)"""
        if c["ymd"] and not c["dmy"]:
            return "ymd"
        if c["dmy"] and not c["ymd"]:
            return "dmy"
        # 여기부터가 모호한 경우
        if rule == "R0_always_dmy":
            return "dmy"
        if rule == "R1_always_ymd":
            return "ymd"
        if rule == "R2_latin":
            return "dmy" if r["latin"] else "ymd"
        if rule == "R3_anchor":
            if c["frag_anchor"] or r["blob_anchor"]:
                return "ymd"
            return "dmy" if r["latin"] else "ymd"
        if rule == "R4_barcode":
            b = r["barcode"]
            if b == "dmy":
                return "dmy"
            if b in ("kr", None):
                return "dmy" if (b is None and r["latin"]) else "ymd"
            return "ymd"
        if rule == "R5_abstain":
            return None
        return "ymd"

    out = {}
    for name, lab in sets.items():
        files = [f for f in lab if f in res]
        amb_n = tot_n = 0
        stat = {k: {"hit": 0, "miss": 0, "abst": 0} for k in RULES}
        examples = []
        for f in files:
            r = res[f]
            truth = lab[f]
            for c in r["cands"]:
                # 정답과 맞춰볼 수 있는 후보만 본다
                gy = fmt(c["ymd"])
                gd = fmt(c["dmy"])
                if truth not in (gy, gd):
                    continue           # 이 후보는 정답과 무관한 숫자열
                tot_n += 1
                ambiguous = bool(c["ymd"]) and bool(c["dmy"])
                if ambiguous:
                    amb_n += 1
                correct = "ymd" if truth == gy else "dmy"
                for k in RULES:
                    got = decide(k, c, r)
                    if got is None:
                        stat[k]["abst"] += 1
                    elif got == correct:
                        stat[k]["hit"] += 1
                    else:
                        stat[k]["miss"] += 1
                if ambiguous and len(examples) < 14:
                    examples.append((f, c["raw"], truth, correct,
                                     r["latin"], r["barcode"],
                                     c["frag_anchor"] or r["blob_anchor"]))
        out[name] = {"n": len(files), "tot": tot_n, "amb": amb_n,
                     "stat": stat, "ex": examples}

    DESC = {
        "R0_always_dmy": "**현재 동작.** 둘 다 되면 유럽식",
        "R1_always_ymd": "둘 다 되면 한국식",
        "R2_latin": "한글이 거의 없으면 유럽식 (latin_dominant)",
        "R3_anchor": "부터/까지 있으면 한국식, 아니면 latin 신호",
        "R4_barcode": "13자리 EAN 국가 접두어, 없으면 latin 신호",
        "R5_abstain": "모호하면 기권 (연도를 비움)",
    }

    L = ["# EDA 81 — 2자리 연도 순서를 무엇으로 정할지\n",
         "79단계에서 원인을 찾았다. 현재 규칙은 `YY.MM.DD` 와 `DD.MM.YY` 가 모두",
         "성립할 때 **유럽식을 기본으로** 고른다. 78단계 오답 5장이 전부 한국식이었다.",
         "",
         "그런데 근거가 5장뿐이라 그것만으로 기본값을 뒤집는 것은 안전하지 않다.",
         "이미 라벨링한 집합에서 정하고, 검증은 새 표본(홀드아웃3)에서 한다.",
         ""]
    for name in sets:
        d = out[name]
        L.append(f"## {name} — {d['n']}장\n")
        L.append(f"정답과 대조 가능한 2자리 연도 후보 **{d['tot']}건**, "
                 f"그중 양쪽 다 성립하는 모호한 것 **{d['amb']}건**")
        L.append("")
        if d["tot"] == 0:
            L.append("대조 가능한 후보가 없다.")
            L.append("")
            continue
        L.append("| 규칙 | 내용 | 맞음 | 틀림 | 기권 | 정확도 |")
        L.append("| --- | --- | ---: | ---: | ---: | ---: |")
        for k in RULES:
            s = d["stat"][k]
            den = s["hit"] + s["miss"]
            acc = f"{s['hit']/den*100:.1f}%" if den else "—"
            L.append(f"| `{k}` | {DESC[k]} | {s['hit']} | {s['miss']} | "
                     f"{s['abst']} | {acc} |")
        L.append("")

    dd = out["결정용 (튜닝+홀드아웃1)"]
    if dd["tot"]:
        best = max([k for k in RULES if k != "R5_abstain"],
                   key=lambda k: dd["stat"][k]["hit"])
        L.append("## 결정\n")
        L.append(f"결정용 집합에서 가장 많이 맞힌 규칙은 **`{best}`** 다.")
        L.append("")
        L.append("| | 맞음 | 틀림 |")
        L.append("| --- | ---: | ---: |")
        for k in ("R0_always_dmy", best):
            s = dd["stat"][k]
            L.append(f"| `{k}` | {s['hit']} | {s['miss']} |")
        L.append("")

    if dd["ex"]:
        L.append("## 모호한 사례 (결정용/중간확인 통합)\n")
        L.append("| 파일 | 원문 | 정답 | 올바른 해석 | latin | 바코드 | 앵커 |")
        L.append("| --- | --- | --- | --- | :-: | :-: | :-: |")
        for f, raw, tr, cor, lat, bc, anc in dd["ex"]:
            L.append(f"| `{f}` | `{raw}` | {tr} | **{cor}** | "
                     f"{'Y' if lat else ''} | {bc or ''} | {'Y' if anc else ''} |")
        L.append("")

    L.append("## 다음\n")
    L.append("1. 결정용 집합에서 고른 규칙을 `team2_rules.py` 에 넣는다")
    L.append("2. 중간확인(홀드아웃2)으로 회귀가 없는지 본다 — 이미 오염된 표본이라 참고용")
    L.append("3. **홀드아웃3 대표 150장으로 최종 판정한다**")
    L.append("")
    L.append("> ⚠️ 결정용 185장은 1차 예선 설계에 쓰인 집합이다. 여기서 나온 정확도를")
    L.append("> 성능으로 주장하지 않는다. 규칙 선택의 근거로만 쓴다.")

    with open(os.path.join(OUT_DIR, "81_year_order_evidence.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L))
    print()
    print("\n".join(L[7:40]))
    print("\n→ eda/out/81_year_order_evidence.md")


if __name__ == "__main__":
    main()
