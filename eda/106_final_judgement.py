"""
EDA 106단계 — 인식기 칸 최종 판정과 합산 성능

104단계의 자동 판단("악화 0장이니 채택")은 기준이 너무 느슨하다. p=1.0000 을
무시하기 때문이다. 여기서는 **효과 크기**로 다시 본다.

    표본            개선 장수 / 전체      개선율
    580장(오염)     6 / 580              1.03%
    홀드아웃5 100장 1 / 100              1.00%

두 값이 거의 같다. 1600px 칸과 구별되는 지점이 이것이다.
    1600px  580장 +0.6%p  →  홀드아웃4 **0장** (효과가 사라졌다)
    rec 칸  580장 +0.9%p  →  홀드아웃5 1장   (효과 크기가 그대로다)
표본 100장은 1% 짜리 효과를 통계적으로 잡기에 작다. 기대 개선이 1장이고
실제로 1장이 나왔으므로, p=1.0 은 "효과가 없다"가 아니라 "이 표본으로는
있다 없다를 가릴 수 없다"는 뜻이다.

그래서 여기서는
  ① 홀드아웃4(148) + 홀드아웃5(100) = 248장을 **최종 구성 하나로** 다시 재서
     보고용 수치를 낸다. 표본이 커져 신뢰구간이 좁아진다.
  ② 채택 여부를 비용과 함께 정리한다.

    python eda/106_final_judgement.py --workers 4
"""
import os
import sys
import json
import time
import argparse
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

IMG_DIR = os.path.join(ROOT, "images", "상품사진입니다")
OUT = os.path.join(ROOT, "eda", "out")
ONNX = os.path.join(ROOT, "weights", "yolo", "date_v1.onnx")
RAPID_DIR = os.path.join(ROOT, "weights", "rapidocr")

_P = None


def build():
    import pipeline as P
    import team2_rules
    import combined
    import ocrlib
    rapid, _ = combined.build_rapid_ocr(RAPID_DIR)
    fb = P.DatePipeline(ocrlib.build_ocr(det_side=640, threads=1),
                        long_side=640, rotations=(0, 90), adaptive=True, keep=None)
    if os.path.exists(ONNX):
        fb.yolo = P.OnnxDateDetector(ONNX, threads=1)
        fb.yolo_conf = 0.25
        fb.yolo_mode = "always"
    retry, _ = combined.build_rapid_retry(RAPID_DIR)
    ko, _ = combined.build_rapid_rec(RAPID_DIR, combined.REC_KOREAN)
    en, _ = combined.build_rapid_rec(RAPID_DIR, combined.REC_EN)
    import dotcrnn
    dot, _ = dotcrnn.build_crnn(os.path.join(ROOT, "weights"))
    dbox = P.OnnxDateDetector(ONNX, threads=1) if os.path.exists(ONNX) else None
    return combined.CombinedPipeline(rapid, team2_rules, fb, rapid_retry=retry,
                                     rec_korean=ko, rec_en=en,
                                     dot=dot, dot_boxes=dbox)


def work(f):
    global _P
    if _P is None:
        _P = build()
    return {"file": f,
            "pred": "-".join(_P.predict(os.path.join(IMG_DIR, f))[k]
                             for k in ("year", "month", "day"))}


def s3(v):
    if not v or v == "NONE":
        return ("NONE",) * 3
    q = str(v).split("-")
    return tuple(q) if len(q) == 3 else ("NONE",) * 3


def wilson(k, n):
    if n == 0:
        return 0.0, 0.0
    from math import sqrt
    z = 1.96
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    lab, src = {}, {}
    for nm, tag in (("truth_holdout4.json", "홀드아웃4"),
                    ("truth_holdout5.json", "홀드아웃5")):
        d = json.load(open(os.path.join(OUT, nm), encoding="utf-8"))["labels"]
        for k, v in d.items():
            lab[k] = v
            src[k] = tag
    files = [f for f in sorted(lab) if os.path.exists(os.path.join(IMG_DIR, f))]
    print(f"합산 {len(files)}장 (홀드아웃4 + 홀드아웃5) · 최종 구성 1회", flush=True)

    t0 = time.time()
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(work, files, chunksize=2))
    pred = {r["file"]: r["pred"] for r in rows}
    print(f"  {time.time()-t0:.0f}초", flush=True)

    json.dump({"pred": pred, "label": {f: lab[f] for f in files},
               "src": {f: src[f] for f in files}},
              open(os.path.join(OUT, "106_final.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    def score(subset):
        ok = tot = miss = fp = exact = 0
        fy = fm = fd = 0
        for f in subset:
            g, p = s3(lab[f]), s3(pred[f])
            fy += g[0] == p[0]; fm += g[1] == p[1]; fd += g[2] == p[2]
            ok += sum(g[i] == p[i] for i in range(3)); tot += 3
            gn, pn = g == ("NONE",) * 3, p == ("NONE",) * 3
            miss += (pn and not gn)
            fp += (gn and not pn)
            exact += g == p
        n = max(len(subset), 1)
        return {"n": n, "ok": ok, "tot": tot, "score": ok / max(tot, 1),
                "y": fy / n, "m": fm / n, "d": fd / n,
                "exact": exact, "exact_r": exact / n, "miss": miss, "fp": fp}

    h4 = [f for f in files if src[f] == "홀드아웃4"]
    h5 = [f for f in files if src[f] == "홀드아웃5"]
    S = {"전체": score(files), "홀드아웃4": score(h4), "홀드아웃5": score(h5)}
    lo, hi = wilson(S["전체"]["ok"], S["전체"]["tot"])
    elo, ehi = wilson(S["전체"]["exact"], S["전체"]["n"])

    L = ["# EDA 106 — 인식기 칸 최종 판정과 합산 성능\n",
         "104단계의 자동 판단은 기준이 느슨했다(p=1.0000 을 무시하고 '악화 0장이니",
         "채택'). 여기서는 효과 크기로 다시 보고, 표본을 합쳐 보고용 수치를 낸다.",
         "",
         "## 효과 크기로 본 채택 근거\n",
         "| 표본 | 개선 / 전체 | 개선율 |",
         "| --- | ---: | ---: |",
         "| 580장 (오염됨) | 6 / 580 | 1.03% |",
         "| 홀드아웃5 100장 | 1 / 100 | 1.00% |",
         "",
         "두 값이 거의 같다. 이것이 1600px 칸과 갈리는 지점이다.",
         "",
         "| 시도 | 오염 표본 | 깨끗한 표본 | 판정 |",
         "| --- | --- | --- | --- |",
         "| 1600px 칸 | +0.6%p | **0장** | 기각 — 효과가 사라졌다 |",
         "| **rec 한국어·영문** | +0.9%p | **1장 (기대치와 일치)** | **채택** |",
         "",
         "표본 100장은 1% 효과를 통계적으로 잡기에 작다(기대 개선 1장). p=1.0 은",
         "\"효과가 없다\"가 아니라 \"이 표본으로는 가릴 수 없다\"는 뜻이다.",
         "메커니즘도 분명하다. 주력 인식기가 중국어 모델이라 한글을 엉뚱한 한자로",
         "읽고, 그 글자가 날짜에 붙어 있으면 날짜까지 깨뜨린다.",
         "",
         "## 합산 성능 (최종 구성)\n",
         "| 표본 | **필드 정확도** | year | month | day | 완전일치 | 미검출 | 오탐 |",
         "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for k in ("홀드아웃4", "홀드아웃5", "전체"):
        s = S[k]
        nm = f"**{k} {s['n']}장**" if k == "전체" else f"{k} {s['n']}장"
        L.append(f"| {nm} | **{s['score']*100:.1f}%** | {s['y']*100:.1f}% | "
                 f"{s['m']*100:.1f}% | {s['d']*100:.1f}% | {s['exact_r']*100:.1f}% | "
                 f"{s['miss']} | {s['fp']} |")
    L.append("")
    L.append(f"**합산 {S['전체']['n']}장 필드 정확도 {S['전체']['score']*100:.1f}% "
             f"(95% CI {lo*100:.1f}~{hi*100:.1f}%)**")
    L.append(f"완전일치 {S['전체']['exact_r']*100:.1f}% (95% CI {elo*100:.1f}~{ehi*100:.1f}%)")
    L.append("")
    L.append("> 홀드아웃4 는 앞서 최종 판정과 1600px 칸 기각에 썼다. 다만 인식기 칸의")
    L.append("> **채택 여부는 홀드아웃5 에서만 정했으므로**, 여기서 홀드아웃4 를 합친 것은")
    L.append("> 선택이 아니라 표본을 넓혀 신뢰구간을 좁히기 위한 것이다.")
    L.append("")
    L.append("## 비용\n")
    L.append("| 항목 | 값 |")
    L.append("| --- | --- |")
    L.append("| 모델 추가 | korean 12.9MB + en 7.5MB = **20.4MB** |")
    L.append("| 속도 (단일 프로세스) | 0.85초 → **0.95초**/장 |")
    L.append("| 500장 환산 | 425초 → **474초** (제한 2,400초의 **20%**) |")
    L.append("")
    L.append("장당 예산 4.8초의 20% 만 쓴다. 속도 여유가 충분해 지불할 만하다.")

    with open(os.path.join(OUT, "106_final_judgement.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L))
    print()
    print("\n".join(L[22:]))
    print("\n→ eda/out/106_final_judgement.md")


if __name__ == "__main__":
    main()
