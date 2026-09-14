"""
EDA 37단계 — 홀드아웃 검증

지금까지의 설계 결정은 전부 정답 137장으로 골랐다. 개선폭이 +0.5~+1.5%p 수준이라
'그 표본에 맞춰 고른 것' 일 가능성을 배제할 수 없다.

설계에 한 번도 관여하지 않은 무작위 48장으로 같은 파이프라인을 측정한다.
튜닝셋과 비슷한 점수가 나오면 과적합이 아니다.
"""
import os
import sys
import json
import time
import argparse
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "eda"))
import pipeline as P
import ocrlib

IMG_DIR = os.path.join(ROOT, "images", "상품사진입니다")
OUT_DIR = os.path.join(ROOT, "eda", "out")

_PIPE = None


def work(f):
    global _PIPE
    if _PIPE is None:
        _PIPE = P.DatePipeline(ocrlib.build_ocr(det_side=640, threads=1),
                               long_side=640, rotations=(0, 90),
                               adaptive=True, keep=None)
    try:
        fin, _, _ = _PIPE.predict_path(os.path.join(IMG_DIR, f))
    except Exception:
        fin = None
    r = P.to_row(os.path.splitext(f)[0], fin)
    r["file"] = f
    return r


def split3(v):
    if v is None or v == "NONE":
        return ("NONE", "NONE", "NONE")
    p = str(v).split("-")
    return tuple(p) if len(p) == 3 else ("NONE", "NONE", "NONE")


def score(rows, lab):
    ok = tot = miss = 0
    wrong = []
    for r in rows:
        g, p = split3(lab[r["file"]]), (r["year"], r["month"], r["day"])
        h = sum(g[i] == p[i] for i in range(3))
        ok += h
        tot += 3
        if p == ("NONE",) * 3 and g != ("NONE",) * 3:
            miss += 1
        if h < 3:
            wrong.append((r["file"], lab[r["file"]], "-".join(p)))
    return ok / tot, miss, wrong


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    ho = json.load(open(os.path.join(OUT_DIR, "truth_holdout.json"), encoding="utf-8"))
    hl = ho["labels"]
    tv = json.load(open(os.path.join(OUT_DIR, "truth_v4.json"), encoding="utf-8"))
    tl = tv["labels"]
    ev = set(tv["membership"]["eval_random"])

    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        hr = list(ex.map(work, list(hl), chunksize=2))
        tr = list(ex.map(work, list(tl), chunksize=2))
    wall = time.time() - t0

    hs, hmiss, hwrong = score(hr, hl)
    ts, tmiss, _ = score(tr, tl)
    es, emiss, _ = score([r for r in tr if r["file"] in ev], tl)

    L = ["# EDA 37 — 홀드아웃 검증\n",
         "설계 결정(해상도 사다리, 전처리 선택, 부터/까지 재시도)은 전부 정답 137장으로 골랐다.",
         "그 표본에 맞춰 고른 것인지 확인하기 위해, 설계에 한 번도 쓰지 않은 무작위 48장으로 측정한다.",
         "",
         "| 집합 | 장수 | 성격 | 부분점수 | 미검출 |",
         "| --- | ---: | --- | ---: | ---: |",
         f"| 튜닝셋 전체 | {len(tl)} | 설계에 사용됨 | **{ts*100:.1f}%** | {tmiss} |",
         f"| 튜닝셋 내 무작위 | {len(ev)} | 설계에 사용됨 | {es*100:.1f}% | {emiss} |",
         f"| **홀드아웃** | **{len(hl)}** | **미사용** | **{hs*100:.1f}%** | {hmiss} |",
         ""]
    gap = (ts - hs) * 100
    if abs(gap) < 6:
        L.append(f"> ✅ 튜닝셋과 홀드아웃의 차이가 **{gap:+.1f}%p** 다. "
                 "표본 크기를 감안하면 과적합이라 보기 어렵다.")
    else:
        L.append(f"> ⚠️ 홀드아웃이 튜닝셋보다 **{gap:.1f}%p** 낮다. "
                 "설계가 튜닝셋에 맞춰졌을 가능성이 있다.")
    L.append("")
    L.append(f"> 홀드아웃 48장의 95% 신뢰구간은 약 ±{1.96*(0.25/48)**0.5*100:.0f}%p 다.")
    L.append("")
    L.append("## 홀드아웃 오답\n")
    L.append("| 파일 | 정답 | 출력 |")
    L.append("| --- | --- | --- |")
    for f, g, p in hwrong:
        L.append(f"| `{f}` | {g} | {p} |")
    L.append("")
    L.append(f"- 맞춘 장수 {len(hl)-len(hwrong)}/{len(hl)}, 총 {wall:.0f}초")

    with open(os.path.join(OUT_DIR, "37_holdout_eval.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
