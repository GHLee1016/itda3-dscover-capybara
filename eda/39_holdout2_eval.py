"""
EDA 39단계 — 2차 홀드아웃 200장 검증

1차 홀드아웃 48장은 연도 후행형 버그를 찾아 고치는 데 써서 더 이상
과적합 검증에 쓸 수 없다. 설계·디버깅에 한 번도 관여하지 않은 200장으로
같은 파이프라인을 측정한다.

표본 200장이면 95% 신뢰구간이 약 ±7%p 로, 지금까지의 개선폭(±5%p)을
유의하게 판정할 수 있다.
"""
import os
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
    ok = tot = miss = fp = 0
    fy = fm = fd = 0
    wrong = []
    for r in rows:
        g, p = split3(lab[r["file"]]), (r["year"], r["month"], r["day"])
        fy += g[0] == p[0]
        fm += g[1] == p[1]
        fd += g[2] == p[2]
        ok += sum(g[i] == p[i] for i in range(3))
        tot += 3
        gnone = g == ("NONE",) * 3
        pnone = p == ("NONE",) * 3
        if pnone and not gnone:
            miss += 1
        if gnone and not pnone:
            fp += 1
        if g != p:
            wrong.append((r["file"], lab[r["file"]], "-".join(p)))
    n = len(rows)
    return {"score": ok / tot, "y": fy / n, "m": fm / n, "d": fd / n,
            "exact": (n - len(wrong)) / n, "miss": miss, "fp": fp,
            "wrong": wrong, "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    h2 = json.load(open(os.path.join(OUT_DIR, "truth_holdout2.json"), encoding="utf-8"))["labels"]
    h1 = json.load(open(os.path.join(OUT_DIR, "truth_holdout.json"), encoding="utf-8"))["labels"]
    tv = json.load(open(os.path.join(OUT_DIR, "truth_v4.json"), encoding="utf-8"))["labels"]

    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        r2 = list(ex.map(work, list(h2), chunksize=2))
        r1 = list(ex.map(work, list(h1), chunksize=2))
        rt = list(ex.map(work, list(tv), chunksize=2))
    wall = time.time() - t0

    s2, s1, st = score(r2, h2), score(r1, h1), score(rt, tv)

    L = ["# EDA 39 — 2차 홀드аut 200장 검증\n".replace("аut", "아웃"),
         "설계·디버깅에 쓴 표본과, 한 번도 쓰지 않은 표본을 나란히 놓고 본다.",
         "홀드아웃에서 성능이 유지되면 지금까지의 개선이 표본에 맞춘 것이 아니다.",
         "",
         "| 집합 | 장수 | 성격 | **부분점수** | year | month | day | 완전일치 |",
         "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
         f"| 튜닝셋 | {st['n']} | 설계에 사용 | **{st['score']*100:.1f}%** | "
         f"{st['y']*100:.1f}% | {st['m']*100:.1f}% | {st['d']*100:.1f}% | {st['exact']*100:.1f}% |",
         f"| 1차 홀드아웃 | {s1['n']} | 버그 수정에 사용 | **{s1['score']*100:.1f}%** | "
         f"{s1['y']*100:.1f}% | {s1['m']*100:.1f}% | {s1['d']*100:.1f}% | {s1['exact']*100:.1f}% |",
         f"| **2차 홀드아웃** | **{s2['n']}** | **미사용** | **{s2['score']*100:.1f}%** | "
         f"{s2['y']*100:.1f}% | {s2['m']*100:.1f}% | {s2['d']*100:.1f}% | {s2['exact']*100:.1f}% |",
         ""]
    gap = (st["score"] - s2["score"]) * 100
    ci = 1.96 * (0.25 / s2["n"]) ** 0.5 * 100
    L.append(f"> 튜닝셋과 2차 홀드아웃의 차이 **{gap:+.1f}%p**. "
             f"200장의 95% 신뢰구간이 ±{ci:.0f}%p 이므로 "
             + ("**과적합이라 보기 어렵다.**" if abs(gap) < ci else
                "**표본에 맞춰졌을 가능성이 있다.**"))
    L.append("")
    L.append(f"- 미검출(정답 있는데 NONE): {s2['miss']}장 ({s2['miss']/s2['n']*100:.1f}%)")
    L.append(f"- 오탐(정답 NONE인데 날짜): {s2['fp']}장")
    L.append(f"- 실행 {wall:.0f}초")
    L.append("")

    L.append("## 2차 홀드아웃 오답 전체\n")
    L.append("| 파일 | 정답 | 출력 |")
    L.append("| --- | --- | --- |")
    for f, g, p in s2["wrong"]:
        L.append(f"| `{f}` | {g} | {p} |")
    L.append("")

    # 오답 유형
    def kind(g, p):
        G, Pp = split3(g), tuple(p.split("-"))
        if Pp == ("NONE", "NONE", "NONE"):
            return "미검출"
        if G[1:] == Pp[1:]:
            return "연도만 오독"
        if G[0] == Pp[0]:
            return "월/일 오독"
        return "완전히 다름"
    c = Counter(kind(g, p) for _, g, p in s2["wrong"])
    L.append("## 오답 유형\n")
    L.append("| 유형 | 장수 |")
    L.append("| --- | ---: |")
    for k, v in c.most_common():
        L.append(f"| {k} | {v} |")

    with open(os.path.join(OUT_DIR, "39_holdout2_eval.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L[:24]))


if __name__ == "__main__":
    main()
