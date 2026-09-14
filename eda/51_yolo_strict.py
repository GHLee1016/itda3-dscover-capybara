"""
EDA 51단계 — 전패스 YOLO 의 퇴행을 신뢰도 문턱으로 막아본다

48단계 결과
  기준선   64.5%
  최후수단 66.2%  (개선 4 / 악화 0)   ← 구조적으로 떨어질 수 없다
  전패스   66.8%  (개선 8 / 악화 4)   ← 정답 4장을 망가뜨렸다

망가진 4장은 모두 YOLO 크롭의 오독이 올바른 det 판독을 이긴 경우다.
  000149  2026-04-28 → 2025-04-28
  000293  2025-10-24 → 2035-10-24
  000417  2026-03-15 → 2025-05-15
  001029  2021-07-07 → 2021-07-08
크롭 TTA(68.4→62.5) 가 실패한 것과 같은 기제다.

가설: YOLO 크롭에만 더 높은 rec 신뢰도를 요구하면 8개 개선은 남기고
      4개 악화만 걸러낼 수 있다.

⚠️ 문턱을 여러 개 쓸어보면 198장의 잡음에 맞추는 것이 된다. 그래서
   **0.90 하나만** 미리 정해 놓고 잰다. 실패하면 최후수단 방식으로 간다.
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
WEIGHTS = os.path.join(ROOT, "yolo_runs", "date_v1", "weights", "best.pt")

THR = 0.90          # 미리 정한 값. 쓸어보지 않는다.

_PIPE = None
_KEY = None


def work(args):
    global _PIPE, _KEY
    f, mode, thr = args
    if _PIPE is None or _KEY != (mode, thr):
        import torch
        torch.set_num_threads(1)
        pipe = P.DatePipeline(ocrlib.build_ocr(det_side=640, threads=1),
                              long_side=640, rotations=(0, 90),
                              adaptive=True, keep=None)
        if mode != "off":
            from ultralytics import YOLO
            pipe.yolo = YOLO(WEIGHTS)
            pipe.yolo_conf = 0.25
            pipe.yolo_mode = mode
            pipe.yolo_drop_score = thr
        _PIPE, _KEY = pipe, (mode, thr)
    try:
        fin, _, _ = _PIPE.predict_path(os.path.join(IMG_DIR, f))
    except Exception:
        fin = None
    r = P.to_row(os.path.splitext(f)[0], fin)
    r["file"] = f
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    h2 = json.load(open(os.path.join(OUT_DIR, "truth_holdout2.json"),
                        encoding="utf-8"))["labels"]
    files = list(h2)

    def split3(v):
        if v is None or v == "NONE":
            return ("NONE",) * 3
        p = str(v).split("-")
        return tuple(p) if len(p) == 3 else ("NONE",) * 3

    def score(rows):
        ok = tot = miss = fp = 0
        fy = fm = fd = 0
        per, wrong = {}, []
        for r in rows:
            g, p = split3(h2[r["file"]]), (r["year"], r["month"], r["day"])
            hit = sum(g[i] == p[i] for i in range(3))
            per[r["file"]] = hit
            fy += g[0] == p[0]; fm += g[1] == p[1]; fd += g[2] == p[2]
            ok += hit; tot += 3
            gn, pn = g == ("NONE",) * 3, p == ("NONE",) * 3
            if pn and not gn:
                miss += 1
            if gn and not pn:
                fp += 1
            if g != p:
                wrong.append(r["file"])
        n = len(rows)
        return {"score": ok / tot, "y": fy / n, "m": fm / n, "d": fd / n,
                "exact": (n - len(wrong)) / n, "miss": miss, "fp": fp,
                "per": per, "n": n}

    def mcnemar(pa, pb):
        up = sum(1 for k in pa if pb.get(k, 0) > pa[k])
        dn = sum(1 for k in pa if pb.get(k, 0) < pa[k])
        if up + dn == 0:
            return up, dn, 1.0
        from math import comb
        n, k = up + dn, min(up, dn)
        return up, dn, min(1.0, sum(comb(n, i) for i in range(k + 1)) / 2 ** n * 2)

    CFG = [("off", None, "기준선 (det 만)"),
           ("always", THR, f"전패스 + YOLO 크롭 신뢰도 ≥ {THR}")]
    from concurrent.futures import ProcessPoolExecutor
    S, R = {}, {}
    for mode, thr, label in CFG:
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            rows = list(ex.map(work, [(f, mode, thr) for f in files], chunksize=2))
        S[label] = score(rows)
        R[label] = {r["file"]: "-".join((r["year"], r["month"], r["day"]))
                    for r in rows}
        print(f"  {label}: {S[label]['score']*100:.1f}%  ({time.time()-t0:.0f}s)",
              flush=True)

    base, var = CFG[0][2], CFG[1][2]
    so, sv = S[base], S[var]
    up, dn, pv = mcnemar(so["per"], sv["per"])
    d = (sv["score"] - so["score"]) * 100

    L = [f"# EDA 51 — 전패스 YOLO + 크롭 신뢰도 문턱 {THR}\n",
         "48단계에서 전패스 방식은 +2.4%p 를 얻었지만 정답 4장을 망가뜨렸다.",
         "YOLO 크롭의 오독이 올바른 det 판독을 이긴 것이다. 크롭에만 더 높은",
         "신뢰도를 요구해 그 4장을 지킬 수 있는지 본다.",
         "",
         f"문턱은 {THR} 하나만 미리 정해 놓고 쟀다. 여러 값을 쓸어보면 198장의",
         "잡음에 맞추는 것이 되기 때문이다.",
         "",
         "| 구성 | **부분점수** | year | month | day | 완전일치 | 미검출 | 오탐 |",
         "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for _, _, label in CFG:
        s = S[label]
        L.append(f"| {label} | **{s['score']*100:.1f}%** | {s['y']*100:.1f}% | "
                 f"{s['m']*100:.1f}% | {s['d']*100:.1f}% | {s['exact']*100:.1f}% | "
                 f"{s['miss']} | {s['fp']} |")
    L.append("")
    L.append("참고 — 48단계: 최후수단 66.2%(개선 4/악화 0), 전패스 66.8%(개선 8/악화 4)")
    L.append("")
    L.append(f"차이 **{d:+.1f}%p**, 개선 {up}장 / 악화 {dn}장, McNemar p={pv:.3f}")
    L.append("")
    L.append("## 기준선과 달라진 것 전부\n")
    L.append("| 파일 | 정답 | 기준선 | 변형 | |")
    L.append("| --- | --- | --- | --- | --- |")
    for f in files:
        if sv["per"][f] != so["per"][f]:
            mark = "개선" if sv["per"][f] > so["per"][f] else "악화"
            L.append(f"| `{f}` | {h2[f]} | {R[base][f]} | {R[var][f]} | {mark} |")
    L.append("")
    L.append("## 판정\n")
    if dn == 0 and d > 0:
        L.append(f"**채택.** 악화 0장으로 {d:+.1f}%p. 문턱이 퇴행을 막았다.")
    elif d > 2.4 and dn < 4:
        L.append(f"**채택.** 48단계 전패스(+2.4%p, 악화 4)보다 낫다.")
    else:
        L.append("**기각.** 문턱으로 퇴행을 막지 못했거나 개선분까지 함께 잃었다.")
        L.append("최후수단 방식(`yolo_mode=\"fallback\"`, +1.7%p, 악화 0)으로 간다.")

    with open(os.path.join(OUT_DIR, "51_yolo_strict.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print()
    print("\n".join(L[8:16]))
    print("\n→ eda/out/51_yolo_strict.md")


if __name__ == "__main__":
    main()
