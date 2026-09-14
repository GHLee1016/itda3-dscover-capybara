"""
EDA 56단계 — ONNX 백엔드로 홀드아웃2 재측정 (최종 판정)

48단계는 torch 백엔드로 쟀다. 그 경로는 속도 때문에 폐기했으므로(52~53),
실제로 제출할 ONNX 경로에서 같은 수치가 나오는지 다시 확인한다.
55단계에서 박스는 IoU 1.000 으로 일치했으니 같은 점수가 나와야 정상이다.
다르게 나오면 어딘가 틀린 것이다.

48단계(torch) 기준값
  기준선   64.5%
  최후수단 66.2%  (개선 4 / 악화 0)
  전패스   66.8%  (개선 8 / 악화 4)
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
ONNX = os.path.join(ROOT, "weights", "yolo", "date_v1.onnx")

_PIPE = None
_KEY = None


def work(args):
    global _PIPE, _KEY
    f, mode, thr = args
    if _PIPE is None or _KEY != (mode, thr):
        pipe = P.DatePipeline(ocrlib.build_ocr(det_side=640, threads=1),
                              long_side=640, rotations=(0, 90),
                              adaptive=True, keep=None)
        if mode != "off":
            # 워커마다 세션을 만든다. 프로세스가 4개이므로 스레드는 1로 묶는다.
            pipe.yolo = P.OnnxDateDetector(ONNX, threads=1)
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


def split3(v):
    if v is None or v == "NONE":
        return ("NONE",) * 3
    p = str(v).split("-")
    return tuple(p) if len(p) == 3 else ("NONE",) * 3


def mcnemar(pa, pb):
    up = sum(1 for k in pa if pb.get(k, 0) > pa[k])
    dn = sum(1 for k in pa if pb.get(k, 0) < pa[k])
    if up + dn == 0:
        return up, dn, 1.0
    from math import comb
    n, k = up + dn, min(up, dn)
    return up, dn, min(1.0, sum(comb(n, i) for i in range(k + 1)) / 2 ** n * 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    h2 = json.load(open(os.path.join(OUT_DIR, "truth_holdout2.json"),
                        encoding="utf-8"))["labels"]
    files = list(h2)

    def score(rows):
        ok = tot = miss = fp = 0
        fy = fm = fd = 0
        per, wrong = {}, 0
        for r in rows:
            g, p = split3(h2[r["file"]]), (r["year"], r["month"], r["day"])
            hit = sum(g[i] == p[i] for i in range(3))
            per[r["file"]] = hit
            fy += g[0] == p[0]; fm += g[1] == p[1]; fd += g[2] == p[2]
            ok += hit; tot += 3
            gn, pn = g == ("NONE",) * 3, p == ("NONE",) * 3
            miss += (pn and not gn)
            fp += (gn and not pn)
            wrong += (g != p)
        n = len(rows)
        return {"score": ok / tot, "y": fy / n, "m": fm / n, "d": fd / n,
                "exact": (n - wrong) / n, "miss": miss, "fp": fp, "per": per}

    CFG = [("off", None, "기준선 (det 만)"),
           ("fallback", None, "+ YOLO 최후수단"),
           ("always", None, "+ YOLO 전패스"),
           ("always", 0.90, "+ YOLO 전패스, 크롭 신뢰도 ≥ 0.90")]

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

    base = CFG[0][2]
    so = S[base]
    L = ["# EDA 56 — ONNX 백엔드 홀드아웃2 재측정 (최종 판정)\n",
         "48단계는 torch 백엔드로 쟀지만 그 경로는 속도 때문에 폐기했다(52~53).",
         "실제로 제출할 ONNX 경로에서 같은 수치가 나오는지 확인한다.",
         "55단계에서 박스가 IoU 1.000 으로 일치했으므로 같아야 정상이다.",
         "",
         "| 구성 | **부분점수** | year | month | day | 완전일치 | 미검출 | 오탐 |",
         "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for _, _, label in CFG:
        s = S[label]
        L.append(f"| {label} | **{s['score']*100:.1f}%** | {s['y']*100:.1f}% | "
                 f"{s['m']*100:.1f}% | {s['d']*100:.1f}% | {s['exact']*100:.1f}% | "
                 f"{s['miss']} | {s['fp']} |")
    L.append("")
    L.append("참고 — 48단계(torch): 기준선 64.5%, 최후수단 66.2%, 전패스 66.8%")
    L.append("")
    for _, _, label in CFG[1:]:
        s = S[label]
        up, dn, pv = mcnemar(so["per"], s["per"])
        d = (s["score"] - so["score"]) * 100
        L.append(f"- **{label}** — {d:+.1f}%p, 개선 {up}장 / 악화 {dn}장, "
                 f"McNemar p={pv:.3f}")
    L.append("")
    for _, _, label in CFG[1:]:
        s = S[label]
        L.append(f"## {label} — 기준선과 달라진 것\n")
        L.append("| 파일 | 정답 | 기준선 | 변형 | |")
        L.append("| --- | --- | --- | --- | --- |")
        for f in files:
            if s["per"][f] != so["per"][f]:
                mark = "개선" if s["per"][f] > so["per"][f] else "악화"
                L.append(f"| `{f}` | {h2[f]} | {R[base][f]} | {R[label][f]} | {mark} |")
        L.append("")

    with open(os.path.join(OUT_DIR, "56_onnx_holdout2.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n→ eda/out/56_onnx_holdout2.md")


if __name__ == "__main__":
    main()
