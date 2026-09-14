"""
EDA 65단계 — 합친 파이프라인 실측 검증

64단계는 두 파이프라인의 **예측 파일을 사후에 합쳐서** 82.3% 를 얻었다.
실제로 한 프로세스에서 합쳐 돌렸을 때 그 수치가 나오는지는 별개 문제다.
구현이 끼어들면 달라질 수 있으므로 직접 돌려 확인한다.

세 구성을 같은 198장에서 잰다.
  RapidOCR 단독      — 보조 경로 없음 (64단계 예상 79.8%)
  합친 것            — RapidOCR 주, 실패 시 기존 (예상 82.3%)
  (기준) 기존 단독   — 57단계 스냅샷 66.8% 을 그대로 참조

속도는 단일 프로세스로 따로 잰다 (66단계).
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

IMG_DIR = os.path.join(ROOT, "images", "상품사진입니다")
OUT_DIR = os.path.join(ROOT, "eda", "out")
ONNX = os.path.join(ROOT, "weights", "yolo", "date_v1.onnx")

_STATE = None


def build(use_fallback):
    import pipeline as P
    import team2_rules
    import combined
    import ocrlib
    from rapidocr import RapidOCR

    fb = None
    if use_fallback:
        fb = P.DatePipeline(ocrlib.build_ocr(det_side=640, threads=1),
                            long_side=640, rotations=(0, 90),
                            adaptive=True, keep=None)
        fb.yolo = P.OnnxDateDetector(ONNX, threads=1)
        fb.yolo_conf = 0.25
        fb.yolo_mode = "always"
    return combined.CombinedPipeline(RapidOCR(), team2_rules, fb), combined


def work(args):
    global _STATE
    f, use_fb = args
    if _STATE is None or _STATE[0] != use_fb:
        _STATE = (use_fb,) + build(use_fb)
    _, pipe, _mod = _STATE
    fields = pipe.predict(os.path.join(IMG_DIR, f))
    return {"file": f, **fields}


def split3(v):
    if not v or v == "NONE":
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
    old = json.load(open(os.path.join(OUT_DIR, "57_env_snapshot.json"),
                         encoding="utf-8"))["pred"]

    def score(pred):
        ok = tot = miss = fp = 0
        fy = fm = fd = 0
        per, wrong = {}, 0
        for f in files:
            g, p = split3(h2[f]), pred[f]
            hit = sum(g[i] == p[i] for i in range(3))
            per[f] = hit
            fy += g[0] == p[0]; fm += g[1] == p[1]; fd += g[2] == p[2]
            ok += hit; tot += 3
            gn, pn = g == ("NONE",) * 3, p == ("NONE",) * 3
            miss += (pn and not gn)
            fp += (gn and not pn)
            wrong += (g != p)
        n = len(files)
        return {"score": ok / tot, "y": fy / n, "m": fm / n, "d": fd / n,
                "exact": (n - wrong) / n, "miss": miss, "fp": fp, "per": per}

    from concurrent.futures import ProcessPoolExecutor
    S, R = {}, {}
    R["기존 단독 (PP-OCRv3+v4 + YOLO)"] = {f: split3(old[f]) for f in files}
    for use_fb, label in ((False, "RapidOCR 단독"), (True, "합친 것 (RapidOCR 주 + 기존 보조)")):
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            rows = list(ex.map(work, [(f, use_fb) for f in files], chunksize=2))
        R[label] = {r["file"]: (r["year"], r["month"], r["day"]) for r in rows}
        print(f"  {label}: {score(R[label])['score']*100:.1f}%  ({time.time()-t0:.0f}s)",
              flush=True)
    for k in R:
        S[k] = score(R[k])

    base = "기존 단독 (PP-OCRv3+v4 + YOLO)"
    L = ["# EDA 65 — 합친 파이프라인 실측 검증\n",
         "64단계는 예측 파일을 **사후에** 합쳐 82.3% 를 얻었다. 한 프로세스에서",
         "실제로 합쳐 돌렸을 때 같은 수치가 나오는지 확인한다.",
         "",
         "| 구성 | **부분점수** | year | month | day | 완전일치 | 미검출 | 오탐 |",
         "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for k in (base, "RapidOCR 단독", "합친 것 (RapidOCR 주 + 기존 보조)"):
        s = S[k]
        L.append(f"| {k} | **{s['score']*100:.1f}%** | {s['y']*100:.1f}% | "
                 f"{s['m']*100:.1f}% | {s['d']*100:.1f}% | {s['exact']*100:.1f}% | "
                 f"{s['miss']} | {s['fp']} |")
    L.append("")
    L.append("64단계 사후 합성 예상값: RapidOCR 단독 79.8%, 합친 것 82.3%")
    L.append("")
    for k in ("RapidOCR 단독", "합친 것 (RapidOCR 주 + 기존 보조)"):
        up, dn, pv = mcnemar(S[base]["per"], S[k]["per"])
        d = (S[k]["score"] - S[base]["score"]) * 100
        L.append(f"- **{k}** — 기존 대비 {d:+.1f}%p, 개선 {up}장 / 악화 {dn}장, p={pv:.4f}")
    up, dn, pv = mcnemar(S["RapidOCR 단독"]["per"],
                         S["합친 것 (RapidOCR 주 + 기존 보조)"]["per"])
    d = (S["합친 것 (RapidOCR 주 + 기존 보조)"]["score"] - S["RapidOCR 단독"]["score"]) * 100
    L.append(f"- 보조 경로의 순효과 — {d:+.1f}%p, 개선 {up}장 / 악화 {dn}장, p={pv:.3f}")
    L.append("")

    Rc = R["합친 것 (RapidOCR 주 + 기존 보조)"]
    Rr = R["RapidOCR 단독"]
    diff = [f for f in files if Rc[f] != Rr[f]]
    L.append(f"## 보조 경로가 실제로 건진 이미지 — {len(diff)}장\n")
    if diff:
        L.append("| 파일 | 정답 | RapidOCR 단독 | 합친 것 | |")
        L.append("| --- | --- | --- | --- | --- |")
        for f in diff:
            g = split3(h2[f])
            hr = sum(g[i] == Rr[f][i] for i in range(3))
            hc = sum(g[i] == Rc[f][i] for i in range(3))
            mark = "개선" if hc > hr else ("악화" if hc < hr else "무변")
            L.append(f"| `{f}` | {h2[f]} | {'-'.join(Rr[f])} | {'-'.join(Rc[f])} | {mark} |")
    L.append("")

    json.dump({k: {f: "-".join(v[f]) for f in files} for k, v in R.items()},
              open(os.path.join(OUT_DIR, "65_combined_pred.json"), "w",
                   encoding="utf-8"), ensure_ascii=False, indent=1)
    with open(os.path.join(OUT_DIR, "65_combined_eval.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print()
    print("\n".join(L[4:16]))
    print("\n→ eda/out/65_combined_eval.md")


if __name__ == "__main__":
    main()
