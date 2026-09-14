"""
EDA 61단계 — 이식한 팀원 규칙의 홀드아웃 판정

60단계 단위 검사는 10/10 통과했지만 그것으로 채택하지 않는다.
TAIL_PATTERNS 는 합성 10/11 + 회귀 24/24 를 통과하고도 홀드아웃에서
64.5% → 63.8% 로 떨어졌다. 판정은 언제나 홀드아웃 부분점수로만 한다.

네 구성을 같은 198장에서 돌려 짝지어 비교한다.
  기준선        — 두 규칙 모두 끔 (66.8%)
  + 바코드      — EAN-13 국가 접두어로 2자리 연도 순서 판별
  + 시각 제외   — 2자리 연도 날짜 뒤 '시:분' 이면 제조 시각으로 보고 버림
  + 둘 다
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
    f, bc, te = args
    key = (bc, te)
    if _PIPE is None or _KEY != key:
        pipe = P.DatePipeline(ocrlib.build_ocr(det_side=640, threads=1),
                              long_side=640, rotations=(0, 90),
                              adaptive=True, keep=None)
        pipe.yolo = P.OnnxDateDetector(ONNX, threads=1)
        pipe.yolo_conf = 0.25
        pipe.yolo_mode = "always"
        _PIPE, _KEY = pipe, key
    # 모듈 전역 플래그라 워커마다 매 호출 앞에서 맞춰 준다
    P.ENABLE_BARCODE_HINT = bc
    P.ENABLE_TIME_EXCLUDE = te
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

    CFG = [(False, False, "기준선 (두 규칙 끔)"),
           (True,  False, "+ 바코드 국가 힌트"),
           (False, True,  "+ 날짜 뒤 시각 제외"),
           (True,  True,  "+ 둘 다")]

    from concurrent.futures import ProcessPoolExecutor
    S, R = {}, {}
    for bc, te, label in CFG:
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            rows = list(ex.map(work, [(f, bc, te) for f in files], chunksize=2))
        S[label] = score(rows)
        R[label] = {r["file"]: "-".join((r["year"], r["month"], r["day"]))
                    for r in rows}
        print(f"  {label}: {S[label]['score']*100:.1f}%  ({time.time()-t0:.0f}s)",
              flush=True)

    base = CFG[0][2]
    so = S[base]
    L = ["# EDA 61 — 팀원 규칙 이식의 홀드아웃 판정\n",
         "60단계 단위 검사는 10/10 통과했지만 그것으로 채택하지 않는다.",
         "TAIL_PATTERNS 는 합성 10/11 + 회귀 24/24 를 통과하고도 홀드아웃에서",
         "64.5% → 63.8% 로 떨어졌다. 판정은 홀드아웃 부분점수로만 한다.",
         "",
         "| 구성 | **부분점수** | year | month | day | 완전일치 | 미검출 | 오탐 |",
         "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for _, _, label in CFG:
        s = S[label]
        L.append(f"| {label} | **{s['score']*100:.1f}%** | {s['y']*100:.1f}% | "
                 f"{s['m']*100:.1f}% | {s['d']*100:.1f}% | {s['exact']*100:.1f}% | "
                 f"{s['miss']} | {s['fp']} |")
    L.append("")
    for _, _, label in CFG[1:]:
        s = S[label]
        up, dn, pv = mcnemar(so["per"], s["per"])
        d = (s["score"] - so["score"]) * 100
        L.append(f"- **{label}** — {d:+.1f}%p, 개선 {up}장 / 악화 {dn}장, p={pv:.3f}")
    L.append("")
    for _, _, label in CFG[1:]:
        s = S[label]
        diff = [f for f in files if s["per"][f] != so["per"][f]]
        L.append(f"## {label} — 기준선과 달라진 것 ({len(diff)}장)\n")
        if diff:
            L.append("| 파일 | 정답 | 기준선 | 변경 후 | |")
            L.append("| --- | --- | --- | --- | --- |")
            for f in diff:
                mark = "개선" if s["per"][f] > so["per"][f] else "악화"
                L.append(f"| `{f}` | {h2[f]} | {R[base][f]} | {R[label][f]} | {mark} |")
        else:
            L.append("변화 없음 — 이 표본에서는 규칙이 한 번도 발동하지 않았거나,")
            L.append("발동했어도 최종 선택을 바꾸지 못했다.")
        L.append("")

    with open(os.path.join(OUT_DIR, "61_rules_holdout.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print()
    print("\n".join(L[5:16]))
    print("\n→ eda/out/61_rules_holdout.md")


if __name__ == "__main__":
    main()
