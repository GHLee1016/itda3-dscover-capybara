"""
EDA 104단계 — 홀드아웃5 판정: 인식기 칸을 채택할 것인가

라벨 100장을 새로 만들어(103단계) 여기서만 판정한다. 홀드아웃4 는 이미
두 번 썼으므로(최종 판정 · 1600px 칸 기각) 더 쓰지 않는다.

비교 구성
  A_현재      계단 4칸 (det small → adaptive → det medium → CLAHE+medium)
  B_rec추가   + rec 한국어 → rec 영문

왜 이 두 칸인가 (EDA 101~102)
  주력 인식기 PP-OCRv6_rec_small 은 중국어 모델이라 한글을 제대로 읽지 못한다.
  한글이 날짜에 바로 붙어 있으면(`나 2026.03.11 까지 홍병현`) 그 글자를 엉뚱한
  한자로 읽으면서 날짜까지 함께 깨뜨린다. 580장에서 개선 6 / 악화 1(오탐)이었다.

  ⚠️ 1600px 칸도 580장에서 +0.6%p 로 보였다가 홀드아웃4 에서 0장이었다.
     그래서 580장 수치만으로는 채택하지 않는다.

    python eda/104_holdout5_eval.py --workers 4
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
_K = None


def build(mode):
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

    ko = en = None
    if mode == "B_rec추가":
        ko, _ = combined.build_rapid_rec(RAPID_DIR, combined.REC_KOREAN)
        en, _ = combined.build_rapid_rec(RAPID_DIR, combined.REC_EN)
    return combined.CombinedPipeline(rapid, team2_rules, fb, rapid_retry=retry,
                                     rec_korean=ko, rec_en=en)


def work(args):
    global _P, _K
    f, mode = args
    if _P is None or _K != mode:
        _P, _K = build(mode), mode
    t0 = time.time()
    fields = _P.predict(os.path.join(IMG_DIR, f))
    return {"file": f, "sec": round(time.time() - t0, 3),
            "pred": "-".join(fields[k] for k in ("year", "month", "day"))}


def s3(v):
    if not v or v == "NONE":
        return ("NONE",) * 3
    q = str(v).split("-")
    return tuple(q) if len(q) == 3 else ("NONE",) * 3


def mcnemar(pa, pb):
    up = sum(1 for k in pa if pb.get(k, 0) > pa[k])
    dn = sum(1 for k in pa if pb.get(k, 0) < pa[k])
    if up + dn == 0:
        return up, dn, 1.0
    from math import comb
    n, k = up + dn, min(up, dn)
    return up, dn, min(1.0, sum(comb(n, i) for i in range(k + 1)) / 2 ** n * 2)


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

    lab = json.load(open(os.path.join(OUT, "truth_holdout5.json"),
                         encoding="utf-8"))["labels"]
    files = [f for f in sorted(lab) if os.path.exists(os.path.join(IMG_DIR, f))]
    print(f"홀드아웃5 {len(files)}장 × 2구성", flush=True)

    MODES = ["A_현재", "B_rec추가"]
    from concurrent.futures import ProcessPoolExecutor
    out = {}
    for mode in MODES:
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            rows = list(ex.map(work, [(f, mode) for f in files], chunksize=2))
        out[mode] = {r["file"]: r for r in rows}
        print(f"  {mode:10s} {time.time()-t0:.0f}초", flush=True)

    json.dump({m: {f: r["pred"] for f, r in d.items()} for m, d in out.items()},
              open(os.path.join(OUT, "104_holdout5.json"), "w",
                   encoding="utf-8"), ensure_ascii=False, indent=1)

    def score(mode):
        d = out[mode]
        ok = tot = miss = fp = exact = 0
        fy = fm = fd = 0
        per = {}
        for f in files:
            g, p = s3(lab[f]), s3(d[f]["pred"])
            hit = sum(g[i] == p[i] for i in range(3))
            per[f] = hit
            fy += g[0] == p[0]; fm += g[1] == p[1]; fd += g[2] == p[2]
            ok += hit; tot += 3
            gn, pn = g == ("NONE",) * 3, p == ("NONE",) * 3
            miss += (pn and not gn)
            fp += (gn and not pn)
            exact += g == p
        n = len(files)
        return {"score": ok / tot, "ok": ok, "tot": tot,
                "y": fy / n, "m": fm / n, "d": fd / n,
                "exact": exact, "exact_r": exact / n, "miss": miss, "fp": fp,
                "per": per, "sec": sum(d[f]["sec"] for f in files) / n}

    S = {m: score(m) for m in MODES}
    up, dn, pv = mcnemar(S["A_현재"]["per"], S["B_rec추가"]["per"])
    diff = (S["B_rec추가"]["score"] - S["A_현재"]["score"]) * 100
    lo, hi = wilson(S["B_rec추가"]["ok"], S["B_rec추가"]["tot"])

    L = ["# EDA 104 — 홀드아웃5 판정: 인식기 칸을 채택할 것인가\n",
         "라벨 100장을 새로 만들어 여기서만 판정한다. 홀드아웃4 는 이미 두 번",
         "썼으므로(최종 판정 · 1600px 칸 기각) 더 쓰지 않는다.",
         "",
         f"- 무작위 {len(files)}장 (103단계에서 남은 2,619장에서 뽑음)",
         "",
         "| 구성 | **필드 정확도** | year | month | day | 완전일치 | 미검출 | 오탐 | 장당(병렬) |",
         "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    NAME = {"A_현재": "계단 4칸 (현재)", "B_rec추가": "**+ rec 한국어 · rec 영문**"}
    for m in MODES:
        s = S[m]
        L.append(f"| {NAME[m]} | **{s['score']*100:.1f}%** | {s['y']*100:.1f}% | "
                 f"{s['m']*100:.1f}% | {s['d']*100:.1f}% | {s['exact_r']*100:.1f}% | "
                 f"{s['miss']} | {s['fp']} | {s['sec']:.2f}초 |")
    L.append("")
    L.append(f"차이 **{diff:+.1f}%p** (개선 {up}장 / 악화 {dn}장, McNemar p={pv:.4f})")
    L.append("")

    # 달라진 장을 모두 적는다
    changed = [f for f in files
               if out["A_현재"][f]["pred"] != out["B_rec추가"][f]["pred"]]
    L.append(f"## 달라진 {len(changed)}장\n")
    if changed:
        L.append("| 파일 | 정답 | 현재 | rec 추가 | |")
        L.append("| --- | --- | --- | --- | --- |")
        for f in changed:
            a_, b_ = S["A_현재"]["per"][f], S["B_rec추가"]["per"][f]
            mark = "개선" if b_ > a_ else ("악화" if b_ < a_ else "무변")
            L.append(f"| `{f}` | {lab[f]} | {out['A_현재'][f]['pred']} | "
                     f"{out['B_rec추가'][f]['pred']} | {mark} |")
    else:
        L.append("없음. 인식기를 바꿔도 결과가 하나도 달라지지 않았다.")
    L.append("")

    L.append("## 판단\n")
    if diff > 0 and dn == 0:
        L.append(f"**채택.** {diff:+.1f}%p 이고 악화가 0장이다.")
    elif diff > 0:
        L.append(f"{diff:+.1f}%p 이지만 악화 {dn}장이 있다. 순이득은 {up-dn}장이다.")
        if pv < 0.05:
            L.append("McNemar p<0.05 로 우연으로 보기 어렵다. 채택할 만하다.")
        else:
            L.append(f"다만 p={pv:.3f} 로 표본 {len(files)}장에서는 우연과 구별되지 않는다.")
    elif diff == 0:
        L.append("**차이가 없다.** 580장에서 보이던 이득이 새 표본에서 사라졌다.")
        L.append("1600px 칸과 같은 경우다. 비용만 늘므로 채택하지 않는다.")
    else:
        L.append(f"**{diff:+.1f}%p 로 오히려 나빠졌다.** 채택하지 않는다.")
    L.append("")
    L.append(f"필드 정확도 {S['B_rec추가']['score']*100:.1f}% 의 95% 신뢰구간은 "
             f"{lo*100:.1f}~{hi*100:.1f}% 다. 표본 {len(files)}장이라 폭이 넓다.")

    with open(os.path.join(OUT, "104_holdout5_eval.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L))
    print()
    print("\n".join(L[5:]))
    print("\n→ eda/out/104_holdout5_eval.md")


if __name__ == "__main__":
    main()
