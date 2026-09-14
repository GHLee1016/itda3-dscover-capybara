"""
EDA 48단계 — YOLO 날짜 검출기를 붙인 효과 측정

측정 대상은 **2차 홀드아웃 198장**이다. 이유가 있다.
  - YOLO 학습셋(130장)은 truth_v4 + 1차 홀드아웃에서만 뽑았다. 그 두 집합으로
    재면 학습 데이터를 다시 채점하는 셈이라 무조건 좋게 나온다.
  - 홀드아웃2 는 46단계에서 의도적으로 val 로만 돌렸으므로 학습에 없다.

판정 기준은 mAP 가 아니라 **부분점수**다. 지금까지 mAP·합성테스트·회귀테스트를
통과하고도 홀드아웃에서 떨어진 변경이 두 번 있었다(TAIL_PATTERNS 64.5→63.8,
크롭 TTA 68.4→62.5). 최종 지표로만 판단한다.

같은 이미지 집합을 on/off 로 돌리므로 McNemar 짝지은 검정이 가능하다.
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
WEIGHTS = os.path.join(ROOT, "yolo_runs", "date_v1", "weights", "best.pt")

_PIPE = None
_CONF = None


def work(args):
    """mode 에 따라 같은 이미지를 세 구성으로 돌린다 (off / fallback / always)."""
    global _PIPE, _CONF
    f, mode, conf = args
    if _PIPE is None or _CONF != (mode, conf):
        import torch
        torch.set_num_threads(1)          # 프로세스 4개와 스레드 경합 방지
        pipe = P.DatePipeline(ocrlib.build_ocr(det_side=640, threads=1),
                              long_side=640, rotations=(0, 90),
                              adaptive=True, keep=None)
        if mode != "off":
            from ultralytics import YOLO
            pipe.yolo = YOLO(WEIGHTS)
            pipe.yolo_conf = conf
            pipe.yolo_mode = mode
        _PIPE, _CONF = pipe, (mode, conf)
    t0 = time.time()
    try:
        fin, _, _ = _PIPE.predict_path(os.path.join(IMG_DIR, f))
    except Exception:
        fin = None
    r = P.to_row(os.path.splitext(f)[0], fin)
    r["file"] = f
    r["sec"] = time.time() - t0
    return r


def split3(v):
    if v is None or v == "NONE":
        return ("NONE", "NONE", "NONE")
    p = str(v).split("-")
    return tuple(p) if len(p) == 3 else ("NONE", "NONE", "NONE")


def score(rows, lab):
    ok = tot = miss = fp = 0
    fy = fm = fd = 0
    wrong, per = [], {}
    for r in rows:
        g, p = split3(lab[r["file"]]), (r["year"], r["month"], r["day"])
        hit = sum(g[i] == p[i] for i in range(3))
        per[r["file"]] = hit
        fy += g[0] == p[0]
        fm += g[1] == p[1]
        fd += g[2] == p[2]
        ok += hit
        tot += 3
        gnone, pnone = g == ("NONE",) * 3, p == ("NONE",) * 3
        if pnone and not gnone:
            miss += 1
        if gnone and not pnone:
            fp += 1
        if g != p:
            wrong.append((r["file"], lab[r["file"]], "-".join(p)))
    n = len(rows)
    return {"score": ok / tot, "y": fy / n, "m": fm / n, "d": fd / n,
            "exact": (n - len(wrong)) / n, "miss": miss, "fp": fp,
            "wrong": wrong, "n": n, "per": per,
            "sec": sum(r["sec"] for r in rows) / n}


def mcnemar(a, b):
    """칸 단위(연/월/일 각각)가 아니라 이미지 단위 부분점수로 비교한다."""
    up = sum(1 for k in a if b.get(k, 0) > a[k])
    dn = sum(1 for k in a if b.get(k, 0) < a[k])
    if up + dn == 0:
        return up, dn, 1.0
    # 이항검정 양측 (p=0.5)
    from math import comb
    n, k = up + dn, min(up, dn)
    p = sum(comb(n, i) for i in range(k + 1)) / (2 ** n) * 2
    return up, dn, min(1.0, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--conf", type=float, default=0.25)
    a = ap.parse_args()

    if not os.path.exists(WEIGHTS):
        print(f"가중치 없음: {WEIGHTS}")
        return
    h2 = json.load(open(os.path.join(OUT_DIR, "truth_holdout2.json"),
                        encoding="utf-8"))["labels"]
    files = list(h2)
    print(f"홀드아웃2 {len(files)}장, YOLO on/off 비교 (conf={a.conf})", flush=True)

    from concurrent.futures import ProcessPoolExecutor
    MODES = [("off", "기준선 (det 만)"),
             ("fallback", f"+ YOLO 최후수단 (conf {a.conf})"),
             ("always", f"+ YOLO 전패스 (conf {a.conf})")]
    S, R = {}, {}
    for m, label in MODES:
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            rows = list(ex.map(work, [(f, m, a.conf) for f in files], chunksize=2))
        S[m] = score(rows, h2)
        R[m] = {r["file"]: "-".join((r["year"], r["month"], r["day"])) for r in rows}
        print(f"  {label}: {S[m]['score']*100:.1f}%  ({time.time()-t0:.0f}s)", flush=True)

    so = S["off"]
    L = ["# EDA 48 — YOLO 날짜 검출기 효과 (홀드아웃2 198장)\n",
         "홀드아웃2 는 YOLO 학습에서 제외한 집합이다(46단계에서 val 로만 썼다).",
         "학습셋(truth_v4·1차 홀드아웃)으로 재면 학습 데이터를 다시 채점하는 셈이라",
         "의미가 없다.",
         "",
         "두 가지 붙이는 방식을 나란히 본다.",
         "",
         "- **최후수단**: 기존 경로가 전부 실패한 이미지에만 YOLO 를 켠다.",
         "  성공한 이미지에 닿지 않으므로 구조적으로 떨어질 수 없다.",
         "- **전패스**: 모든 패스에 YOLO 박스를 덧붙인다. 커버리지는 크지만",
         "  이미 맞추던 이미지의 후보군도 바뀐다 — 크롭 TTA 가 실패한 경로다.",
         "",
         "| 구성 | **부분점수** | year | month | day | 완전일치 | 미검출 | 오탐 | 장당 |",
         "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for m, label in MODES:
        s = S[m]
        L.append(f"| {label} | **{s['score']*100:.1f}%** | {s['y']*100:.1f}% | "
                 f"{s['m']*100:.1f}% | {s['d']*100:.1f}% | {s['exact']*100:.1f}% | "
                 f"{s['miss']} | {s['fp']} | {s['sec']:.2f}s |")
    L.append("")

    best, bestd = None, 0.0
    for m, label in MODES[1:]:
        s = S[m]
        d = (s["score"] - so["score"]) * 100
        up, dn, pv = mcnemar(so["per"], s["per"])
        L.append(f"**{label}** — 차이 {d:+.1f}%p, 개선 {up}장 / 악화 {dn}장, "
                 f"McNemar p={pv:.3f} → "
                 + ("유의하다." if pv < 0.05 else "유의하지 않다."))
        L.append(f"  장당 {s['sec']-so['sec']:+.2f}초, 500장 추정 {s['sec']*500:.0f}초 "
                 f"(제한 2400초의 {s['sec']*500/2400*100:.0f}%)")
        L.append("")
        if d > bestd:
            best, bestd = m, d

    for m, label in MODES[1:]:
        s = S[m]
        L.append(f"## {label} — 새로 맞춘 것 / 망친 것\n")
        L.append("| 파일 | 정답 | 기준선 | +YOLO | |")
        L.append("| --- | --- | --- | --- | --- |")
        for f in files:
            if s["per"][f] != so["per"][f]:
                mark = "개선" if s["per"][f] > so["per"][f] else "악화"
                L.append(f"| `{f}` | {h2[f]} | {R['off'][f]} | {R[m][f]} | {mark} |")
        L.append("")

    L.append("## 판정\n")
    if best is None:
        L.append("**기각.** 두 방식 모두 홀드아웃에서 기준선을 넘지 못했다.")
        L.append("`pipeline.yolo = None` 로 둔다. 합성·회귀 테스트를 통과하고도")
        L.append("홀드아웃에서 떨어진 변경이 이미 두 번 있었다(TAIL_PATTERNS, 크롭 TTA).")
        L.append("같은 기준을 적용한다.")
    else:
        lbl = dict(MODES)[best]
        L.append(f"**{lbl} 방식 채택** ({bestd:+.1f}%p).")
        L.append(f"`pipeline.yolo_mode = \"{best}\"` 로 설정한다.")

    with open(os.path.join(OUT_DIR, "48_yolo_eval.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print()
    print("\n".join(L[12:20]))
    print(f"\n→ eda/out/48_yolo_eval.md")


if __name__ == "__main__":
    main()
