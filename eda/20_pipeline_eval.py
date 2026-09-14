"""
EDA 20단계 — src/pipeline.py end-to-end 검증

사람 정답 34장에 대한 정확도와, 무작위 표본에 대한 실제 처리속도를 함께 잰다.
속도는 채점과 같은 조건(4 프로세스 병렬)으로 wall-clock 을 측정한다.
"""
import os
import sys
import json
import time
import argparse
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "eda"))
import pipeline as P
import ocrlib

IMG_DIR = os.path.join(ROOT, "images", "상품사진입니다")
OUT_DIR = os.path.join(ROOT, "eda", "out")

_PIPE = {}


def get_pipe(adaptive=True, keep=0.20):
    key = (adaptive, keep)
    if key not in _PIPE:
        ocr = ocrlib.build_ocr(det_side=960, threads=1)
        _PIPE[key] = P.DatePipeline(ocr, long_side=960, rotations=(0, 90),
                                    adaptive=adaptive, keep=keep)
    return _PIPE[key]


def work(args):
    path, adaptive, keep = args[0], args[1], (args[2] if len(args) > 2 else 0.20)
    pipe = get_pipe(adaptive, keep)
    t = time.time()
    final, cands, anc = pipe.predict_path(path)
    return {"file": os.path.basename(path), "final": final,
            "cands": sorted(cands), "anchors": sorted(anc),
            "sec": time.time() - t}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed-n", type=int, default=200)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--keeps", default="none,0.30,0.20,0.10")
    ap.add_argument("--out", default="20_pipeline_eval.md")
    a = ap.parse_args()

    truth = json.load(open(os.path.join(OUT_DIR, "truth_v2.json"), encoding="utf-8"))
    from concurrent.futures import ProcessPoolExecutor

    keeps = [None if k == "none" else float(k) for k in a.keeps.split(",")]

    # ---------------- 정확도 (필터 설정별) ----------------
    files = [os.path.join(IMG_DIR, f) for f in truth]
    acc_table = {}
    for k in keeps:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            rs = list(ex.map(work, [(f, True, k) for f in files], chunksize=1))
        h = sum(1 for r in rs if r["final"] == truth[r["file"]])
        nn = sum(1 for r in rs if r["final"] is None)
        wr = [(r["file"], truth[r["file"]], r["final"], r["cands"])
              for r in rs if r["final"] is not None and r["final"] != truth[r["file"]]]
        acc_table[k] = {"hit": h, "none": nn, "wrong": wr, "res": rs}
        print(f"  keep={k}: {h}/{len(rs)}", flush=True)

    base_k = keeps[0]
    acc_res = acc_table[base_k]["res"]
    hit = acc_table[base_k]["hit"]
    none_cnt = acc_table[base_k]["none"]
    wrong = acc_table[base_k]["wrong"]
    n = len(acc_res)

    # ---------------- 속도 ----------------
    meta = pd.read_csv(os.path.join(OUT_DIR, "metadata.csv"))
    rng = np.random.default_rng(99)
    pool = [f for f in meta["file"] if f not in truth]
    sam = [os.path.join(IMG_DIR, f) for f in rng.choice(pool, a.speed_n, replace=False)]

    speed = {}
    for k in keeps:
        label = "필터 없음" if k is None else f"필터 상위 {int(k*100)}%"
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            rs = list(ex.map(work, [(f, True, k) for f in sam], chunksize=2))
        wall = time.time() - t0
        found = sum(1 for r in rs if r["final"])
        speed[label] = {"wall": wall, "per": wall / len(sam),
                        "cov": found / len(sam), "keep": k}
        print(f"  speed keep={k}: {wall/len(sam):.3f}s/img", flush=True)

    L = ["# EDA 20 — 파이프라인 end-to-end 검증\n",
         f"- 정확도: 사람 정답 **{n}장**",
         f"- 속도: 무작위 **{len(sam)}장**, {a.workers} 프로세스 병렬 wall-clock",
         "- CPU: AMD Ryzen 5 7500F (채점용 4 vCPU 보다 빠름 → 낙관적 하한)",
         "",
         "## 정확도 — 박스 필터 설정별\n",
         "| 필터 | 정확히 맞춤 | NONE | 틀린 날짜 |",
         "| --- | ---: | ---: | ---: |"]
    for k in keeps:
        lab = "없음 (전부 rec)" if k is None else f"상위 {int(k*100)}%"
        v = acc_table[k]
        L.append(f"| {lab} | **{v['hit']}/{n} ({v['hit']/n*100:.1f}%)** | "
                 f"{v['none']} | {len(v['wrong'])} |")
    L.append("")
    if wrong:
        L.append("### 틀린 출력\n")
        L.append("| 파일 | 정답 | 출력 | 후보 |")
        L.append("| --- | --- | --- | --- |")
        for f, gt, got, c in wrong:
            L.append(f"| `{f}` | {gt} | **{got}** | {c} |")
        L.append("")
    L.append("### NONE 출력한 이미지\n")
    nones = [r["file"] for r in acc_res if r["final"] is None]
    L.append(f"{len(nones)}장: `{', '.join(nones)}`")
    L.append("")

    L.append("## 속도 (타임아웃 2,400초 / 3,352장 = 0.72 초/장)\n")
    L.append("| 필터 | 초/장 | 3,352장 환산 | 타임아웃 대비 | coverage |")
    L.append("| --- | ---: | ---: | ---: | ---: |")
    for k, v in speed.items():
        total = v["per"] * 3352
        mark = "✅" if total < 2400 else "❌"
        L.append(f"| {k} | **{v['per']:.3f}** | {total:.0f}초 {mark} | "
                 f"{total/2400*100:.0f}% | {v['cov']*100:.1f}% |")
    L.append("")
    ok = [(lab, v) for lab, v in speed.items() if v["per"] * 3352 < 2400]
    if ok:
        lab, v = max(ok, key=lambda t: (t[1]["keep"] or 0))
        tot = v["per"] * 3352
        L.append(f"> ✅ **`{lab}` 로 3,352장 {tot:.0f}초** — 타임아웃까지 "
                 f"여유 {2400-tot:.0f}초 ({2400/tot:.2f}배).")
        if 2400 / tot < 1.5:
            L.append(">")
            L.append("> ⚠️ 여유가 1.5배 미만이다. 채점 CPU 가 더 느리면 위험하므로 "
                     "더 공격적인 필터를 택하는 편이 안전하다.")
    else:
        L.append("> ❌ **모든 설정이 타임아웃을 초과한다.**")
    L.append("")
    L.append("> ⚠️ 측정 CPU 가 채점 환경보다 빠르므로 실제로는 더 느릴 수 있다. "
             "여유가 2배 이상 아니면 안심할 수 없다.")

    with open(os.path.join(OUT_DIR, a.out), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
