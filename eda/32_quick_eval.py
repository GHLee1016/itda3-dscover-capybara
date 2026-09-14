"""
EDA 32단계 — 정답 137장 빠른 재측정

파이프라인 규칙을 고칠 때마다 581장 전체를 돌리면 느리다.
정답이 있는 137장만 돌려 부분점수를 바로 본다.
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
_NEEDS_MORE = P.needs_more


def get_pipe(pair_retry=True, fb=None, crop_tta=True, pps=()):
    global _PIPE
    if _PIPE is None:
        _PIPE = P.DatePipeline(ocrlib.build_ocr(det_side=640, threads=1),
                               long_side=640, rotations=(0, 90),
                               adaptive=True, keep=None)
    if fb is not None:
        _PIPE.fallback_sides = fb
    _PIPE.crop_tta = crop_tta
    _PIPE.fallback_preprocs = [P.PREPROCS[x] for x in pps if x in P.PREPROCS]
    # 부터/까지 재시도 규칙 on/off — 같은 하네스에서 A/B 하기 위해
    P.needs_more = _NEEDS_MORE if pair_retry else (lambda c, a: False)
    return _PIPE


def work(args):
    f, pair_retry, fb, crop_tta, pps = args
    pipe = get_pipe(pair_retry, fb, crop_tta, pps)
    try:
        fin, cands, anc = pipe.predict_path(os.path.join(IMG_DIR, f))
    except Exception:
        fin, cands, anc = None, {}, set()
    r = P.to_row(os.path.splitext(f)[0], fin)
    r["file"] = f
    r["n_cands"] = len(cands)
    r["anchors"] = "|".join(sorted(anc))
    return r


def split3(v):
    if v is None or v == "NONE":
        return ("NONE", "NONE", "NONE")
    p = str(v).split("-")
    return tuple(p) if len(p) == 3 else ("NONE", "NONE", "NONE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tag", default="cur")
    ap.add_argument("--pair-retry", default="on", choices=["on", "off"])
    ap.add_argument("--fb", default="512,448,384", help="빈 문자열이면 재시도 없음")
    ap.add_argument("--crop-tta", default="on", choices=["on", "off"])
    ap.add_argument("--pp", default="", help="쉼표구분 전처리 이름 (마지막 대안으로만 적용)")
    a = ap.parse_args()

    t = json.load(open(os.path.join(OUT_DIR, "truth_v4.json"), encoding="utf-8"))
    lab = t["labels"]
    ev = set(t["membership"]["eval_random"])
    files = list(lab)
    fb = tuple(int(x) for x in a.fb.split(",") if x.strip())
    pr = a.pair_retry == "on"

    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        pps = tuple(x.strip() for x in a.pp.split(",") if x.strip())
        rs = list(ex.map(work, [(f, pr, fb, a.crop_tta == "on", pps) for f in files], chunksize=2))
    wall = time.time() - t0

    ok = tot = oke = tote = miss = 0
    wrong = []
    for r in rs:
        g = split3(lab[r["file"]])
        p = (r["year"], r["month"], r["day"])
        h = sum(g[i] == p[i] for i in range(3))
        ok += h
        tot += 3
        if r["file"] in ev:
            oke += h
            tote += 3
        if p == ("NONE",) * 3 and g != ("NONE",) * 3:
            miss += 1
        if h < 3:
            wrong.append((r["file"], lab[r["file"]], "-".join(p), r["n_cands"], r["anchors"]))

    print(f"[{a.tag}]")
    print(f"  전체 137장 부분점수  {ok/tot*100:.1f}%")
    print(f"  무작위 43장          {oke/max(tote,1)*100:.1f}%")
    print(f"  미검출               {miss}장")
    print(f"  틀린 장수            {len(wrong)}")
    print(f"  속도                 {wall/len(files):.2f}초/장 -> 500장 {wall/len(files)*500:.0f}초")

    import csv
    with open(os.path.join(OUT_DIR, f"quick_{a.tag}.csv"), "w", encoding="utf-8-sig",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "truth", "pred", "n_cands", "anchors"])
        w.writerows(wrong)


if __name__ == "__main__":
    main()
