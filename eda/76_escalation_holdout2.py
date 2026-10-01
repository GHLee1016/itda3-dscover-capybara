"""
EDA 76단계 — 검출기 계단을 붙이고 홀드아웃2 로 최종 판정

경위
  69  전량 3,164장 실행 → 실패 271장(8.6%). 원인 분류가 거칠어 방향을 두 번 잘못 잡았다
  73  해상도를 올려도 19장(7.0%)만 회수
  74  **검출기를 medium 으로 키우니 69장(27.4%) 회수.** 인식기 쪽 전부보다 컸다
  75  라벨 179장에서 85.1% → 89.4%, 악화 2장. 오탐은 3장 그대로
  76  악화 2장을 없애기 위해 **계단 구조**로 붙이고 여기서 최종 판정   ← 지금

계단 구조
    주력(det small @1024) → adaptive 재시도 → **det medium 재시도** → 보조(PP-OCR+YOLO)
  재시도는 이미 실패한 이미지에만 돈다. 맞히던 것을 건드릴 수 없으므로
  회귀가 구조적으로 0 이다.

⚠️ 홀드아웃2 198장은 이 실험 전까지 이번 라운드에서 한 번도 쓰지 않았다.
   69~75단계는 모두 홀드아웃2 를 제외한 집합에서 돌렸다.

    python eda/76_escalation_holdout2.py --workers 4
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
    retry = retry2 = None
    if mode in ("escalation", "escalation2"):
        retry, _src = combined.build_rapid_retry(RAPID_DIR)
    if mode == "escalation2":
        # det server 칸은 이 실험에서 기각됐고, 최종 combined.py 에서 빠졌다.
        # 재현을 위해 rejected_rungs 에 근거와 함께 남겨 두었다.
        import rejected_rungs
        retry2, _src2 = rejected_rungs.build_rapid_retry2(RAPID_DIR)
    pipe = combined.CombinedPipeline(rapid, team2_rules, fb, rapid_retry=retry)
    if retry2 is not None:
        # 기각된 칸을 이 스크립트 안에서만 되살린다. 최종 구성에는 없다.
        pipe.retry2 = retry2
        pipe.stats.setdefault("retry2", 0)
        _attach_retry2(pipe)
    return pipe


def _attach_retry2(pipe):
    """
    기각된 det server 칸을 런타임에 덧붙인다 (EDA 76 재현 전용).

    최종 `combined.py` 는 이 칸을 갖고 있지 않다. 여기서는 5·6칸 앞에
    끼워 넣던 당시 순서를 재현하기 위해 `_full` 을 감싼다.
    """
    base_rapid = pipe._rapid

    def patched(img):
        got = base_rapid(img)
        if got is not None and all(
                str(got.get(k, "NONE")) != "NONE" for k in ("year", "month", "day")):
            return got
        texts = pipe._texts(combined.downscale_max(img, pipe.MAX_SIDE), pipe.retry2)
        alt = pipe._full(texts)
        if alt is not None:
            pipe.stats["retry2"] += 1
            return alt
        return got

    pipe._rapid = patched


def work(args):
    global _P, _K
    f, mode = args
    if _P is None or _K != mode:
        _P, _K = build(mode), mode
    t0 = time.time()
    fields = _P.predict(os.path.join(IMG_DIR, f))
    return {"file": f, "sec": round(time.time() - t0, 3),
            "pred": "-".join(fields[k] for k in ("year", "month", "day")),
            "stats": dict(_P.stats)}


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    h2 = json.load(open(os.path.join(OUT_DIR, "truth_holdout2.json"),
                        encoding="utf-8"))["labels"]
    files = [f for f in h2 if os.path.exists(os.path.join(IMG_DIR, f))]
    print(f"홀드아웃2 {len(files)}장 × 2구성", flush=True)

    from concurrent.futures import ProcessPoolExecutor
    out = {}
    for mode in ("current", "escalation", "escalation2"):
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            rows = list(ex.map(work, [(f, mode) for f in files], chunksize=2))
        out[mode] = {r["file"]: r for r in rows}
        print(f"  {mode:11s} {time.time()-t0:.0f}초", flush=True)

    json.dump({m: {f: r["pred"] for f, r in d.items()} for m, d in out.items()},
              open(os.path.join(OUT_DIR, "76_escalation.json"), "w",
                   encoding="utf-8"), ensure_ascii=False, indent=1)

    def score(mode):
        d = out[mode]
        ok = tot = miss = fp = 0
        fy = fm = fd = 0
        per, exact = {}, 0
        for f in files:
            g, p = s3(h2[f]), s3(d[f]["pred"])
            hit = sum(g[i] == p[i] for i in range(3))
            per[f] = hit
            fy += g[0] == p[0]; fm += g[1] == p[1]; fd += g[2] == p[2]
            ok += hit; tot += 3
            gn, pn = g == ("NONE",) * 3, p == ("NONE",) * 3
            miss += (pn and not gn)
            fp += (gn and not pn)
            exact += g == p
        n = len(files)
        return {"score": ok / tot, "y": fy / n, "m": fm / n, "d": fd / n,
                "exact": exact / n, "miss": miss, "fp": fp, "per": per,
                "sec": sum(d[f]["sec"] for f in files) / n}

    S = {m: score(m) for m in out}
    up, dn, pv = mcnemar(S["current"]["per"], S["escalation2"]["per"])
    diff = (S["escalation2"]["score"] - S["current"]["score"]) * 100

    NAME = {"current": "기준선 (주력 → 보조)",
            "escalation": "계단1 (+ det medium 재시도)",
            "escalation2": "**계단2** (+ det medium + det server)"}

    L = ["# EDA 76 — 검출기 계단을 붙이고 홀드아웃2 로 최종 판정\n",
         "69~75단계는 모두 홀드아웃2 를 제외한 집합에서 돌렸다. 이 표본은 이번",
         "라운드에서 여기 처음 쓴다.",
         "",
         "| 구성 | **필드 정확도** | year | month | day | 완전일치 | 미검출 | 오탐 | 장당 |",
         "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for m in ("current", "escalation", "escalation2"):
        s = S[m]
        L.append(f"| {NAME[m]} | **{s['score']*100:.1f}%** | {s['y']*100:.1f}% | "
                 f"{s['m']*100:.1f}% | {s['d']*100:.1f}% | {s['exact']*100:.1f}% | "
                 f"{s['miss']} | {s['fp']} | {s['sec']:.2f}초 |")
    L.append("")
    L.append(f"차이 **{diff:+.1f}%p** — 개선 {up}장 / 악화 {dn}장, McNemar p={pv:.3f}")
    L.append("")
    if dn == 0:
        L.append("**악화 0장.** 재시도가 이미 실패한 이미지에만 돌기 때문에 구조적으로 그렇다.")
    else:
        L.append(f"⚠️ 악화 {dn}장이 나왔다. 계단 구조에서는 원래 0 이어야 하므로 원인을 봐야 한다.")
    L.append("")

    # 경로 분담
    last = out["escalation2"][files[-1]]["stats"]
    L.append("## 경로 분담 (계단 구성, 워커 1개 기준 누적)\n")
    L.append("| 경로 | 장수 |")
    L.append("| --- | ---: |")
    KO = {"rapid": "주력 1차", "adaptive": "주력 2차(adaptive)",
          "retry": "**det medium 재시도**", "retry2": "**det server 재시도**",
          "fallback": "보조 경로", "none": "실패"}
    for k in ("rapid", "adaptive", "retry", "retry2", "fallback", "none"):
        L.append(f"| {KO[k]} | {last.get(k, 0)} |")
    L.append("")
    L.append("> 워커가 여러 개라 합계가 198 이 아니다. 비율만 참고한다.")
    L.append("")

    changed = [f for f in files if out["current"][f]["pred"] != out["escalation2"][f]["pred"]]
    L.append(f"## 달라진 {len(changed)}장\n")
    if changed:
        L.append("| 파일 | 정답 | 현재 | 계단 추가 | |")
        L.append("| --- | --- | --- | --- | --- |")
        for f in changed:
            a_, b_ = S["current"]["per"][f], S["escalation2"]["per"][f]
            mark = "개선" if b_ > a_ else ("악화" if b_ < a_ else "무변")
            L.append(f"| `{f}` | {h2[f]} | {out['current'][f]['pred']} | "
                     f"{out['escalation2'][f]['pred']} | {mark} |")
    L.append("")

    sec = S["escalation2"]["sec"]
    L.append("## 속도\n")
    L.append(f"- 장당 {S['current']['sec']:.2f}초 → **{sec:.2f}초**")
    L.append(f"- 500장 환산 {sec*500:.0f}초, 제한 2,400초의 **{sec*500/2400*100:.0f}%**")
    L.append("")
    L.append("## 판단\n")
    if diff > 0 and dn == 0:
        L.append(f"**채택.** {diff:+.1f}%p 이고 악화가 0 장이다. 재시도가 실패분에만 돌아")
        L.append("맞히던 것을 건드리지 않는다. 비용도 실패한 이미지에만 든다.")
    elif diff > 0:
        L.append(f"{diff:+.1f}%p 이지만 악화 {dn}장이 있다. 원인을 확인하고 판단한다.")
    else:
        L.append(f"{diff:+.1f}%p. 홀드아웃2 에서는 이득이 확인되지 않는다. 채택하지 않는다.")
    L.append("")
    L.append("> ⚠️ 배포하려면 det medium 모델을 weights/ 에 동봉하고")
    L.append("> download_weights.sh 에 추가해야 한다. 채점 환경은 오프라인이다.")

    with open(os.path.join(OUT_DIR, "76_escalation_holdout2.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L))
    print()
    print("\n".join(L[3:14]))
    print("\n→ eda/out/76_escalation_holdout2.md")


if __name__ == "__main__":
    main()
