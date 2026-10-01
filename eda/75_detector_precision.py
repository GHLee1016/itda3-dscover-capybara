"""
EDA 75단계 — 검출기 medium 의 정밀도 검증 (B 적용 전 안전 점검)

74단계에서 B_det_medium 이 어려운 실패 252장 중 69장(27.4%)을 회수했다.
그러나 그 252장 중 라벨이 있는 것이 **4장뿐**이라 '회수한 값이 맞는가'를
사실상 확인하지 못했다. 검출기를 키우면 바코드·영양성분 같은 숫자도 더 많이
잡히므로, 회수가 곧 개선이라는 보장이 없다.

여기서 재는 것은 두 가지다.
  1) 미검출이 줄어드는가
  2) **이미 맞히던 것을 망가뜨리지 않는가**  ← 이쪽이 더 중요하다

측정 집합: 튜닝셋 137장 + 1차 홀드아웃 48장 = **라벨 있는 185장**.
   홀드아웃2 198장은 최종 판정용으로 계속 동결한다. 이 185장은 이미 설계에
   쓰여 낙관 편향이 있지만, 여기서 보려는 것은 절대 성능이 아니라
   **기준선 대비 회귀 여부**라 목적에 맞는다.

설정을 셋으로 갈라 검출기 변경과 해상도 변경을 분리한다.
   A  det small  @1024   현재 운영 구성
   B1 det medium @1024   검출기만 바꿈
   B2 det medium @1536   검출기 + 해상도

    python eda/75_detector_precision.py --workers 4
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

_OCR = None
_KEY = None


def cfgs():
    from rapidocr.utils.typings import ModelType as MT, LangDet as LD, OCRVersion as OV
    med = {"Det.lang_type": LD.CH, "Det.model_type": MT.MEDIUM,
           "Det.ocr_version": OV.PPOCRV6}
    return {
        "A_small_1024":  ({}, 1024),
        "B1_medium_1024": (med, 1024),
        "B2_medium_1536": (med, 1536),
    }


def work(args):
    global _OCR, _KEY
    fname, key = args
    import cv2
    import combined
    import team2_rules as rules
    from rapidocr import RapidOCR

    params, side = cfgs()[key]
    if _OCR is None or _KEY != key:
        _OCR = RapidOCR(params=params)
        _KEY = key

    img = combined.imread_any(os.path.join(IMG_DIR, fname))
    if img is None:
        return {"file": fname, "pred": "NONE-NONE-NONE", "sec": 0.0}
    h, w = img.shape[:2]
    if max(h, w) > side:
        s = side / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

    t0 = time.time()
    try:
        r = _OCR(img)
        texts = list(r.txts) if getattr(r, "txts", None) else []
    except Exception:
        texts = []
    got = None
    if texts:
        full = rules.choose_expiration_date(texts)
        if full:
            y, m, d = full.split("-")
            got = {"year": y, "month": m, "day": d}
        else:
            got = rules.extract_partial_date(texts)
    pred = "-".join(str(got.get(k, "NONE") or "NONE")
                    for k in ("year", "month", "day")) if got else "NONE-NONE-NONE"
    return {"file": fname, "pred": pred, "sec": round(time.time() - t0, 3)}


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

    lab = {}
    for fn in ("truth_v4.json", "truth_holdout.json"):
        p = os.path.join(OUT_DIR, fn)
        if os.path.exists(p):
            lab.update(json.load(open(p, encoding="utf-8"))["labels"])
    hold2 = set(json.load(open(os.path.join(OUT_DIR, "truth_holdout2.json"),
                               encoding="utf-8"))["labels"])
    files = [f for f in lab if f not in hold2 and
             os.path.exists(os.path.join(IMG_DIR, f))]
    print(f"라벨 있는 {len(files)}장 (홀드아웃2 제외) × 설정 3개", flush=True)

    from concurrent.futures import ProcessPoolExecutor
    out = {}
    for key in cfgs():
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            rows = list(ex.map(work, [(f, key) for f in files], chunksize=2))
        out[key] = {r["file"]: r for r in rows}
        print(f"  {key:16s} {time.time()-t0:.0f}초", flush=True)

    json.dump({k: {f: v["pred"] for f, v in d.items()} for k, d in out.items()},
              open(os.path.join(OUT_DIR, "75_precision.json"), "w",
                   encoding="utf-8"), ensure_ascii=False, indent=1)

    def score(key):
        d = out[key]
        ok = tot = miss = fp = 0
        per = {}
        for f in files:
            g, p = s3(lab[f]), s3(d[f]["pred"])
            hit = sum(g[i] == p[i] for i in range(3))
            per[f] = hit
            ok += hit
            tot += 3
            gn, pn = g == ("NONE",) * 3, p == ("NONE",) * 3
            miss += (pn and not gn)
            fp += (gn and not pn)
        sec = sum(d[f]["sec"] for f in files) / len(files)
        return {"score": ok / tot, "miss": miss, "fp": fp, "per": per, "sec": sec}

    S = {k: score(k) for k in out}
    base = "A_small_1024"
    DESC = {"A_small_1024": "det small @1024 (현재 운영)",
            "B1_medium_1024": "det medium @1024 (검출기만)",
            "B2_medium_1536": "det medium @1536 (검출기+해상도)"}

    L = ["# EDA 75 — 검출기 medium 의 정밀도 검증\n",
         "74단계에서 `B_det_medium` 이 어려운 실패 252장 중 69장을 회수했다. 그러나",
         "그 252장 중 라벨이 있는 것이 **4장뿐**이라 회수한 값이 맞는지 확인하지 못했다.",
         "검출기를 키우면 바코드·영양성분 같은 숫자도 더 잡히므로 회수가 곧 개선은 아니다.",
         "",
         f"측정 집합: 라벨 있는 **{len(files)}장** (튜닝셋 + 1차 홀드아웃).",
         "홀드아웃2 198장은 최종 판정용으로 계속 동결했다. 여기서 보려는 것은",
         "절대 성능이 아니라 **기준선 대비 회귀 여부**다.",
         "",
         "| 설정 | 필드 정확도 | 미검출 | 오탐 | 장당 |",
         "| --- | ---: | ---: | ---: | ---: |"]
    for k in out:
        s = S[k]
        L.append(f"| {DESC[k]} | **{s['score']*100:.1f}%** | {s['miss']} | {s['fp']} | "
                 f"{s['sec']:.2f}초 |")
    L.append("")

    L.append("## 짝지은 비교 (기준선 대비)\n")
    L.append("| 설정 | 차이 | 개선 | **악화** | McNemar p |")
    L.append("| --- | ---: | ---: | ---: | ---: |")
    for k in out:
        if k == base:
            continue
        up, dn, pv = mcnemar(S[base]["per"], S[k]["per"])
        d = (S[k]["score"] - S[base]["score"]) * 100
        L.append(f"| {DESC[k]} | {d:+.1f}%p | {up} | **{dn}** | {pv:.3f} |")
    L.append("")
    L.append("악화 장수가 핵심이다. 검출기를 키워 새로 읽은 것이 오답이면 여기 잡힌다.")
    L.append("")

    # 악화된 것 전부 나열 — 무엇을 망가뜨렸는지 직접 본다
    for k in out:
        if k == base:
            continue
        worse = [f for f in files if S[k]["per"][f] < S[base]["per"][f]]
        L.append(f"## {DESC[k]} — 악화된 {len(worse)}장\n")
        if worse:
            L.append("| 파일 | 정답 | 기준선 | 변경 후 |")
            L.append("| --- | --- | --- | --- |")
            for f in worse:
                L.append(f"| `{f}` | {lab[f]} | {out[base][f]['pred']} | "
                         f"{out[k][f]['pred']} |")
        else:
            L.append("없음.")
        L.append("")

    best = max((k for k in out if k != base), key=lambda k: S[k]["score"])
    L.append("## 판단\n")
    upb, dnb, pvb = mcnemar(S[base]["per"], S[best]["per"])
    db = (S[best]["score"] - S[base]["score"]) * 100
    if db > 0 and dnb <= upb * 0.4:
        L.append(f"**{DESC[best]} 를 홀드아웃2 에 적용해 볼 근거가 된다.** "
                 f"{db:+.1f}%p, 개선 {upb} / 악화 {dnb}.")
    elif db > 0:
        L.append(f"{DESC[best]} 가 {db:+.1f}%p 이지만 악화가 {dnb}장으로 적지 않다. "
                 "계단 구조(실패분에만 적용)로 붙이면 악화를 구조적으로 0 으로 만들 수 있다.")
    else:
        L.append(f"검출기를 키워도 이 집합에서는 {db:+.1f}%p 다. "
                 "회수는 늘지만 오답도 같이 늘어 상쇄된다.")
    L.append("")
    L.append("> ⚠️ 이 185장은 설계에 이미 쓰인 집합이라 절대 성능은 낙관적이다.")
    L.append("> 최종 판정은 동결한 홀드아웃2 로만 한다.")

    with open(os.path.join(OUT_DIR, "75_detector_precision.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L))
    print()
    print("\n".join(L[8:24]))
    print("\n→ eda/out/75_detector_precision.md")


if __name__ == "__main__":
    main()
