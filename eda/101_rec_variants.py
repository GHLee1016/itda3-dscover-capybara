"""
EDA 101단계 — 인식기(rec)를 바꾸면 상한이 오르는가

99·100단계에서 **검출기 쪽은 상한에 닿았다**. 해상도·전처리·det 크기를 아무리
조합해도 여덟 칸 오라클 상한이 94.9% 였고, 1600px 칸은 홀드아웃4 에서 0장이었다.
남은 오답 51장 중 49장이 "OCR 이 글자를 못 만드는" 경우이므로, 손대야 할 곳은
검출기가 아니라 **인식기**다.

번들에 넣을 수 있는 인식기 (미리 받아 weights/ 에 둔다. 자동 내려받기는 0점 사유)
    PP-OCRv6_rec_small     20MB   현재 사용 중
    PP-OCRv6_rec_medium    73MB   같은 세대의 큰 모델
    korean_PP-OCRv5_rec    13MB   한국어 전용
    en_PP-OCRv5_rec         8MB   영문 전용 (수입품 표기)

먼저 **상한이 오르는지**만 본다. 오르지 않으면 인식기 교체도 의미가 없고,
이 방향 전체를 접는다.

⚠️ 580장은 오염된 표본이다. 여기서는 상한과 후보를 보는 데만 쓴다.
   채택 판정은 새로 뽑은 표본에서 한다(홀드아웃4 를 더 쓰지 않는다).

    python eda/101_rec_variants.py --workers 4
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
RAPID_DIR = os.path.join(ROOT, "weights", "rapidocr")

DET_SMALL = "PP-OCRv6_det_small.onnx"
CLS = "ch_ppocr_mobile_v2.0_cls_mobile.onnx"

# 이름 -> (rec 파일, rec lang enum 이름)
RECS = {
    "rec_small (현재)": ("PP-OCRv6_rec_small.onnx", "CH"),
    "rec_medium":       ("PP-OCRv6_rec_medium.onnx", "CH"),
    "rec_korean":       ("korean_PP-OCRv5_rec_mobile.onnx", "KOREAN"),
    "rec_en":           ("en_PP-OCRv5_rec_mobile.onnx", "EN"),
}

_E = None


def build_one(rec_file, lang):
    """
    번들된 파일만 지정해 엔진을 만든다. 없으면 None (자동 내려받기 금지).

    combined.build_rapid_ocr 과 같이 **경로만** 넘긴다. model_type/ocr_version 은
    내려받을 모델을 고르는 용도라 이미 파일이 있으면 필요 없고,
    `Global.with_cls` 는 아예 없는 키다(처음에 넣었다가 전부 실패했다).
    """
    from rapidocr import RapidOCR
    det = os.path.join(RAPID_DIR, DET_SMALL)
    rec = os.path.join(RAPID_DIR, rec_file)
    cls = os.path.join(RAPID_DIR, CLS)
    if not (os.path.exists(det) and os.path.exists(rec)):
        return None
    p = {"Det.model_path": det, "Rec.model_path": rec}
    if os.path.exists(cls):
        p["Cls.model_path"] = cls
    try:
        from rapidocr.utils.typings import LangRec as LR
        p["Rec.lang_type"] = getattr(LR, lang)
    except Exception:
        pass
    try:
        return RapidOCR(params=p)
    except Exception as e:
        # 사전(dict) 이 안 맞을 수 있으니 lang 없이 한 번 더
        p.pop("Rec.lang_type", None)
        try:
            return RapidOCR(params=p)
        except Exception as e2:
            print(f"  엔진 생성 실패 {rec_file}: {type(e).__name__} {e} / "
                  f"{type(e2).__name__} {e2}", flush=True)
            return None


def setup():
    global _E
    if _E is None:
        import cv2
        import combined
        import team2_rules
        eng = {}
        for nm, (f, lang) in RECS.items():
            eng[nm] = build_one(f, lang)
        _E = (cv2, combined, team2_rules, eng)
    return _E


def work(f):
    cv2, combined, R, eng = setup()
    img = combined.imread_any(os.path.join(IMG_DIR, f))
    if img is None:
        return None
    small = combined.downscale_max(img, 1024)
    got, secs = {}, {}
    for nm, e in eng.items():
        if e is None:
            continue
        t0 = time.time()
        try:
            r = e(small)
            txt = list(r.txts) if getattr(r, "txts", None) else []
        except Exception:
            txt = []
        d = R.choose_expiration_date(txt) if txt else None
        if d is None and txt:
            p = R.extract_partial_date(txt)
            if p:
                d = "-".join(p[k] for k in ("year", "month", "day"))
        secs[nm] = round(time.time() - t0, 3)
        got[nm] = d
    return {"file": f, "got": got, "sec": secs}


def s3(v):
    if not v or v == "NONE":
        return ("NONE",) * 3
    q = str(v).split("-")
    return tuple(q) if len(q) == 3 else ("NONE",) * 3


def hits(g, p):
    return sum(g[i] == p[i] for i in range(3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    D = json.load(open(os.path.join(OUT, "88_all_pred.json"), encoding="utf-8"))
    lab, split = D["label"], D["split"]
    # 홀드아웃4 는 최종 판정 전용이므로 뺀다
    files = [f for f in sorted(lab)
             if split.get(f) != "holdout4"
             and os.path.exists(os.path.join(IMG_DIR, f))]
    print(f"{len(files)}장 × 인식기 {len(RECS)}종 (홀드아웃4 제외)", flush=True)

    t0 = time.time()
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        res = {r["file"]: r for r in ex.map(work, files, chunksize=2) if r}
    print(f"  {time.time()-t0:.0f}초", flush=True)

    json.dump({f: {"got": r["got"], "sec": r["sec"]} for f, r in res.items()},
              open(os.path.join(OUT, "101_recs.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    names = [nm for nm in RECS if any(nm in r["got"] for r in res.values())]
    n = len(res)
    tot = n * 3

    def acc(pick):
        ok = exact = 0
        for f, r in res.items():
            g, p = s3(lab[f]), s3(pick(f, r))
            ok += hits(g, p)
            exact += (g == p)
        return ok / tot * 100, exact / n * 100

    L = ["# EDA 101 — 인식기를 바꾸면 상한이 오르는가\n",
         "검출기 쪽은 상한에 닿았다(EDA 99·100). 남은 오답의 대부분이",
         "\"글자를 못 만드는\" 경우이므로 인식기를 본다.",
         "",
         f"- 라벨 {n}장 (홀드아웃4 는 최종 판정 전용이라 제외)",
         "- det 는 small@1024 로 고정하고 rec 만 바꿨다",
         "",
         "## 인식기별 단독 성능\n",
         "| 인식기 | 필드 정확도 | 완전일치 | 장당(병렬) |",
         "| --- | ---: | ---: | ---: |"]
    for nm in names:
        fa, ex = acc(lambda f, r, k=nm: r["got"].get(k))
        sec = sum(r["sec"].get(nm, 0) for r in res.values()) / n
        L.append(f"| {nm} | {fa:.1f}% | {ex:.1f}% | {sec:.2f}초 |")
    L.append("")

    base = names[0]

    def oracle(f, r):
        g = s3(lab[f])
        best, bh = None, -1
        for v in r["got"].values():
            h = hits(g, s3(v))
            if h > bh:
                best, bh = v, h
        return best
    ofa, oex = acc(oracle)
    bfa, bex = acc(lambda f, r: r["got"].get(base))

    L.append("## 상한\n")
    L.append("| | 필드 정확도 | 완전일치 |")
    L.append("| --- | ---: | ---: |")
    L.append(f"| 현재 인식기 단독 | {bfa:.1f}% | {bex:.1f}% |")
    L.append(f"| **네 인식기 오라클 상한** | **{ofa:.1f}%** | {oex:.1f}% |")
    L.append("")

    # 계단으로 붙였을 때 (실패분에만 다음 인식기)
    def ladder(order):
        def pick(f, r):
            part = None
            for nm in order:
                v = r["got"].get(nm)
                if v and "NONE" not in v:
                    return v
                if part is None and v:
                    part = v
            return part
        return acc(pick)

    L.append("## 계단으로 붙였을 때 (실패분에만 다음 인식기)\n")
    L.append("| 구성 | 필드 정확도 | 완전일치 |")
    L.append("| --- | ---: | ---: |")
    combos = [
        ([base], "현재 하나만"),
        ([base, "rec_medium"], "+ rec_medium"),
        ([base, "rec_korean"], "+ rec_korean"),
        ([base, "rec_en"], "+ rec_en"),
        ([base, "rec_medium", "rec_korean"], "+ medium + korean"),
        ([base, "rec_medium", "rec_korean", "rec_en"], "+ 셋 다"),
    ]
    best_combo = None
    for order, nm in combos:
        order = [o for o in order if o in names]
        fa, ex = ladder(order)
        L.append(f"| {nm} | **{fa:.1f}%** | {ex:.1f}% |")
        if best_combo is None or fa > best_combo[1]:
            best_combo = (nm, fa, order)
    L.append("")

    gain = ofa - bfa
    L.append("## 판단\n")
    if gain < 0.5:
        L.append(f"**오라클 상한이 현재보다 {gain:+.1f}%p 뿐이다. 인식기를 바꿔도 얻을 것이 없다.**")
        L.append("검출기에 이어 인식기도 막혔다. 이 엔진 계열로는 여기가 한계다.")
    else:
        d = best_combo[1] - bfa
        L.append(f"오라클 상한이 {gain:+.1f}%p 남아 있고, 계단으로 붙이면")
        L.append(f"**{best_combo[0]}** 가 {d:+.1f}%p 를 가져온다.")
        L.append("")
        L.append("이 수치는 오염된 표본에서 나온 것이므로, 채택하려면 새 표본이 필요하다.")

    with open(os.path.join(OUT, "101_rec_variants.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print()
    print("\n".join(L[6:]))
    print("\n→ eda/out/101_rec_variants.md")


if __name__ == "__main__":
    main()
