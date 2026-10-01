"""
EDA 74단계 — 다른 OCR 엔진 조합이 어려운 실패를 읽어내는가

73단계까지의 결론: 실패 271장 중 해상도로 회수되는 것은 19장(7.0%)뿐이다.
나머지 252장은 캔 각인·점자 인쇄·저대비 잉크젯을 **엔진이 근본적으로 못 읽는** 경우다.

라벨링(사람 3~4시간)으로 가기 전에, 라벨이 필요 없는 엔진 교체를 먼저 본다.
RapidOCR 은 같은 인터페이스로 여러 모델 조합을 제공한다.

  A_baseline      현재 구성. multi PP-OCRv6 det/rec small
  B_det_medium    검출기를 medium 으로 (검출 실패 39장이 표적)
  C_rec_medium    인식기를 medium 으로
  D_rec_korean    한국어 전용 rec (PP-OCRv5). 현재는 다국어 모델이다
  E_rec_en        영문 전용 rec (PP-OCRv5). 날짜는 숫자열이라 한글 어휘가 오히려 방해일 수 있다
  F_det_server    ch PP-OCRv5 det server. 가장 큰 검출기
  G_det_bigside   현재 검출기에 limit_side_len 736 → 1280

⚠️ RapidOCR 의 Global.max_side_len 이 2000 이다. 73단계에서 3072px 가 2048px 대비
   아무 이득이 없었던 이유가 이것이다. 입력을 더 키워도 내부에서 잘린다.

⚠️ 여기서 회수되는 장수는 **실패 집합 안에서의 비율**이다. 전체 성능 개선폭은
   동결한 홀드아웃2 에 적용해 따로 재야 한다.

    python eda/74_engine_variants.py --workers 4
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

IMG_DIR = os.path.join(ROOT, "images", "상품사진입니다")
OUT_DIR = os.path.join(ROOT, "eda", "out")

SIDE = 1536          # 73단계에서 회수가 거의 다 일어난 지점


def configs():
    from rapidocr.utils.typings import (ModelType as MT, LangRec as LR,
                                        LangDet as LD, OCRVersion as OV)
    return {
        "A_baseline":    {},
        "B_det_medium":  {"Det.lang_type": LD.CH, "Det.model_type": MT.MEDIUM,
                          "Det.ocr_version": OV.PPOCRV6},
        "C_rec_medium":  {"Rec.lang_type": LR.CH, "Rec.model_type": MT.MEDIUM,
                          "Rec.ocr_version": OV.PPOCRV6},
        "D_rec_korean":  {"Rec.lang_type": LR.KOREAN, "Rec.model_type": MT.MOBILE,
                          "Rec.ocr_version": OV.PPOCRV5},
        "E_rec_en":      {"Rec.lang_type": LR.EN, "Rec.model_type": MT.MOBILE,
                          "Rec.ocr_version": OV.PPOCRV5},
        "F_det_server":  {"Det.lang_type": LD.CH, "Det.model_type": MT.SERVER,
                          "Det.ocr_version": OV.PPOCRV5},
        "G_det_bigside": {"Det.limit_side_len": 1280},
    }


_OCR = None
_KEY = None


def work(args):
    global _OCR, _KEY
    fname, key = args
    import cv2
    import combined
    import team2_rules as rules
    from rapidocr import RapidOCR

    if _OCR is None or _KEY != key:
        _OCR = RapidOCR(params=configs()[key])
        _KEY = key

    img = combined.imread_any(os.path.join(IMG_DIR, fname))
    if img is None:
        return {"file": fname, "ok": False, "pred": "NONE-NONE-NONE", "n": 0, "sec": 0}
    h, w = img.shape[:2]
    if max(h, w) > SIDE:
        s = SIDE / max(h, w)
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
    ok = got is not None and any(v != "NONE" for v in got.values())
    return {"file": fname, "ok": bool(ok), "n": len(texts),
            "sec": round(time.time() - t0, 3),
            "pred": "-".join(str(got.get(k, "NONE") or "NONE")
                             for k in ("year", "month", "day")) if got
                    else "NONE-NONE-NONE"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    # 73단계에서 해상도로도 회수되지 않은 '어려운' 집합
    res73 = json.load(open(os.path.join(OUT_DIR, "73_resolution.json"),
                           encoding="utf-8"))
    hard = [r["file"] for r in res73 if not r["first_ok"]]
    inv = {r["file"]: r for r in
           json.load(open(os.path.join(OUT_DIR, "69_failures.json"), encoding="utf-8"))}
    if a.limit:
        hard = hard[:a.limit]
    print(f"어려운 실패 {len(hard)}장 × 설정 {len(configs())}개", flush=True)

    lab = {}
    for fn in ("truth_v4.json", "truth_holdout.json"):
        p = os.path.join(OUT_DIR, fn)
        if os.path.exists(p):
            lab.update(json.load(open(p, encoding="utf-8"))["labels"])

    from concurrent.futures import ProcessPoolExecutor
    out = {}
    for key in configs():
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            rows = list(ex.map(work, [(f, key) for f in hard], chunksize=2))
        out[key] = rows
        nok = sum(r["ok"] for r in rows)
        avg = sum(r["sec"] for r in rows) / max(len(rows), 1)
        print(f"  {key:14s} 회수 {nok:3d}/{len(rows)}  ({nok/len(rows)*100:4.1f}%)  "
              f"{avg:.2f}초/장  총 {time.time()-t0:.0f}초", flush=True)

    json.dump(out, open(os.path.join(OUT_DIR, "74_engine_variants.json"), "w",
                        encoding="utf-8"), ensure_ascii=False)

    def s3(v):
        if not v or v == "NONE":
            return ("NONE",) * 3
        q = str(v).split("-")
        return tuple(q) if len(q) == 3 else ("NONE",) * 3

    DESC = {
        "A_baseline": "현재 구성 (multi v6 det/rec small)",
        "B_det_medium": "검출기 medium (ch v6)",
        "C_rec_medium": "인식기 medium (ch v6)",
        "D_rec_korean": "인식기 한국어 전용 (v5)",
        "E_rec_en": "인식기 영문 전용 (v5)",
        "F_det_server": "검출기 server (ch v5)",
        "G_det_bigside": "검출 입력 736 → 1280",
    }

    n = len(hard)
    base_ok = {r["file"] for r in out["A_baseline"] if r["ok"]}

    L = ["# EDA 74 — 엔진 조합을 바꾸면 어려운 실패를 읽는가\n",
         "73단계까지의 결론은 '실패 271장 중 해상도로 회수되는 것은 19장뿐' 이었다.",
         "남은 252장은 엔진이 근본적으로 못 읽는 경우다. 라벨링으로 가기 전에",
         "라벨이 필요 없는 엔진 교체를 먼저 확인한다.",
         "",
         f"- 대상 **{n}장** (73단계에서 해상도로도 회수 안 된 것)",
         f"- 입력 긴 변 {SIDE}px 고정. RapidOCR 내부 상한이 2000px 라 더 키워도 소용없다",
         "",
         "## 설정별 회수\n",
         "| 설정 | 내용 | 회수 | 회수율 | 기준선 대비 신규 | 장당 |",
         "| --- | --- | ---: | ---: | ---: | ---: |"]
    for key, rows in out.items():
        okf = {r["file"] for r in rows if r["ok"]}
        new = okf - base_ok
        avg = sum(r["sec"] for r in rows) / max(len(rows), 1)
        mark = "**" if key != "A_baseline" and len(new) >= 10 else ""
        L.append(f"| `{key}` | {DESC[key]} | {mark}{len(okf)}{mark} | "
                 f"{len(okf)/n*100:.1f}% | "
                 f"{'—' if key=='A_baseline' else f'+{len(new)}'} | {avg:.2f}초 |")
    L.append("")

    # 합집합 상한
    union = set()
    for key, rows in out.items():
        union |= {r["file"] for r in rows if r["ok"]}
    L.append(f"> 모든 설정의 **합집합은 {len(union)}장 ({len(union)/n*100:.1f}%)**. "
             "여러 설정을 이어 붙였을 때의 상한이다.")
    L.append("")

    # 라벨 있는 것에 한해 정답률
    labeled = [f for f in hard if f in lab]
    if labeled:
        L.append(f"## 회수한 값이 맞는가 (라벨 있는 {len(labeled)}장)\n")
        L.append("| 설정 | 회수 | 그중 완전일치 | 정밀도 |")
        L.append("| --- | ---: | ---: | ---: |")
        for key, rows in out.items():
            m = {r["file"]: r for r in rows}
            rec = [f for f in labeled if m[f]["ok"]]
            exact = sum(1 for f in rec if s3(lab[f]) == s3(m[f]["pred"]))
            L.append(f"| `{key}` | {len(rec)} | {exact} | "
                     f"{exact/len(rec)*100:.0f}% |" if rec else
                     f"| `{key}` | 0 | 0 | — |")
        L.append("")
        L.append("회수가 곧 정답은 아니다. 정밀도가 낮으면 오답을 늘리는 변경이다.")
        L.append("")

    # 원인별
    L.append("## 69단계 원인별 (기준선 대비 개선이 큰 설정)\n")
    KO = {"no_text": "글자 못 찾음", "text_no_digits": "숫자 줄 없음",
          "digits_no_date": "숫자는 있으나 날짜 아님"}
    best = max((k for k in out if k != "A_baseline"),
               key=lambda k: len({r["file"] for r in out[k] if r["ok"]} - base_ok))
    m = {r["file"]: r for r in out[best]}
    L.append(f"가장 많이 건진 설정: `{best}` ({DESC[best]})\n")
    L.append("| 원인 | 대상 | 기준선 회수 | " + best + " 회수 |")
    L.append("| --- | ---: | ---: | ---: |")
    mb = {r["file"]: r for r in out["A_baseline"]}
    for c in ("digits_no_date", "text_no_digits", "no_text"):
        sub = [f for f in hard if inv.get(f, {}).get("cause") == c]
        if not sub:
            continue
        L.append(f"| {KO[c]} | {len(sub)} | {sum(1 for f in sub if mb[f]['ok'])} | "
                 f"{sum(1 for f in sub if m[f]['ok'])} |")
    L.append("")
    L.append("## 판단\n")
    gain = len({r["file"] for r in out[best] if r["ok"]} - base_ok)
    if gain >= 25:
        L.append(f"**엔진 교체에 값이 있다.** `{best}` 가 기준선 대비 {gain}장을 더 읽는다.")
        L.append("다음은 이 설정을 홀드아웃2 에 적용해 전체 성능 변화를 재는 것이다.")
    elif gain >= 10:
        L.append(f"`{best}` 가 {gain}장을 더 읽는다. 크지 않지만 실패분에만 걸면 잃을 것이 없다.")
        L.append("보조 단계로 붙여 홀드아웃2 에서 확인할 가치가 있다.")
    else:
        L.append(f"가장 나은 설정도 {gain}장에 그친다. **엔진 교체로는 풀리지 않는다.**")
        L.append("라벨링(크롭 + 인식기 파인튜닝) 쪽으로 방향을 돌리는 것이 맞다.")

    with open(os.path.join(OUT_DIR, "74_engine_variants.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L))
    print()
    print("\n".join(L[8:22]))
    print("\n→ eda/out/74_engine_variants.md")


if __name__ == "__main__":
    main()
