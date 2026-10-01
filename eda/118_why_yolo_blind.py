"""
EDA 118단계 — YOLO 는 왜 날짜 영역을 못 잡는가

117단계에서 병목이 드러났다.
    YOLO 가 잡은 635장   96.1% · 미검출률 1.1%
    YOLO 가 못 잡은 193장 85.8% · 미검출률 7.8%   ← -10.2%p

왜 못 잡는지 알아야 고칠 수 있다. 그런데 라벨에는 날짜 **값**만 있고
위치가 없다. 그래서 OCR 이 읽은 박스에서 역산한다.
    RapidOCR 로 텍스트와 박스를 얻고 → 날짜꼴 문자열이 든 조각을 찾고
    → 그 박스의 크기·위치·대비를 잰다

두 무리(잡은 것 / 못 잡은 것)에서 같은 값을 재서 **무엇이 다른지** 본다.
한쪽만 보면 원인을 짚을 수 없다. 흐릿함을 원인으로 잘못 지목했던 것도
오답만 보고 정답 쪽을 재지 않았기 때문이었다(EDA 89).

    python eda/118_why_yolo_blind.py --workers 4
"""
import os
import re
import csv
import sys
import json
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

# 날짜꼴 — 느슨하게 잡는다. 위치만 알면 되므로 정확한 파싱은 필요 없다.
DATEISH = re.compile(
    r"(?<!\d)(?:20)?\d{2}\s*[.\-/]\s*\d{1,2}\s*[.\-/]\s*\d{1,2}"
    r"|(?<!\d)\d{1,2}\s*(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)",
    re.I)

_S = None


def setup():
    global _S
    if _S is None:
        import cv2
        import numpy as np
        import combined
        rapid, _ = combined.build_rapid_ocr(RAPID_DIR)
        _S = (cv2, np, combined, rapid)
    return _S


def work(f):
    cv2, np, combined, rapid = setup()
    img = combined.imread_any(os.path.join(IMG_DIR, f))
    if img is None or rapid is None:
        return None
    H, W = img.shape[:2]
    small = combined.downscale_max(img, 1024)
    sh, sw = small.shape[:2]
    try:
        r = rapid(small)
        txts = list(r.txts) if getattr(r, "txts", None) else []
        boxes = np.array(r.boxes) if getattr(r, "boxes", None) is not None else None
    except Exception:
        return None
    if not txts or boxes is None or len(boxes) != len(txts):
        return {"file": f, "found": False}

    # 날짜꼴이 든 조각의 박스를 고른다
    hit = None
    for t, b in zip(txts, boxes):
        if DATEISH.search(str(t).upper()):
            xs, ys = b[:, 0], b[:, 1]
            hit = (float(xs.min()), float(ys.min()),
                   float(xs.max()), float(ys.max()))
            break
    if hit is None:
        return {"file": f, "found": False}

    x1, y1, x2, y2 = hit
    bw, bh = x2 - x1, y2 - y1
    # YOLO 는 긴 변을 640 정사각에 맞춘다. 그 기준에서 날짜가 몇 픽셀인지가 관건.
    yolo_scale = 640.0 / max(sh, sw)
    h640 = bh * yolo_scale
    w640 = bw * yolo_scale

    # 대비 — 글자와 주변 배경의 밝기 차
    g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    X1, Y1 = max(0, int(x1)), max(0, int(y1))
    X2, Y2 = min(sw, int(x2)), min(sh, int(y2))
    inner = g[Y1:Y2, X1:X2]
    pad = int(max(bh, 4))
    oy1, oy2 = max(0, Y1 - pad), min(sh, Y2 + pad)
    ox1, ox2 = max(0, X1 - pad), min(sw, X2 + pad)
    outer = g[oy1:oy2, ox1:ox2]
    if inner.size < 9 or outer.size < 9:
        return {"file": f, "found": False}
    contrast = float(abs(float(inner.mean()) - float(outer.mean())))
    inner_std = float(inner.std())

    # 가장자리에 붙어 있는가 (0 = 한가운데, 1 = 테두리)
    cx, cy = (x1 + x2) / 2 / sw, (y1 + y2) / 2 / sh
    edge = float(max(abs(cx - 0.5), abs(cy - 0.5)) * 2)

    return {"file": f, "found": True,
            "h640": round(h640, 1), "w640": round(w640, 1),
            "contrast": round(contrast, 1), "std": round(inner_std, 1),
            "edge": round(edge, 2), "srcH": H, "srcW": W,
            "area640": round(h640 * w640, 0)}


def q(vals, p):
    v = sorted(x for x in vals if x is not None)
    if not v:
        return 0.0
    return v[min(int(len(v) * p), len(v) - 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    lab = {}
    with open(os.path.join(ROOT, "custom_data", "labels.csv"),
              encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            lab[r["source_file"]] = r["final_date"]
    dot = json.load(open(os.path.join(OUT, "116_dotness.json"), encoding="utf-8"))
    seen = set(dot)

    files = [f for f in sorted(lab) if os.path.exists(os.path.join(IMG_DIR, f))]
    print(f"{len(files)}장에서 날짜 위치를 역산한다", flush=True)

    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        res = {r["file"]: r for r in ex.map(work, files, chunksize=4) if r}

    got = [r for f, r in res.items() if f in seen and r.get("found")]
    blind = [r for f, r in res.items() if f not in seen and r.get("found")]
    nf_got = sum(1 for f, r in res.items() if f in seen and not r.get("found"))
    nf_bl = sum(1 for f, r in res.items() if f not in seen and not r.get("found"))
    print(f"  YOLO 잡음 {len(got)}장 / 못 잡음 {len(blind)}장 "
          f"(날짜 위치를 못 찾은 것 {nf_got}+{nf_bl})", flush=True)

    KEYS = [("h640", "날짜 높이 (640 기준, px)"),
            ("area640", "날짜 넓이 (640 기준, px²)"),
            ("contrast", "글자-배경 밝기 차"),
            ("std", "글자 영역 표준편차"),
            ("edge", "가장자리 정도 (0=중앙)")]

    L = ["# EDA 118 — YOLO 는 왜 날짜 영역을 못 잡는가\n",
         "117단계에서 YOLO 가 못 잡는 193장이 -10.2%p 를 만든다는 것이 드러났다.",
         "라벨에는 날짜 값만 있고 위치가 없어, **OCR 이 읽은 박스에서 역산**했다.",
         "",
         "두 무리에서 같은 값을 재서 무엇이 다른지 본다. 한쪽만 보면 원인을",
         "짚을 수 없다(오답만 보고 흐릿함을 원인으로 잘못 지목한 적이 있다).",
         "",
         f"- YOLO 가 잡은 것 {len(got)}장 · 못 잡은 것 {len(blind)}장",
         "",
         "| 지표 | 잡은 것 (중앙값) | 못 잡은 것 (중앙값) | 차이 |",
         "| --- | ---: | ---: | ---: |"]
    for k, name in KEYS:
        A = q([r[k] for r in got], 0.5)
        B = q([r[k] for r in blind], 0.5)
        rel = (B / A) if A else 0
        L.append(f"| {name} | {A:.1f} | {B:.1f} | **{rel:.2f}배** |")
    L.append("")

    # 높이 구간별 검출률 — 가장 직접적인 증거
    L.append("## 날짜 높이별 검출률\n")
    L.append("YOLO 입력은 640 정사각이다. 그 안에서 날짜가 몇 픽셀인지가 관건이다.\n")
    L.append("| 날짜 높이 (640 기준) | 전체 | YOLO 가 잡음 | 검출률 |")
    L.append("| --- | ---: | ---: | ---: |")
    bands = [(0, 8), (8, 12), (12, 16), (16, 24), (24, 999)]
    for b1, b2 in bands:
        g = sum(1 for r in got if b1 <= r["h640"] < b2)
        bl = sum(1 for r in blind if b1 <= r["h640"] < b2)
        tot = g + bl
        if tot == 0:
            continue
        nm = f"{b1}~{b2}px" if b2 < 999 else f"{b1}px 이상"
        L.append(f"| {nm} | {tot} | {g} | **{g/tot*100:.0f}%** |")
    L.append("")

    # 대비 구간별
    L.append("## 대비별 검출률\n")
    L.append("| 글자-배경 밝기 차 | 전체 | YOLO 가 잡음 | 검출률 |")
    L.append("| --- | ---: | ---: | ---: |")
    for b1, b2 in [(0, 10), (10, 20), (20, 35), (35, 999)]:
        g = sum(1 for r in got if b1 <= r["contrast"] < b2)
        bl = sum(1 for r in blind if b1 <= r["contrast"] < b2)
        tot = g + bl
        if tot == 0:
            continue
        nm = f"{b1}~{b2}" if b2 < 999 else f"{b1} 이상"
        L.append(f"| {nm} | {tot} | {g} | **{g/tot*100:.0f}%** |")
    L.append("")

    json.dump(res, open(os.path.join(OUT, "118_date_geom.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    with open(os.path.join(OUT, "118_why_yolo_blind.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L))
    print()
    print("\n".join(L[8:]))
    print("\n→ eda/out/118_why_yolo_blind.md")


if __name__ == "__main__":
    main()
