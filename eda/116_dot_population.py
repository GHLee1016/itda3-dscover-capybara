"""
EDA 116단계 — 데이터셋에서 도트 인쇄를 찾아 그 그룹의 성능을 잰다

지금까지는 **미검출 12장**만 보고 도트 판독기를 만들었다. 그런데 도트 인쇄가
전체에 얼마나 있고, 그 그룹에서 우리 성능이 어떤지는 모른다.
  · 도트가 드물다면 이 방향의 천장이 낮다는 뜻이고
  · 도트에서만 유독 나쁘다면 아직 남은 손잡이가 있다는 뜻이다

도트를 어떻게 가르나
  눈으로 828장을 다 볼 수는 없다. 도트는 획이 점으로 끊겨 있으므로
  **닫기 연산 전후로 연결요소 수가 크게 줄어든다.** 연속 획은 별로 안 변한다.
      지표 = (닫기 전 요소 수) / (닫기 후 요소 수)
  날짜 영역만 보려고 YOLO 크롭 안에서 잰다. 배경 글자가 섞이면 지표가 흐려진다.

⚠️ 이 지표는 추정이다. 상위 몇 장은 눈으로 확인해 맞는지 본다.

    python eda/116_dot_population.py --workers 4
"""
import os
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
DUMP = os.path.join(OUT, "116_dot_crops")
ONNX = os.path.join(ROOT, "weights", "yolo", "date_v1.onnx")

_S = None


def setup():
    global _S
    if _S is None:
        import cv2
        import numpy as np
        import combined
        import pipeline as P
        det = P.OnnxDateDetector(ONNX, threads=1) if os.path.exists(ONNX) else None
        _S = (cv2, np, combined, det)
    return _S


def dotness(cv2, np, gray):
    """
    도트다움. 닫기 전후 연결요소 수의 비.

    도트 인쇄는 점이 끊겨 있어 닫기로 이으면 요소가 확 준다.
    연속 획(일반 인쇄)은 애초에 이어져 있어 별로 안 변한다.
    """
    g = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if bw.mean() > 127:
        bw = 255 - bw
    area = int((bw > 0).sum())
    if area < 60:
        return None
    n0 = cv2.connectedComponentsWithStats(bw, 8)[0] - 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    n1 = cv2.connectedComponentsWithStats(
        cv2.morphologyEx(bw, cv2.MORPH_CLOSE, ker), 8)[0] - 1
    if n0 < 4 or n1 < 1:
        return None
    return n0 / max(n1, 1)


def work(f):
    cv2, np, combined, det = setup()
    if det is None:
        return None
    img = combined.imread_any(os.path.join(IMG_DIR, f))
    if img is None:
        return None
    h, w = img.shape[:2]
    side = max(h, w)
    canvas = np.zeros((side, side, 3), np.uint8)
    canvas[:h, :w] = img
    try:
        boxes = det.detect(canvas, conf=0.25)
    except Exception:
        return None
    best = None
    for x1, y1, x2, y2 in boxes[:3]:
        bw_, bh_ = x2 - x1, y2 - y1
        X1, Y1 = max(0, int(x1)), max(0, int(y1))
        X2, Y2 = min(w, int(x2)), min(h, int(y2))
        if X2 - X1 < 14 or Y2 - Y1 < 9:
            continue
        crop = cv2.cvtColor(img[Y1:Y2, X1:X2], cv2.COLOR_BGR2GRAY)
        d = dotness(cv2, np, crop)
        if d is not None and (best is None or d > best[0]):
            best = (d, (X1, Y1, X2, Y2))
    if best is None:
        return {"file": f, "dot": None}
    return {"file": f, "dot": round(best[0], 2), "box": best[1]}


def s3(v):
    if not v or v == "NONE":
        return ("NONE",) * 3
    q = str(v).split("-")
    return tuple(q) if len(q) == 3 else ("NONE",) * 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--thr", type=float, default=3.0, help="도트 판정 문턱")
    a = ap.parse_args()

    import cv2
    import numpy as np
    import combined

    # 라벨 828장 전체
    import csv
    lab = {}
    with open(os.path.join(ROOT, "custom_data", "labels.csv"),
              encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            lab[r["source_file"]] = r["final_date"]
    files = [f for f in sorted(lab) if os.path.exists(os.path.join(IMG_DIR, f))]
    print(f"라벨 {len(files)}장에서 도트 인쇄를 찾는다", flush=True)

    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        res = {r["file"]: r for r in ex.map(work, files, chunksize=4) if r}

    have = {f: r for f, r in res.items() if r.get("dot") is not None}
    print(f"  날짜 영역을 잡은 것 {len(have)}장", flush=True)

    dots = sorted(have.items(), key=lambda kv: -kv[1]["dot"])
    hi = [f for f, r in dots if r["dot"] >= a.thr]
    lo = [f for f, r in dots if r["dot"] < a.thr]
    print(f"  도트 판정(>={a.thr}) {len(hi)}장 / 일반 {len(lo)}장", flush=True)

    # 눈으로 확인할 크롭을 남긴다 (상위 12장 + 하위 6장)
    os.makedirs(DUMP, exist_ok=True)
    for tag, sel in (("hi", [f for f, _ in dots[:12]]),
                     ("lo", [f for f, _ in dots[-6:]])):
        for f in sel:
            r = have[f]
            img = combined.imread_any(os.path.join(IMG_DIR, f))
            if img is None:
                continue
            X1, Y1, X2, Y2 = r["box"]
            cv2.imencode(".png", img[Y1:Y2, X1:X2])[1].tofile(
                os.path.join(DUMP, f"{tag}_{r['dot']:.1f}_{f}.png"))

    # 성능 비교
    pred = json.load(open(os.path.join(OUT, "88_all_pred.json"),
                          encoding="utf-8"))["pred"]

    def score(group):
        ok = tot = miss = 0
        for f in group:
            if f not in pred:
                continue
            g, p = s3(lab[f]), s3(pred[f])
            ok += sum(g[i] == p[i] for i in range(3))
            tot += 3
            if p == ("NONE",) * 3 and g != ("NONE",) * 3:
                miss += 1
        return (ok / tot * 100 if tot else 0), tot // 3, miss

    fa_hi, n_hi, m_hi = score(hi)
    fa_lo, n_lo, m_lo = score(lo)

    L = ["# EDA 116 — 데이터셋의 도트 인쇄 분포와 그 그룹의 성능\n",
         "지금까지는 미검출 12장만 보고 도트 판독기를 만들었다. 전체에 도트가",
         "얼마나 있고 그 그룹에서 성능이 어떤지 본다.",
         "",
         "## 가르는 방법\n",
         "도트는 획이 점으로 끊겨 있어 **닫기 연산으로 이으면 연결요소가 확 준다.**",
         "연속 획은 별로 안 변한다. 그 비를 도트다움으로 쓴다.",
         "날짜 영역(YOLO 크롭) 안에서만 재서 배경 글자가 섞이지 않게 했다.",
         "",
         f"- 라벨 {len(files)}장 중 날짜 영역을 잡은 것 **{len(have)}장**",
         f"- 도트 판정(지표 ≥ {a.thr}) **{len(hi)}장** ({len(hi)/max(len(have),1)*100:.0f}%)",
         "",
         "## 두 그룹의 성능\n",
         "| 그룹 | 장수 | 필드 정확도 | 미검출 |",
         "| --- | ---: | ---: | ---: |",
         f"| **도트 인쇄** | {n_hi} | **{fa_hi:.1f}%** | {m_hi} |",
         f"| 일반 인쇄 | {n_lo} | **{fa_lo:.1f}%** | {m_lo} |",
         "",
         f"차이 **{fa_hi - fa_lo:+.1f}%p**",
         ""]
    if fa_hi < fa_lo - 3:
        L.append("도트 그룹이 뚜렷하게 나쁘다. 아직 손잡이가 남아 있다는 뜻이다.")
    elif fa_hi > fa_lo - 3:
        L.append("두 그룹 차이가 크지 않다. 도트가 유독 어려운 것은 아니다.")
    L.append("")
    L.append("## 지표 상위 (도트로 판정된 것)\n")
    L.append("| 파일 | 지표 | 정답 | 예측 |")
    L.append("| --- | ---: | --- | --- |")
    for f, r in dots[:15]:
        L.append(f"| `{f}` | {r['dot']:.1f} | {lab[f]} | {pred.get(f, '—')} |")
    L.append("")
    L.append("## 지표 하위 (일반 인쇄)\n")
    L.append("| 파일 | 지표 | 정답 | 예측 |")
    L.append("| --- | ---: | --- | --- |")
    for f, r in dots[-6:]:
        L.append(f"| `{f}` | {r['dot']:.1f} | {lab[f]} | {pred.get(f, '—')} |")
    L.append("")
    L.append(f"크롭은 `eda/out/116_dot_crops/` 에 남겼다. **지표가 맞는지 눈으로 볼 것.**")

    json.dump({f: r.get("dot") for f, r in have.items()},
              open(os.path.join(OUT, "116_dotness.json"), "w",
                   encoding="utf-8"), ensure_ascii=False, indent=1)
    with open(os.path.join(OUT, "116_dot_population.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L))
    print()
    print("\n".join(L[8:22]))
    print("\n→ eda/out/116_dot_population.md")


if __name__ == "__main__":
    main()
