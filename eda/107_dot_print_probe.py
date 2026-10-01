"""
EDA 107단계 — 도트 인쇄를 영역 한정 + 색 분리로 읽을 수 있는가

제안된 아이디어
    "날짜가 있는 위치를 먼저 특정하고, 그 안에서 픽셀 색을 비교해 숫자 픽셀만
     남기면 점선(도트 매트릭스)도 배경과 분리해 읽을 수 있지 않을까"

기존 시도와 다른 점이 있다. 91단계의 전처리 열한 가지는 **이미지 전체**에
걸었다. 전체에 걸면 다른 텍스트가 망가지므로 세게 밀어붙일 수 없었다.
영역을 날짜 근처로 좁히면 공격적인 이진화·팽창을 써도 잃을 것이 없다.

확인 순서
  ① YOLO 가 미검출 이미지에서 날짜 영역을 잡는가   ← 이게 안 되면 아이디어가 성립하지 않는다
  ② 크롭 안에서 색 분리 + 도트 연결이 글자를 살리는가
  ③ 살아난 글자를 인식기가 읽는가

대상은 홀드아웃4+5 에서 **완전 미검출인 12장**이다.

    python eda/107_dot_print_probe.py
"""
import os
import sys
import json
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

IMG_DIR = os.path.join(ROOT, "images", "상품사진입니다")
OUT = os.path.join(ROOT, "eda", "out")
DUMP = os.path.join(ROOT, "eda", "out", "107_crops")
RAPID_DIR = os.path.join(ROOT, "weights", "rapidocr")
ONNX = os.path.join(ROOT, "weights", "yolo", "date_v1.onnx")


def variants(cv2, np, crop):
    """크롭 한 장에서 만들어 볼 것들. 전부 BGR 로 돌려준다."""
    out = [("원본", crop)]
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    def bgr(x):
        return cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)

    # 1) Otsu 이진화 — 배경/글자 두 덩어리로 가른다
    _, ot = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    out.append(("Otsu", bgr(ot)))

    # 2) Otsu + 도트 잇기. 점이 끊겨 있으니 닫기로 메운다
    for k in (2, 3):
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        out.append((f"Otsu+닫기{k}", bgr(cv2.morphologyEx(ot, cv2.MORPH_CLOSE, ker))))

    # 3) 크롭을 키운 뒤 잇기 — 도트가 작아 커널이 안 먹을 수 있다
    for scale in (2, 3):
        big = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        _, bo = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (scale + 1, scale + 1))
        out.append((f"{scale}배+닫기", bgr(cv2.morphologyEx(bo, cv2.MORPH_CLOSE, ker))))

    # 4) 색으로 가르기 — 밝기만이 아니라 RGB 거리로 두 무리를 나눈다.
    #    금속 반사처럼 밝기가 고르지 않은 배경을 겨냥한 것이다.
    try:
        Z = crop.reshape(-1, 3).astype(np.float32)
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, lb, cen = cv2.kmeans(Z, 2, None, crit, 3, cv2.KMEANS_PP_CENTERS)
        dark = int(np.argmin(cen.sum(axis=1)))          # 어두운 쪽이 잉크
        mask = (lb.flatten() == dark).astype(np.uint8).reshape(crop.shape[:2]) * 255
        mask = 255 - mask                                # 글자가 검게 오도록 반전
        out.append(("색분리", bgr(mask)))
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        out.append(("색분리+닫기", bgr(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker))))
    except Exception:
        pass

    return out


def main():
    import cv2
    import numpy as np
    import combined
    import team2_rules as R
    import pipeline as P

    D = json.load(open(os.path.join(OUT, "106_final.json"), encoding="utf-8"))
    lab, pred = D["label"], D["pred"]
    miss = [f for f in sorted(lab)
            if pred.get(f) == "NONE-NONE-NONE" and lab[f] != "NONE-NONE-NONE"]
    print(f"완전 미검출 {len(miss)}장을 대상으로 한다\n")

    if not os.path.exists(ONNX):
        print("YOLO 가중치가 없다:", ONNX)
        return
    det = P.OnnxDateDetector(ONNX, threads=1)
    rapid, _ = combined.build_rapid_ocr(RAPID_DIR)
    med, _ = combined.build_rapid_retry(RAPID_DIR)
    os.makedirs(DUMP, exist_ok=True)

    n_box = n_read = 0
    L = ["# EDA 107 — 도트 인쇄를 영역 한정 + 색 분리로 읽을 수 있는가\n",
         "91단계의 전처리는 **이미지 전체**에 걸어서 세게 밀어붙일 수 없었다.",
         "날짜 영역만 잘라내면 공격적인 이진화·팽창을 써도 잃을 것이 없다.",
         "",
         f"- 대상: 홀드아웃4+5 에서 완전 미검출인 {len(miss)}장",
         "",
         "## 이미지별\n"]

    for f in miss:
        img = combined.imread_any(os.path.join(IMG_DIR, f))
        if img is None:
            continue
        h, w = img.shape[:2]
        side = max(h, w)
        canvas = np.zeros((side, side, 3), np.uint8)
        canvas[:h, :w] = img
        boxes = det.detect(canvas, conf=0.25)

        L.append(f"### `{f}` — 정답 {lab[f]}\n")
        if not boxes:
            L.append("- **YOLO 박스 없음** — 영역을 못 잡았다")
            L.append("")
            print(f"  {f:14s} 박스 없음")
            continue
        n_box += 1

        hits = []
        for bi, (x1, y1, x2, y2) in enumerate(boxes[:3]):
            pad = 0.12
            bw, bh = x2 - x1, y2 - y1
            X1 = max(0, int(x1 - bw * pad)); Y1 = max(0, int(y1 - bh * pad))
            X2 = min(w, int(x2 + bw * pad)); Y2 = min(h, int(y2 + bh * pad))
            if X2 - X1 < 12 or Y2 - Y1 < 8:
                continue
            crop = img[Y1:Y2, X1:X2]
            cv2.imencode(".png", crop)[1].tofile(
                os.path.join(DUMP, f"{os.path.splitext(f)[0]}_b{bi}.png"))

            for name, arr in variants(cv2, np, crop):
                for eng_nm, eng in (("small", rapid), ("medium", med)):
                    if eng is None:
                        continue
                    # ⚠️ 크롭에 엔진을 통째로 넣으면 **검출기가 다시 돈다**.
                    #    이미 영역을 아는데 det 를 또 돌릴 이유가 없고, 작은
                    #    크롭에서 det 가 실패하면 rec 까지 가지도 못한다.
                    #    use_det=False 로 인식기만 부른다.
                    for mode, kw in (("rec만", {"use_det": False, "use_cls": False,
                                                "use_rec": True}),
                                     ("det+rec", {})):
                        try:
                            r = eng(arr, **kw)
                            txt = list(r.txts) if getattr(r, "txts", None) else []
                        except Exception:
                            txt = []
                        if not txt:
                            continue
                        got = R.choose_expiration_date(txt)
                        joined = " | ".join(txt)[:70]
                        if got:
                            hits.append((f"{name}·{mode}", eng_nm, got, joined))
                        elif mode == "rec만" and name in ("원본", "2배+닫기"):
                            # 날짜가 안 나와도 무엇을 읽었는지는 남긴다
                            hits.append((f"{name}·{mode}", eng_nm, "—", joined))

        L.append(f"- YOLO 박스 {len(boxes)}개")
        hits = [h for h in hits if h[2] != "—"] or hits
        if hits:
            ok = [h for h in hits if h[2] == lab[f]]
            n_read += bool(ok)
            mark = "**정답 나옴**" if ok else "날짜는 나왔으나 틀림"
            L.append(f"- {mark}")
            L.append("")
            L.append("| 전처리 | 엔진 | 규칙 결과 | 읽은 텍스트 |")
            L.append("| --- | --- | --- | --- |")
            seen = set()
            for name, eng_nm, got, joined in hits:
                if (name, got) in seen:
                    continue
                seen.add((name, got))
                star = " ✅" if got == lab[f] else ""
                L.append(f"| {name} | {eng_nm} | `{got}`{star} | `{joined}` |")
            print(f"  {f:14s} 박스 {len(boxes)}  "
                  f"{'정답 O' if ok else '날짜는 나옴'}")
        else:
            L.append("- 어떤 전처리로도 날짜가 나오지 않음")
            print(f"  {f:14s} 박스 {len(boxes)}  실패")
        L.append("")

    L.append("## 정리\n")
    L.append(f"- YOLO 가 영역을 잡은 것: **{n_box}/{len(miss)}장**")
    L.append(f"- 그중 정답까지 읽힌 것: **{n_read}장**")
    L.append("")
    L.append(f"크롭 이미지는 `eda/out/107_crops/` 에 남겼다. 눈으로 확인할 것.")

    with open(os.path.join(OUT, "107_dot_print_probe.md"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print(f"\nYOLO 박스 {n_box}/{len(miss)} · 정답까지 읽힌 것 {n_read}")
    print("→ eda/out/107_dot_print_probe.md")


if __name__ == "__main__":
    main()
