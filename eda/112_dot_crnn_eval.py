"""
EDA 110단계 — 도트 판독기를 실제 미검출 이미지에 걸어 본다

109단계의 96% 는 **합성 데이터에서의** 정확도다. 실제 사진에서도 그만큼
나오는지는 별개다. 여기서 확인한다.

흐름
    이미지 → YOLO 로 날짜 영역 검출 → 크롭
          → dotseg 로 글자 분할 → 분류기로 숫자 판독
          → 문자열 조립 → 기존 날짜 규칙(team2_rules)

대상은 홀드아웃4+5 에서 **완전 미검출인 12장**이다. 여기서 몇 장을 살리는지
보고, 살린다면 파이프라인에 붙일지 판단한다.

    python eda/110_dot_eval.py
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
WEIGHTS = os.path.join(ROOT, "weights")
ONNX_YOLO = os.path.join(WEIGHTS, "yolo", "date_v1.onnx")


def main():
    import cv2
    import numpy as np
    import combined
    import dotcrnn
    import team2_rules as R
    import pipeline as P

    reader, src = dotcrnn.build_crnn(WEIGHTS)
    if reader is None:
        print("판독기를 만들지 못했다:", src)
        return
    if not os.path.exists(ONNX_YOLO):
        print("YOLO 가중치가 없다")
        return
    det = P.OnnxDateDetector(ONNX_YOLO, threads=1)
    det2 = None   # v2 는 기각했다 (EDA 115)

    D = json.load(open(os.path.join(OUT, "106_final.json"), encoding="utf-8"))
    lab, pred = D["label"], D["pred"]
    miss = [f for f in sorted(lab)
            if pred.get(f) == "NONE-NONE-NONE" and lab[f] != "NONE-NONE-NONE"]
    print(f"완전 미검출 {len(miss)}장에 도트 판독기를 건다\n")

    L = ["# EDA 112 — CTC 판독기를 실제 미검출 이미지에 걸어 본다\n",
         "109단계의 96% 는 합성 데이터에서의 정확도다. 실제 사진은 별개다.",
         "",
         "흐름: YOLO 영역 검출 → 크롭 → 글자 분할 → 숫자 분류 → 기존 날짜 규칙",
         "",
         "| 파일 | 정답 | 판독기가 읽은 줄 | 규칙 결과 | |",
         "| --- | --- | --- | --- | --- |"]

    n_ok = n_any = 0
    for f in miss:
        img = combined.imread_any(os.path.join(IMG_DIR, f))
        if img is None:
            continue
        h, w = img.shape[:2]
        side = max(h, w)
        canvas = np.zeros((side, side, 3), np.uint8)
        canvas[:h, :w] = img
        boxes = list(det.detect(canvas, conf=0.25))
        if det2 is not None:
            boxes += list(det2.detect(canvas, conf=0.25))
        if not boxes:
            L.append(f"| `{f}` | {lab[f]} | (YOLO 박스 없음) | — | |")
            print(f"  {f:14s} 박스 없음")
            continue

        texts, got = [], None
        for x1, y1, x2, y2 in boxes[:5]:
            bw_, bh_ = x2 - x1, y2 - y1
            pad = 0.12
            X1 = max(0, int(x1 - bw_ * pad)); Y1 = max(0, int(y1 - bh_ * pad))
            X2 = min(w, int(x2 + bw_ * pad)); Y2 = min(h, int(y2 + bh_ * pad))
            if X2 - X1 < 12 or Y2 - Y1 < 8:
                continue
            g = cv2.cvtColor(img[Y1:Y2, X1:X2], cv2.COLOR_BGR2GRAY)
            texts += reader.read(g)
        if texts:
            got = R.choose_expiration_date(texts)
            if got is None:
                # `2022.07` 처럼 월·연만 있는 경우. 연·월·일이 독립 채점이라
                # 확인된 필드만이라도 내는 편이 낫다.
                pp = R.extract_partial_date(texts)
                if pp:
                    got = "-".join(pp[k] for k in ("year", "month", "day"))
        shown = " / ".join(texts)[:46] or "(글자 없음)"
        mark = ""
        if got:
            n_any += 1
            if got == lab[f]:
                n_ok += 1
                mark = "**정답**"
            else:
                mark = "틀림"
        L.append(f"| `{f}` | {lab[f]} | `{shown}` | `{got}` | {mark} |")
        print(f"  {f:14s} {shown[:34]:36s} → {got}")

    L.append("")
    L.append(f"**{len(miss)}장 중 날짜를 낸 것 {n_any}장, 정답 {n_ok}장**")
    L.append("")
    if n_ok == 0:
        L.append("실제 사진에서는 통하지 않는다. 합성과 실물의 간극이 크다.")
    else:
        L.append(f"미검출 {len(miss)}장 중 {n_ok}장을 살린다. 파이프라인에 붙일지는")
        L.append("오탐(맞히던 것을 망가뜨리는지)을 확인한 뒤 정한다.")

    with open(os.path.join(OUT, "112_dot_crnn_eval.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print(f"\n날짜를 낸 것 {n_any}장 · 정답 {n_ok}장")
    print("→ eda/out/112_dot_crnn_eval.md")


if __name__ == "__main__":
    main()
