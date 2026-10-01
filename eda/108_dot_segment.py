"""
EDA 108단계 — 도트 인쇄 크롭에서 글자가 하나씩 떨어지는가

107단계에서 확인한 것
  · YOLO 는 날짜 영역을 정확히 잡는다(미검출 12장 중 5장)
  · 그런데 인식기가 도트 글꼴을 못 읽는다. `26.08.13` 을 `2:88:13` 으로 읽는다.
    이진화·모폴로지·색분리·확대를 다 해도 같다. 분리의 문제가 아니라
    **인식기가 그 글꼴을 학습한 적이 없는** 것이다.

그래서 숫자 분류기를 직접 만들어 붙이려 한다. 그 전에 전제를 확인한다.
    **크롭에서 글자가 하나씩 떨어져야** 분류기에 넣을 수 있다.

도트 글꼴은 획이 점으로 끊겨 있어서, 닫기 연산으로 점을 이어 붙인 뒤
연결요소를 뽑아야 한다. 커널이 너무 작으면 한 글자가 여러 조각으로 남고,
너무 크면 옆 글자와 붙는다. 그 사이가 있는지 본다.

    python eda/108_dot_segment.py
"""
import os
import sys
import glob

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CROPS = os.path.join(ROOT, "eda", "out", "107_crops")
DUMP = os.path.join(ROOT, "eda", "out", "108_seg")


def segment(cv2, np, gray, k, scale=3):
    """확대 → 이진화 → 점 잇기 → 연결요소. (박스목록, 시각화이미지)"""
    big = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    # 도트는 배경보다 어둡다. 반전해 글자가 흰색이 되게 한다.
    _, bw = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, ker)

    n, lab, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
    H, W = closed.shape
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        # 너무 작은 얼룩과 너무 큰 덩어리(배경/테두리)는 뺀다
        if area < (H * W) * 0.0008:
            continue
        if h < H * 0.15 or h > H * 0.95:
            continue
        if w > W * 0.4:
            continue
        boxes.append((x, y, w, h))
    boxes.sort(key=lambda b: b[0])

    vis = cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)
    for x, y, w, h in boxes:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 1)
    return boxes, vis, closed


def main():
    import cv2
    import numpy as np

    os.makedirs(DUMP, exist_ok=True)
    files = sorted(glob.glob(os.path.join(CROPS, "*.png")))
    if not files:
        print("크롭이 없다. 107단계를 먼저 돌릴 것.")
        return

    L = ["# EDA 108 — 도트 크롭에서 글자가 하나씩 떨어지는가\n",
         "107단계에서 인식기가 도트 글꼴을 못 읽는 것이 드러났다(`26.08.13` → `2:88:13`).",
         "숫자 분류기를 직접 붙이려면 **글자가 하나씩 떨어져야** 한다. 그 전제를 본다.",
         "",
         "닫기 커널 크기를 바꿔 가며 연결요소를 뽑았다. 커널이 작으면 한 글자가",
         "여러 조각으로 남고, 크면 옆 글자와 붙는다. 그 사이가 있는지가 관건이다.",
         "",
         "| 크롭 | 커널3 | 커널5 | 커널7 | 커널9 |",
         "| --- | ---: | ---: | ---: | ---: |"]

    best = {}
    for p in files:
        nm = os.path.basename(p)
        g = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        row = [nm]
        for k in (3, 5, 7, 9):
            boxes, vis, _ = segment(cv2, np, g, k)
            row.append(str(len(boxes)))
            if k == 5:
                cv2.imencode(".png", vis)[1].tofile(
                    os.path.join(DUMP, nm.replace(".png", f"_k{k}.png")))
            best.setdefault(nm, {})[k] = len(boxes)
        L.append("| " + " | ".join(row) + " |")

    L.append("")
    L.append("> 날짜 한 줄은 보통 글자 8~12개다(`26.08.13` 이면 숫자 6 + 구분자 2).")
    L.append("> 그 근처 숫자가 나오면 분할이 된다는 뜻이고, 훨씬 크면 글자가")
    L.append("> 조각나 있고, 훨씬 작으면 서로 붙어 있다는 뜻이다.")
    L.append("")
    L.append(f"커널5 시각화는 `eda/out/108_seg/` 에 남겼다. 빨간 상자가 글자 후보다.")

    with open(os.path.join(ROOT, "eda", "out", "108_dot_segment.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L[6:]))
    print("\n→ eda/out/108_dot_segment.md")


if __name__ == "__main__":
    main()
