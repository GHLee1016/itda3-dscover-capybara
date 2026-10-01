"""
EDA 115단계 — 파인튜닝한 YOLO(v2)가 실제로 나아졌는가

두 가지를 본다. 두 번째가 더 중요하다.
  ① 못 잡던 미검출 이미지에서 날짜 영역을 잡는가
  ② **기존에 잘 잡던 이미지를 망가뜨리지 않는가**

②를 빠뜨리면 안 된다. 합성으로 이어 학습하면 원래 분포를 잊을 수 있다.
보조 경로가 이 검출기를 쓰므로, 여기가 나빠지면 전체가 나빠진다.

    python eda/115_yolo_v2_check.py
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
V1 = os.path.join(ROOT, "weights", "yolo", "date_v1.onnx")
V2 = os.path.join(ROOT, "weights", "yolo", "date_v2.onnx")


def boxes_of(det, combined, np, f):
    img = combined.imread_any(os.path.join(IMG_DIR, f))
    if img is None:
        return None, 0
    h, w = img.shape[:2]
    side = max(h, w)
    canvas = np.zeros((side, side, 3), np.uint8)
    canvas[:h, :w] = img
    try:
        return img, len(det.detect(canvas, conf=0.25))
    except Exception:
        return img, 0


def main():
    import numpy as np
    import combined
    import pipeline as P

    if not os.path.exists(V2):
        print("date_v2.onnx 가 없다. 114단계를 먼저 끝낼 것.")
        return

    d1 = P.OnnxDateDetector(V1, threads=1)
    d2 = P.OnnxDateDetector(V2, threads=1)

    D = json.load(open(os.path.join(OUT, "106_final.json"), encoding="utf-8"))
    lab, pred = D["label"], D["pred"]
    miss = [f for f in sorted(lab)
            if pred.get(f) == "NONE-NONE-NONE" and lab[f] != "NONE-NONE-NONE"]
    # 잘 맞히던 것에서도 박스가 유지되는지 본다
    good = [f for f in sorted(lab) if pred.get(f) == lab[f]][:60]

    print(f"미검출 {len(miss)}장 · 정답 표본 {len(good)}장\n")

    L = ["# EDA 115 — 파인튜닝한 YOLO(v2) 확인\n",
         "①  못 잡던 이미지에서 잡는가   ②  잘 잡던 것을 망가뜨리지 않는가",
         "",
         "## ① 미검출 이미지\n",
         "| 파일 | v1 박스 | v2 박스 | |",
         "| --- | ---: | ---: | --- |"]
    gain = 0
    for f in miss:
        _, n1 = boxes_of(d1, combined, np, f)
        _, n2 = boxes_of(d2, combined, np, f)
        mark = ""
        if n1 == 0 and n2 > 0:
            mark = "**새로 잡음**"
            gain += 1
        elif n1 > 0 and n2 == 0:
            mark = "놓침"
        L.append(f"| `{f}` | {n1} | {n2} | {mark} |")
        print(f"  {f:14s} v1 {n1} → v2 {n2}")

    L.append("")
    L.append(f"못 잡던 것 중 **{gain}장**을 새로 잡는다.")
    L.append("")
    L.append("## ② 잘 맞히던 이미지에서 박스가 유지되는가\n")
    keep = lost = 0
    for f in good:
        _, n1 = boxes_of(d1, combined, np, f)
        _, n2 = boxes_of(d2, combined, np, f)
        if n1 > 0 and n2 == 0:
            lost += 1
        elif n1 > 0 and n2 > 0:
            keep += 1
    L.append(f"- v1 이 잡던 {keep + lost}장 중 v2 가 유지 **{keep}장**, 놓침 **{lost}장**")
    L.append("")
    if lost == 0:
        L.append("기존 분포를 잊지 않았다.")
    else:
        L.append(f"⚠️ {lost}장을 놓쳤다. 합성으로 이어 학습하며 원래 분포가 밀렸다는 뜻이다.")
        L.append("채택하려면 이 손해와 ①의 이득을 견줘야 한다.")

    with open(os.path.join(OUT, "115_yolo_v2_check.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"\n새로 잡은 것 {gain}장 · 기존 유지 {keep} / 놓침 {lost}")
    print("→ eda/out/115_yolo_v2_check.md")


if __name__ == "__main__":
    main()
