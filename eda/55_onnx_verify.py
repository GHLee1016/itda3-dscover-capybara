"""
EDA 55단계 — ONNX 경로 검증: (1) torch 와 같은 박스를 내는가 (2) 빠른가

ONNX 후처리(디코딩·NMS)를 직접 구현했으므로 먼저 **torch 결과와 일치하는지**
확인해야 한다. 후처리가 틀리면 박스가 엉뚱해지고, 정확도 측정에서야 드러난다.
그건 느린 발견 방식이다.

두 검사를 서로 **다른 프로세스**에서 돌린다.
  A) 일치 검사 : torch 를 싣고 두 백엔드의 박스를 IoU 로 맞춰 본다 (속도 무관)
  B) 속도 측정 : torch 를 절대 싣지 않고 ONNX 만으로 장당 시간을 잰다
한 프로세스에서 둘을 하면 B 가 오염된다 — 그게 52단계에서 확인한 문제다.
"""
import os
import sys
import json
import time
import argparse
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "eda", "out")
LIMIT_SEC = 2400

# ---------------------------------------------------------------- A) 일치 검사
AGREE = r'''
import os, sys, json, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT, N = sys.argv[1], int(sys.argv[2])
import torch
from ultralytics import YOLO
import numpy as np, cv2
sys.path.insert(0, os.path.join(ROOT, "src"))
import pipeline as P

IMG = os.path.join(ROOT, "images", "상품사진입니다")
lab = json.load(open(os.path.join(ROOT,"eda","out","truth_holdout2.json"), encoding="utf-8"))["labels"]
files = list(lab)[:N]

pt  = YOLO(os.path.join(ROOT, "yolo_runs", "date_v1", "weights", "best.pt"))
ox  = P.OnnxDateDetector(os.path.join(ROOT, "weights", "yolo", "date_v1.onnx"))

def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1: return 0.0
    inter = (x2-x1)*(y2-y1)
    ar = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ar if ar > 0 else 0.0

n_pt = n_ox = matched = 0
ious = []
for f in files:
    img = P.imread(os.path.join(IMG, f))
    if img is None: continue
    small = P.downscale(img, 640)
    h, w = small.shape[:2]; side = max(h, w)
    canvas = np.full((side, side, 3), 114, np.uint8)
    y0, x0 = (side-h)//2, (side-w)//2
    canvas[y0:y0+h, x0:x0+w] = small
    A = [[float(v) for v in b.xyxy[0]]
         for r in pt.predict(canvas, imgsz=640, conf=0.25, verbose=False, device="cpu")
         for b in getattr(r, "boxes", [])]
    B = ox.detect(canvas, 0.25)
    n_pt += len(A); n_ox += len(B)
    for a in A:
        best = max([iou(a, b) for b in B], default=0.0)
        if best >= 0.7: matched += 1
        ious.append(best)
print("AGREE %d %d %d %.3f" % (n_pt, n_ox, matched,
      (sum(ious)/len(ious)) if ious else 0.0))
'''

# ---------------------------------------------------------------- B) 속도 측정
SPEED = r'''
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
MODE, N, ROOT = sys.argv[1], int(sys.argv[2]), sys.argv[3]
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "eda"))
import pipeline as P, ocrlib
assert "torch" not in sys.modules, "torch 가 실렸다 — 측정이 오염된다"
from paddleocr import PaddleOCR
M = ocrlib.MODELS
ocr = PaddleOCR(use_angle_cls=False, lang="korean", show_log=False, use_gpu=False,
                det_limit_side_len=640, det_limit_type="max",
                det_model_dir=os.path.join(M,"det","ml","Multilingual_PP-OCRv3_det_infer"),
                rec_model_dir=os.path.join(M,"rec","korean","korean_PP-OCRv4_rec_infer"),
                cls_model_dir=os.path.join(M,"cls","ch_ppocr_mobile_v2.0_cls_infer"))
pipe = P.DatePipeline(ocr, long_side=640, rotations=(0,90), adaptive=True, keep=None)
if MODE != "off":
    pipe.yolo = P.OnnxDateDetector(os.path.join(ROOT, "weights", "yolo", "date_v1.onnx"))
    pipe.yolo_conf = 0.25
    pipe.yolo_mode = MODE
assert "torch" not in sys.modules, "torch 가 실렸다"
lab = json.load(open(os.path.join(ROOT,"eda","out","truth_holdout2.json"), encoding="utf-8"))["labels"]
files = list(lab)[:N]
IMG = os.path.join(ROOT, "images", "상품사진입니다")
pipe.predict_path(os.path.join(IMG, files[0]))
t0 = time.time()
for f in files:
    try: pipe.predict_path(os.path.join(IMG, f))
    except Exception: pass
print("RESULT %.4f" % ((time.time()-t0)/len(files)))
'''


def run_child(src, args, name):
    p = os.path.join(os.environ.get("TEMP", "."), name)
    with open(p, "w", encoding="utf-8") as f:
        f.write(src)
    r = subprocess.run([sys.executable, p] + list(args), capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-agree", type=int, default=25)
    ap.add_argument("--n-speed", type=int, default=40)
    a = ap.parse_args()

    print("A) torch vs ONNX 박스 일치 검사", flush=True)
    r = run_child(AGREE, [ROOT, str(a.n_agree)], "_agree.py")
    line = [l for l in (r.stdout or "").splitlines() if l.startswith("AGREE")]
    if not line:
        print((r.stdout or "")[-400:]); print((r.stderr or "")[-900:]); return
    n_pt, n_ox, matched, miou = line[0].split()[1:]
    n_pt, n_ox, matched, miou = int(n_pt), int(n_ox), int(matched), float(miou)
    print(f"   torch {n_pt}개 / ONNX {n_ox}개, IoU≥0.7 일치 {matched}개, "
          f"평균 최대IoU {miou:.3f}", flush=True)

    print("\nB) torch 없이 속도 측정", flush=True)
    res = {}
    for mode in ("off", "fallback", "always"):
        r = run_child(SPEED, [mode, str(a.n_speed), ROOT], "_speed.py")
        v = [l for l in (r.stdout or "").splitlines() if l.startswith("RESULT")]
        if not v:
            print(f"   {mode}: 실패 — {(r.stderr or '')[-300:]}")
            continue
        res[mode] = float(v[0].split()[1])
        print(f"   {mode:9s} {res[mode]:.2f}s/장 → 500장 {res[mode]*500:.0f}초 "
              f"(제한의 {res[mode]*500/LIMIT_SEC*100:.0f}%)", flush=True)

    NAME = {"off": "기준선 (YOLO 없음)", "fallback": "+ ONNX YOLO 최후수단",
            "always": "+ ONNX YOLO 전패스"}
    L = ["# EDA 55 — ONNX 경로 검증 (일치 + 속도)\n",
         "후처리(디코딩·NMS)를 직접 구현했으니 먼저 torch 결과와 같은 박스를",
         "내는지 확인하고, 그다음 torch 를 싣지 않은 상태의 속도를 잰다.",
         "두 검사는 서로 다른 프로세스에서 돌렸다 — 한 프로세스에서 하면",
         "속도 측정이 오염된다(52단계).",
         "",
         "## A) torch ↔ ONNX 박스 일치\n",
         f"- 표본 {a.n_agree}장",
         f"- torch 박스 {n_pt}개, ONNX 박스 {n_ox}개",
         f"- torch 박스 중 IoU ≥ 0.7 로 대응되는 것 **{matched}/{n_pt}개**",
         f"- torch 박스별 최대 IoU 평균 **{miou:.3f}**",
         ""]
    if n_pt and matched / n_pt >= 0.95 and miou >= 0.9:
        L.append("→ 후처리 구현이 맞다. 두 백엔드를 같은 것으로 취급할 수 있다.")
    else:
        L.append("→ ⚠️ 불일치가 있다. 후처리를 다시 봐야 한다. "
                 "정확도 측정 결과를 ONNX 탓으로 단정하면 안 된다.")
    L += ["", "## B) 속도 (torch 미임포트, 단일 프로세스)\n",
          f"표본 {a.n_speed}장.",
          "",
          "| 구성 | 장당 | 500장 | 제한(2400초) 대비 |",
          "| --- | ---: | ---: | ---: |"]
    for m in ("off", "fallback", "always"):
        if m in res:
            L.append(f"| {NAME[m]} | {res[m]:.2f}s | {res[m]*500:.0f}초 | "
                     f"{res[m]*500/LIMIT_SEC*100:.0f}% |")
    L.append("")
    L.append("비교 — torch 로 돌렸을 때(53단계): YOLO 전패스 5.13초/장, 2564초, 제한의 107%")
    L.append("")
    if "always" in res and res["always"] * 500 < LIMIT_SEC * 0.8:
        L.append(f"→ ONNX 로 바꿔 제한의 {res['always']*500/LIMIT_SEC*100:.0f}% 로 "
                 "내려왔다. 시간 제약은 해소됐다. 남은 판단은 정확도뿐이다.")
    elif "always" in res:
        L.append(f"→ 여전히 제한의 {res['always']*500/LIMIT_SEC*100:.0f}% 다. "
                 "전패스는 위험하고, 최후수단 방식만 검토할 수 있다.")

    with open(os.path.join(OUT_DIR, "55_onnx_verify.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n→ eda/out/55_onnx_verify.md")


if __name__ == "__main__":
    main()
