"""
EDA 53단계 — torch 동거로 생긴 2.4배 저하를 없앨 수 있나

52단계 결과: torch 를 임포트하기만 해도 2.42 → 5.79 초/장 (YOLO 미사용).
OpenMP 런타임이 둘 실리면서 스레드가 경합한 것이다.

시도하는 완화책
  1) torch.set_num_threads(1)      — torch 쪽 스레드를 1개로 묶는다
  2) KMP_BLOCKTIME=0               — 일 끝낸 OMP 스레드가 스핀하지 않고 바로 잠든다
                                     (기본 200ms 스핀이 상대 런타임을 굶긴다)
  3) 1 + 2 동시
  4) onnxruntime 로 YOLO 실행      — torch 를 아예 안 싣는다 (가장 확실하지만
                                     ONNX 내보내기가 필요하다)

목표는 '기준선 2.42초/장에 YOLO 추론 비용만 더한 수준'이다.
"""
import os
import sys
import time
import argparse
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "eda", "out")
LIMIT_SEC = 2400

CHILD = r'''
import os, sys, time, json, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
MODE, N, ROOT = sys.argv[1], int(sys.argv[2]), sys.argv[3]

# 환경 변수는 네이티브 라이브러리가 실리기 **전에** 정해야 한다
if "blocktime" in MODE:
    os.environ["KMP_BLOCKTIME"] = "0"
    os.environ["OMP_WAIT_POLICY"] = "PASSIVE"

use_torch = MODE != "clean"
if use_torch:
    import torch
    from ultralytics import YOLO
    if "t1" in MODE:
        torch.set_num_threads(1)

sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "eda"))
import pipeline as P, ocrlib
from paddleocr import PaddleOCR
M = ocrlib.MODELS
ocr = PaddleOCR(use_angle_cls=False, lang="korean", show_log=False, use_gpu=False,
                det_limit_side_len=640, det_limit_type="max",
                det_model_dir=os.path.join(M,"det","ml","Multilingual_PP-OCRv3_det_infer"),
                rec_model_dir=os.path.join(M,"rec","korean","korean_PP-OCRv4_rec_infer"),
                cls_model_dir=os.path.join(M,"cls","ch_ppocr_mobile_v2.0_cls_infer"))
pipe = P.DatePipeline(ocr, long_side=640, rotations=(0,90), adaptive=True, keep=None)
if use_torch and "yolo" in MODE:
    pipe.yolo = YOLO(os.path.join(ROOT, "yolo_runs", "date_v1", "weights", "best.pt"))
    pipe.yolo_conf = 0.25
    pipe.yolo_mode = "always"

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

CASES = [
    ("clean",                    "기준선 — torch 미임포트"),
    ("torch",                    "torch 임포트만"),
    ("torch_t1",                 "torch + set_num_threads(1)"),
    ("torch_blocktime",          "torch + KMP_BLOCKTIME=0"),
    ("torch_t1_blocktime",       "torch + 둘 다"),
    ("yolo_t1_blocktime",        "YOLO 전패스 + 둘 다"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()

    child = os.path.join(os.environ.get("TEMP", "."), "_omp_fix_child.py")
    with open(child, "w", encoding="utf-8") as f:
        f.write(CHILD)

    res = {}
    for mode, label in CASES:
        p = subprocess.run([sys.executable, child, mode, str(a.n), ROOT],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
        val = None
        for ln in (p.stdout or "").splitlines():
            if ln.startswith("RESULT"):
                val = float(ln.split()[1])
        if val is None:
            print(f"  {label}: 실패 — {(p.stderr or '')[-300:]}")
            continue
        res[mode] = val
        print(f"  {label}: {val:.2f}s/장 → 500장 {val*500:.0f}초", flush=True)

    base = res.get("clean")
    L = ["# EDA 53 — torch 동거 속도 저하 완화\n",
         "52단계에서 torch 를 임포트하기만 해도 2.42 → 5.79 초/장이 됐다.",
         "OpenMP 런타임이 둘 실려 스레드가 경합한 것이다. 완화책을 하나씩 재 본다.",
         "",
         f"표본 {a.n}장, 조건마다 **별도 프로세스**(한 프로세스 안에서는 이미 실린",
         "라이브러리를 되돌릴 수 없다). 환경 변수는 임포트 전에 설정했다.",
         "",
         "| 조건 | 장당 | 기준선 대비 | 500장 | 제한 대비 |",
         "| --- | ---: | ---: | ---: | ---: |"]
    for mode, label in CASES:
        if mode not in res:
            continue
        v = res[mode]
        rel = f"{v/base:.2f}배" if base else "-"
        L.append(f"| {label} | {v:.2f}s | {rel} | {v*500:.0f}초 | "
                 f"{v*500/LIMIT_SEC*100:.0f}% |")
    L.append("")

    best = None
    for mode, label in CASES:
        if mode.startswith("yolo") and mode in res:
            best = (mode, label, res[mode])
    if best and base:
        ok = best[2] * 500 < LIMIT_SEC * 0.8
        L.append(f"**{best[1]}** → 500장 {best[2]*500:.0f}초 "
                 f"(제한의 {best[2]*500/LIMIT_SEC*100:.0f}%). "
                 + ("여유가 있어 쓸 수 있다." if ok else
                    "제한에 너무 가깝다. ONNX 로 torch 를 떼거나 YOLO 를 포기해야 한다."))
    L.append("")
    L.append("## 별도로 바로잡을 것\n")
    L.append("기존 문서의 '0.99초/장, 500장 493초, 제한의 21%' 는 **4프로세스 병렬**")
    L.append("wall-clock 을 장수로 나눈 값이다. 처리량이지 단일 프로세스 지연이 아니다.")
    if base:
        L.append(f"노트북과 같은 단일 순차 루프의 실제 값은 **장당 {base:.2f}초, "
                 f"500장 {base*500:.0f}초(제한의 {base*500/LIMIT_SEC*100:.0f}%)** 다.")
        L.append("YOLO 와 무관하게 이 수치로 고쳐야 한다. 여전히 여유는 있다.")

    with open(os.path.join(OUT_DIR, "53_omp_fix.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n→ eda/out/53_omp_fix.md")


if __name__ == "__main__":
    main()
