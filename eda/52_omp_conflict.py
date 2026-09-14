"""
EDA 52단계 — torch 를 같이 싣는 것만으로 파이프라인이 느려지는지 확인

50단계에서 기준선(YOLO 없음)이 장당 5.81초로 나왔다. 22단계의 단일 프로세스
측정(960px, 1.29초/장)과 4배 이상 차이가 난다. 설정 차이로는 설명이 안 된다.

의심: torch 와 paddle 이 각자 OpenMP 런타임(libiomp5md.dll)을 싣는다.
      KMP_DUPLICATE_LIB_OK=TRUE 로 둘을 공존시키면 스레드가 서로 경합해
      전체가 느려지는 것이 알려진 현상이다. 50단계는 `off` 구성에서도
      모듈 최상단에서 torch 를 임포트했으므로 기준선까지 오염됐을 수 있다.

검증: 같은 40장을, torch 임포트 없이 / torch 임포트하고 / YOLO 까지 돌리고
      각각 **별도 프로세스**에서 잰다. 한 프로세스 안에서 비교하면 이미
      torch 가 실린 상태라 의미가 없다.

이건 속도 보고를 바로잡는 문제가 아니라 **YOLO 를 쓸 수 있느냐**의 문제다.
"""
import os
import sys
import json
import time
import argparse
import subprocess
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "eda", "out")
LIMIT_SEC = 2400

CHILD = r'''
import os, sys, time, json, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
MODE = sys.argv[1]
N = int(sys.argv[2])
ROOT = sys.argv[3]
if MODE in ("torch", "yolo"):
    import torch                      # paddle 보다 먼저 (DLL 로딩 제약)
    from ultralytics import YOLO
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
if MODE == "yolo":
    pipe.yolo = YOLO(os.path.join(ROOT, "yolo_runs", "date_v1", "weights", "best.pt"))
    pipe.yolo_conf = 0.25
    pipe.yolo_mode = "always"
lab = json.load(open(os.path.join(ROOT,"eda","out","truth_holdout2.json"), encoding="utf-8"))["labels"]
files = list(lab)[:N]
IMG = os.path.join(ROOT, "images", "상품사진입니다")
pipe.predict_path(os.path.join(IMG, files[0]))      # 워밍업
t0 = time.time()
for f in files:
    try: pipe.predict_path(os.path.join(IMG, f))
    except Exception: pass
print("RESULT %.4f" % ((time.time()-t0)/len(files)))
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()

    child = os.path.join(os.environ.get("TEMP", "."), "_omp_child.py")
    with open(child, "w", encoding="utf-8") as f:
        f.write(CHILD)

    NAME = {"clean": "torch 미임포트 (현재 제출 구성)",
            "torch": "torch 임포트만 (YOLO 미사용)",
            "yolo": "torch + YOLO 전패스"}
    res = {}
    for mode in ("clean", "torch", "yolo"):
        p = subprocess.run([sys.executable, child, mode, str(a.n), ROOT],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
        val = None
        for ln in (p.stdout or "").splitlines():
            if ln.startswith("RESULT"):
                val = float(ln.split()[1])
        if val is None:
            print(f"  {mode}: 실패\n{(p.stdout or '')[-500:]}\n{(p.stderr or '')[-800:]}")
            continue
        res[mode] = val
        print(f"  {NAME[mode]}: {val:.2f}s/장 → 500장 {val*500:.0f}초", flush=True)

    L = ["# EDA 52 — torch 동거로 인한 속도 저하 확인\n",
         "50단계 기준선이 장당 5.81초로 나왔다. 22단계 단일 프로세스 측정",
         "(960px, 1.29초/장)과 4배 이상 차이라 설정 차이로 설명되지 않는다.",
         "",
         "의심한 원인은 OpenMP 런타임 중복이다. torch 와 paddle 이 각자",
         "`libiomp5md.dll` 을 싣고, `KMP_DUPLICATE_LIB_OK=TRUE` 로 공존시키면",
         "스레드가 경합한다. 50단계는 `off` 구성에서도 torch 를 임포트했으므로",
         "기준선까지 오염됐을 수 있다.",
         "",
         f"같은 {a.n}장을 **별도 프로세스**에서 세 조건으로 쟀다.",
         "",
         "| 조건 | 장당 | 500장 환산 | 제한(2400초) 대비 |",
         "| --- | ---: | ---: | ---: |"]
    for m in ("clean", "torch", "yolo"):
        if m in res:
            t = res[m] * 500
            L.append(f"| {NAME[m]} | {res[m]:.2f}s | {t:.0f}초 | {t/LIMIT_SEC*100:.0f}% |")
    L.append("")
    if "clean" in res and "torch" in res:
        r = res["torch"] / res["clean"]
        L.append(f"torch 를 임포트하기만 해도 **{r:.1f}배** 느려진다."
                 if r > 1.3 else
                 f"torch 임포트의 영향은 {r:.2f}배로 크지 않다. 느린 원인은 다른 데 있다.")
    L.append("")
    if "clean" in res:
        L.append(f"- 현재 제출 구성(torch 없음)의 실제 단일 프로세스 속도는 "
                 f"**장당 {res['clean']:.2f}초, 500장 {res['clean']*500:.0f}초** "
                 f"(제한의 {res['clean']*500/LIMIT_SEC*100:.0f}%)")
        L.append("- 기존 문서의 '0.99초/장'은 4프로세스 병렬 wall-clock 을 장수로 나눈 값이다.")
        L.append("  즉 처리량이지 단일 프로세스 지연이 아니다. 표기를 바로잡아야 한다.")

    with open(os.path.join(OUT_DIR, "52_omp_conflict.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n→ eda/out/52_omp_conflict.md")


if __name__ == "__main__":
    main()
