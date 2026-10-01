"""
EDA 92단계 — 단일 프로세스 실측 속도

채점은 **500장 / 2,400초**(운영진 Q&A 확정)이고 제출 노트북은 **순차 루프**다.
그런데 88단계의 "장당 2.62초"는 4워커 병렬에서 잰 개별 지연이라 채점 조건과
다르다. 이 프로젝트는 예전에 정확히 같은 함정에 빠진 적이 있다.
    RULES_QNA.md — "4프로세스 병렬 wall-clock 을 장수로 나눈 것으로,
                   처리량이지 단일 프로세스 지연이 아니다"
그래서 여기서는 노트북과 같은 방식으로 한 프로세스에서 순차로 잰다.

같이 보는 것
  · 칸별 분담. 계단을 몇 칸까지 오르는지가 곧 비용이다.
  · 예선 제출본(0.53초/장, 제한의 11%)과 견준다.

    python eda/92_single_speed.py --n 120
"""
import os
import sys
import time
import json
import random
import argparse
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

IMG_DIR = os.path.join(ROOT, "images", "상품사진입니다")
OUT = os.path.join(ROOT, "eda", "out")
ONNX = os.path.join(ROOT, "weights", "yolo", "date_v1.onnx")
RAPID_DIR = os.path.join(ROOT, "weights", "rapidocr")

LIMIT_SEC = 2400
EVAL_N = 500


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()

    import pipeline as P
    import team2_rules
    import combined
    import ocrlib

    rapid, _ = combined.build_rapid_ocr(RAPID_DIR)
    fb = P.DatePipeline(ocrlib.build_ocr(det_side=640, threads=1),
                        long_side=640, rotations=(0, 90), adaptive=True, keep=None)
    if os.path.exists(ONNX):
        fb.yolo = P.OnnxDateDetector(ONNX, threads=1)
        fb.yolo_conf = 0.25
        fb.yolo_mode = "always"
    retry, _ = combined.build_rapid_retry(RAPID_DIR)
    ko, _ = combined.build_rapid_rec(RAPID_DIR, combined.REC_KOREAN)
    en, _ = combined.build_rapid_rec(RAPID_DIR, combined.REC_EN)
    import dotcrnn
    dot, _ = dotcrnn.build_crnn(os.path.join(ROOT, "weights"))
    dbox = P.OnnxDateDetector(ONNX, threads=1) if os.path.exists(ONNX) else None
    pipe = combined.CombinedPipeline(rapid, team2_rules, fb, rapid_retry=retry,
                                     rec_korean=ko, rec_en=en,
                                     dot=dot, dot_boxes=dbox)

    files = sorted(os.listdir(IMG_DIR))
    files = [f for f in files if os.path.splitext(f)[1].lower()
             in (".jpg", ".jpeg", ".png")]
    pick = random.Random(a.seed).sample(files, min(a.n, len(files)))

    print(f"단일 프로세스 · 순차 · {len(pick)}장", flush=True)

    # 첫 장은 모델 워밍업이 섞이므로 따로 잰다
    t0 = time.time()
    pipe.predict(os.path.join(IMG_DIR, pick[0]))
    warm = time.time() - t0

    secs = []
    t_all = time.time()
    for i, f in enumerate(pick[1:], 1):
        t = time.time()
        pipe.predict(os.path.join(IMG_DIR, f))
        secs.append(time.time() - t)
        if i % 30 == 0:
            print(f"  {i}장  누적 평균 {sum(secs)/len(secs):.2f}초", flush=True)
    wall = time.time() - t_all

    secs.sort()
    n = len(secs)
    avg = sum(secs) / n
    p50 = secs[n // 2]
    p90 = secs[int(n * 0.9)]
    p99 = secs[min(int(n * 0.99), n - 1)]

    est = avg * EVAL_N
    L = ["# EDA 92 — 단일 프로세스 실측 속도\n",
         "채점은 500장 / 2,400초이고 제출 노트북은 순차 루프다.",
         "88단계의 2.62초는 4워커 병렬에서 잰 값이라 채점 조건과 다르다.",
         "여기서는 노트북과 같은 방식으로 한 프로세스에서 잰다.",
         "",
         f"- 표본 {n}장 (첫 장 {warm:.2f}초는 워밍업이라 제외)",
         "",
         "| 지표 | 값 |",
         "| --- | ---: |",
         f"| 평균 | **{avg:.2f}초/장** |",
         f"| 중앙값 | {p50:.2f}초 |",
         f"| 상위 10% | {p90:.2f}초 |",
         f"| 상위 1% | {p99:.2f}초 |",
         f"| 최대 | {secs[-1]:.2f}초 |",
         "",
         "## 채점 환산\n",
         "| 항목 | 값 |",
         "| --- | ---: |",
         f"| 500장 예상 | **{est:.0f}초** ({est/60:.1f}분) |",
         f"| 제한 2,400초 대비 | **{est/LIMIT_SEC*100:.0f}%** |",
         f"| 장당 예산 4.8초 대비 | {avg/4.8*100:.0f}% |",
         ""]
    if est > LIMIT_SEC:
        L.append("⚠️ **제한을 넘는다.** 속도 항목이 0점이 된다(정확도는 살아남는다).")
    elif est > LIMIT_SEC * 0.6:
        L.append("제한은 넘지 않지만 여유가 크지 않다. 속도는 '훨씬 빨라야' 만점이다.")
    else:
        L.append("여유가 있다.")
    L.append("")
    L.append("## 칸별 분담\n")
    L.append("계단을 몇 칸까지 오르는지가 곧 비용이다.\n")
    L.append("| 칸 | 장수 | 비율 |")
    L.append("| --- | ---: | ---: |")
    tot = sum(pipe.stats.values()) or 1
    KO = {"rapid": "1칸 주력 det small", "adaptive": "2칸 adaptive",
          "retry": "3칸 det medium", "clahe": "4칸 CLAHE+medium",
          "rec_ko": "5칸 rec 한국어", "rec_en": "6칸 rec 영문",
          "dot": "7칸 도트 판독기",
          "big_med": "(기각) 1600 medium", "big_small": "(기각) 1600 small",
          "retry2": "(기각) det server", "partial": "부분 결과",
          "fallback": "보조 경로(PP-OCR+YOLO)", "none": "실패"}
    for k, v in pipe.stats.items():
        L.append(f"| {KO.get(k, k)} | {v} | {v/tot*100:.0f}% |")
    L.append("")
    L.append("## 예선 제출본과 견주기\n")
    L.append("| 구성 | 장당 | 500장 | 제한 대비 |")
    L.append("| --- | ---: | ---: | ---: |")
    L.append("| 1차 예선 제출본 (RapidOCR v4 주 + 보조) | 0.53초 | 267초 | 11% |")
    L.append(f"| **현재 (v6 + 계단 4칸)** | **{avg:.2f}초** | "
             f"**{est:.0f}초** | **{est/LIMIT_SEC*100:.0f}%** |")
    L.append("")
    L.append(f"> 벽시계 {wall:.0f}초 / {n}장 = {wall/n:.2f}초. 위 평균과 같아야 한다.")

    with open(os.path.join(OUT, "92_single_speed.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print()
    print("\n".join(L[5:]))
    print("\n→ eda/out/92_single_speed.md")


if __name__ == "__main__":
    main()
