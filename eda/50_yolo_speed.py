"""
EDA 50단계 — YOLO 를 붙인 뒤의 실제 추론 속도

48단계가 낸 '장당 7.7초' 는 **워커 4개가 CPU 를 나눠 쓰는 상태의 벽시계 시간**이다.
채점 환경은 단일 프로세스로 한 장씩 돌므로 그 수치를 500배 하면 안 된다.
(그대로 하면 제한의 160% 라는 엉뚱한 결론이 나온다)

여기서는 **프로세스 하나, 스레드 전부** 로 재서 제출 환경과 맞춘다.
실패 이미지가 비싼 경로(해상도 래더 + 전처리 + YOLO)를 타므로,
성공/실패가 섞인 홀드아웃2 표본을 그대로 쓴다.
"""
import os
import sys
import json
import time
import argparse
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ⚠️ torch 는 paddle 보다 **먼저** 임포트해야 한다.
#    순서를 뒤집으면 torch\lib\shm.dll 의존성 로딩이 실패한다
#    (OSError WinError 127). 둘이 같은 MKL/OpenMP 런타임을 건드리기 때문이다.
#    제출 노트북에서도 이 순서를 지켜야 한다.
import torch                                             # noqa: E402  (순서 고정)
from ultralytics import YOLO                             # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "eda"))
import pipeline as P                                     # noqa: E402
import ocrlib                                            # noqa: E402

IMG_DIR = os.path.join(ROOT, "images", "상품사진입니다")
OUT_DIR = os.path.join(ROOT, "eda", "out")
WEIGHTS = os.path.join(ROOT, "yolo_runs", "date_v1", "weights", "best.pt")
LIMIT_SEC = 2400          # 40분


def build_ocr_like_notebook():
    """
    제출 노트북과 **똑같은** 설정으로 만든다.

    ocrlib.build_ocr 은 cpu_threads 를 명시로 넘기는데(병렬 실험용), 노트북은
    그 인자를 아예 주지 않아 PaddleOCR 기본값(멀티스레드)을 쓴다.
    threads=0 으로 재면 장당 5.8초가 나와 실제(0.99초)의 6배로 잡힌다.
    속도를 논하려면 채점 환경과 같은 설정이어야 한다.
    """
    from paddleocr import PaddleOCR
    M = ocrlib.MODELS
    return PaddleOCR(
        use_angle_cls=False, lang="korean", show_log=False, use_gpu=False,
        det_limit_side_len=640, det_limit_type="max",
        det_model_dir=os.path.join(M, "det", "ml", "Multilingual_PP-OCRv3_det_infer"),
        rec_model_dir=os.path.join(M, "rec", "korean", "korean_PP-OCRv4_rec_infer"),
        cls_model_dir=os.path.join(M, "cls", "ch_ppocr_mobile_v2.0_cls_infer"),
    )


def run(mode, files, conf):
    pipe = P.DatePipeline(build_ocr_like_notebook(),
                          long_side=640, rotations=(0, 90),
                          adaptive=True, keep=None)
    if mode != "off":
        pipe.yolo = YOLO(WEIGHTS)
        pipe.yolo_conf = conf
        pipe.yolo_mode = mode
    # 첫 장은 모델 로딩·캐시 워밍업이 섞이므로 버린다
    pipe.predict_path(os.path.join(IMG_DIR, files[0]))
    t0 = time.time()
    for f in files:
        try:
            pipe.predict_path(os.path.join(IMG_DIR, f))
        except Exception:
            pass
    return (time.time() - t0) / len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--conf", type=float, default=0.25)
    a = ap.parse_args()

    lab = json.load(open(os.path.join(OUT_DIR, "truth_holdout2.json"),
                         encoding="utf-8"))["labels"]
    files = list(lab)[:a.n]
    print(f"단일 프로세스, {len(files)}장으로 측정", flush=True)

    res = {}
    for mode in ("off", "fallback", "always"):
        res[mode] = run(mode, files, a.conf)
        print(f"  {mode:9s} {res[mode]:.2f}s/장 → 500장 {res[mode]*500:.0f}초", flush=True)

    L = ["# EDA 50 — YOLO 적용 후 실제 추론 속도\n",
         "48단계의 '장당 7.7초' 는 워커 4개가 CPU 를 나눠 쓴 벽시계 시간이다.",
         "채점은 단일 프로세스이므로 그 수치를 500배 하면 안 된다. 여기서 다시 잰다.",
         "",
         f"표본 {len(files)}장(홀드아웃2 앞부분, 성공·실패 혼재), 프로세스 1개.",
         "PaddleOCR 설정은 제출 노트북과 동일하다(cpu_threads 지정 없음 = 기본값).",
         "`cpu_threads=0` 으로 재면 장당 5.8초가 나와 실제의 6배로 잡힌다 — 그 함정을 피했다.",
         "",
         "| 구성 | 장당 | 500장 추정 | 제한(2400초) 대비 |",
         "| --- | ---: | ---: | ---: |"]
    name = {"off": "기준선 (det 만)", "fallback": "+ YOLO 최후수단",
            "always": "+ YOLO 전패스"}
    for m in ("off", "fallback", "always"):
        t = res[m] * 500
        L.append(f"| {name[m]} | {res[m]:.2f}s | {t:.0f}초 | {t/LIMIT_SEC*100:.0f}% |")
    L.append("")
    over = res["always"] - res["off"]
    L.append(f"- 전패스 추가 비용 장당 {over:+.2f}초 ({over/max(res['off'],1e-9)*100:+.0f}%)")
    L.append(f"- 최후수단 추가 비용 장당 {res['fallback']-res['off']:+.2f}초")
    L.append("")
    worst = max(res.values()) * 500
    if worst > LIMIT_SEC * 0.8:
        L.append(f"⚠️ 가장 비싼 구성이 제한의 {worst/LIMIT_SEC*100:.0f}% 다. "
                 "TimeBudget 이 해상도·2차패스 비율을 깎아 맞추지만 여유가 적다.")
    else:
        L.append(f"여유가 있다. 가장 비싼 구성도 제한의 {worst/LIMIT_SEC*100:.0f}% 다.")

    with open(os.path.join(OUT_DIR, "50_yolo_speed.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n→ eda/out/50_yolo_speed.md")


if __name__ == "__main__":
    main()
