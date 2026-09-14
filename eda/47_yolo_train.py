"""
EDA 47단계 — YOLOv8n 날짜 영역 검출기 학습

학습 표본이 130장뿐이라 과적합이 기본값이다. 완화 수단
  - 가장 작은 모델(yolov8n, 3.2M 파라미터)
  - 증강을 강하게 (HSV·회전·스케일·이동·모자이크)
  - 조기 종료(patience)
  - 단일 클래스

CPU 학습이므로 느리다. epochs 를 조절해 시간을 맞춘다.
추론 비용은 640px 에서 40~80ms 수준이라 예산(4.8초/장)에 영향이 없다.
"""
import os
import sys
import time
import argparse
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = os.path.join(ROOT, "yolo_ds")
RUNS = os.path.join(ROOT, "yolo_runs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--name", default="date_v1")
    a = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")          # COCO 사전학습에서 출발
    t0 = time.time()
    model.train(
        data=os.path.join(DS, "data.yaml"),
        epochs=a.epochs,
        imgsz=a.imgsz,
        batch=a.batch,
        workers=a.workers,
        device="cpu",
        project=RUNS,
        name=a.name,
        exist_ok=True,
        patience=30,
        # --- 표본이 작아 증강을 강하게 ---
        hsv_h=0.02, hsv_s=0.8, hsv_v=0.5,
        degrees=8.0,          # 포장지가 기울어 찍히는 경우가 많다
        translate=0.15,
        scale=0.6,            # 날짜 영역 크기가 제각각이다
        shear=3.0,
        perspective=0.0006,   # 곡면·비스듬한 촬영
        fliplr=0.0,           # 글자는 좌우반전하면 안 된다
        flipud=0.0,
        mosaic=0.8,
        close_mosaic=15,
        # --- 작은 물체 위주라 box 손실 비중을 올린다 ---
        box=9.0, cls=0.4, dfl=1.5,
        lr0=0.002, lrf=0.02, warmup_epochs=5,
        val=True, plots=False, verbose=True,
    )
    print(f"\n학습 {time.time()-t0:.0f}s")
    best = os.path.join(RUNS, a.name, "weights", "best.pt")
    print("best:", best, os.path.exists(best))


if __name__ == "__main__":
    main()
