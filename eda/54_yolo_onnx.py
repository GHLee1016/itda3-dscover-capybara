"""
EDA 54단계 — YOLO 를 ONNX 로 내보내 torch 없이 돌린다

52~53단계에서 확인한 것: torch 를 **임포트하기만 해도** 파이프라인이 2.4배
느려진다(2.76 → 5.88 초/장). paddle 과 torch 가 각자 OpenMP 런타임을 싣고
스레드를 다투기 때문이고, set_num_threads·KMP_BLOCKTIME 로는 해결되지 않았다.
torch 를 쓰면 500장이 제한(2400초)을 넘는다.

해법: 가중치를 ONNX 로 내보내고 추론은 onnxruntime 으로 한다.
      내보내기는 오프라인 작업이라 torch 를 써도 되고, 추론 시점에는
      torch 를 임포트하지 않는다.

ONNX 후처리는 직접 해야 한다. YOLOv8 의 ONNX 출력은
  (1, 4+nc, 8400)  — 앞 4채널이 cx,cy,w,h (입력 픽셀 단위), 나머지가 클래스 점수
이므로 점수 문턱 + NMS 를 직접 적용한다.
"""
import os
import sys
import time
import argparse
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "yolo_runs", "date_v1", "weights")
PT = os.path.join(RUNS, "best.pt")
ONNX = os.path.join(ROOT, "weights", "yolo", "date_v1.onnx")


def export():
    """best.pt → ONNX. torch 가 필요하지만 이 단계는 오프라인이라 무관하다."""
    if os.path.exists(ONNX):
        print(f"이미 있음: {ONNX}")
        return
    os.makedirs(os.path.dirname(ONNX), exist_ok=True)
    from ultralytics import YOLO
    m = YOLO(PT)
    # opset 12 는 onnxruntime 1.x 어디서나 돌아간다. simplify 는 onnxsim 의존이라 끈다.
    out = m.export(format="onnx", imgsz=640, opset=12, simplify=False, dynamic=False)
    import shutil
    shutil.copyfile(out, ONNX)
    print(f"내보냄: {ONNX}  ({os.path.getsize(ONNX)/1e6:.1f}MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="ONNX 출력 형태만 확인")
    a = ap.parse_args()

    export()
    if not a.check:
        return
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 0
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    s = ort.InferenceSession(ONNX, so, providers=["CPUExecutionProvider"])
    i = s.get_inputs()[0]
    o = s.get_outputs()[0]
    print(f"입력  {i.name} {i.shape} {i.type}")
    print(f"출력  {o.name} {o.shape} {o.type}")
    x = np.zeros((1, 3, 640, 640), np.float32)
    t0 = time.time()
    for _ in range(5):
        y = s.run(None, {i.name: x})
    print(f"추론 {(time.time()-t0)/5*1000:.0f}ms/회, 출력 {y[0].shape}")


if __name__ == "__main__":
    main()
