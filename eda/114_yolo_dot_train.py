"""
EDA 114단계 — 도트 인쇄를 잡도록 YOLO 를 파인튜닝한다

왜 (EDA 112)
  도트 판독기(CTC)는 되는데 **YOLO 가 미검출 12장 중 7장에서 영역을 못 잡는다.**
  판독기가 아무리 좋아도 그 7장은 손도 못 댄다.
  못 잡는 것들의 공통점: 투명 필름 아래 인쇄 · 연한 회색 도트 · 기울어진 포장.

데이터 (EDA 113)
  실제 상품 사진에 도트 날짜를 합성했다. 붙인 자리가 곧 정답 박스다.
  ⚠️ 배경에 원래 있던 날짜도 **기존 검출기로 찾아 함께 라벨에 넣었다.**
     합성 자리만 라벨로 주면 "저기 날짜가 있는데 정답이 아니다" 를 배워
     검출력이 오히려 떨어진다.
  배경은 라벨이 없는 2,524장에서만 골랐다(평가 표본 오염 방지).

기존 `best.pt` 에서 이어 학습한다. 처음부터 하면 일반 인쇄를 잊는다.
학습은 torch 로 하고 추론은 ONNX 로 한다(EDA 52~53).

    python eda/114_yolo_dot_train.py --epochs 40
"""
import os
import sys
import shutil
import argparse
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = os.path.join(ROOT, "yolo_ds_dot")
OUT = os.path.join(ROOT, "eda", "out")
PREV = os.path.join(ROOT, "yolo_runs", "date_v1", "weights", "best.pt")
DEST = os.path.join(ROOT, "weights", "yolo", "date_v2.onnx")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()

    yaml = os.path.join(DS, "data.yaml")
    if not os.path.exists(yaml):
        print("데이터셋이 없다. 113단계를 먼저 돌릴 것.")
        return
    if not os.path.exists(PREV):
        print("이전 가중치가 없다:", PREV)
        return

    from ultralytics import YOLO
    print(f"기존 가중치에서 이어 학습: {PREV}")
    model = YOLO(PREV)
    model.train(
        data=yaml, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
        device="cpu", workers=2, project=os.path.join(ROOT, "yolo_runs"),
        name="date_v2", exist_ok=True,
        # 합성이 이미 기울기·흐림을 담고 있으므로 증강은 가볍게 둔다
        degrees=5.0, translate=0.08, scale=0.3, fliplr=0.0, mosaic=0.3,
        patience=15, verbose=True, plots=False,
    )

    best = os.path.join(ROOT, "yolo_runs", "date_v2", "weights", "best.pt")
    if not os.path.exists(best):
        print("학습 결과를 찾지 못했다")
        return
    print(f"\nONNX 로 내보낸다: {best}")
    YOLO(best).export(format="onnx", imgsz=a.imgsz, opset=12, simplify=False)
    made = best.replace(".pt", ".onnx")
    if os.path.exists(made):
        os.makedirs(os.path.dirname(DEST), exist_ok=True)
        shutil.copy2(made, DEST)
        mb = os.path.getsize(DEST) / 1024 / 1024
        print(f"→ {DEST}  ({mb:.1f}MB)")
        with open(os.path.join(OUT, "114_yolo_dot_train.md"), "w",
                  encoding="utf-8") as f:
            f.write("\n".join([
                "# EDA 114 — 도트 인쇄를 잡도록 YOLO 파인튜닝\n",
                "미검출 12장 중 7장에서 YOLO 가 영역 자체를 못 잡았다.",
                "합성 도트 날짜 1,600장으로 기존 `best.pt` 에서 이어 학습했다.",
                "",
                f"- 에폭 {a.epochs} · 입력 {a.imgsz} · CPU",
                f"- 산출물 `weights/yolo/date_v2.onnx` ({mb:.1f}MB)",
                "",
                "⚠️ 학습이 끝났다는 것뿐이다. 실제로 더 잡는지는 115단계에서",
                "   미검출 이미지로 확인하고, 기존 이미지를 망가뜨리지 않는지도 본다.",
            ]))
    else:
        print("ONNX 내보내기 실패")


if __name__ == "__main__":
    main()
