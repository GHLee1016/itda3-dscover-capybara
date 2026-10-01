"""
기각된 재시도 칸 — **배포에 쓰지 않는다.**

`src/combined.py` 는 최종 구성만 담는다. 여기 있는 것은 측정해 보고 버린
수단이며, 당시 실험 스크립트(EDA 76 등)가 그대로 돌아가도록 남겨 둔다.
기각 근거는 JOURNAL.md 와 METHOD.md 에 있다.

기각 목록과 사유
  det server 칸      홀드아웃2 에서 0장. 아래 함수 주석 참고
  1600px 칸          홀드아웃4 148장에서 0장 회수, 비용만 30% 증가 (EDA 96)
  8칸 전부 + 다수결   93.6% → 93.4% (EDA 99)
  YOLO 도트 v2       mAP50 0.80 이나 기존 정답 6장 상실 (EDA 115)
  conf 0.25 → 0.05   검출률 4→65% 이나 정확도 92.3 → 92.2% (EDA 119)
"""
import os

from combined import RAPID_FILES

RETRY2_DET = "ch_PP-OCRv5_det_server.onnx"


def build_rapid_retry2(weights_dir):
    """
    세 번째 칸: ch PP-OCRv5 det server (84MB). **기각됐다 (EDA 76).**

    기각 경위: EDA 74 에서 '어려운 실패 252장' 중 det medium 이 못 잡은 17장을
    server 가 잡았다. 그래서 계단을 하나 더 두면 이득이 있을 줄 알았다.
    그런데 홀드아웃2 에서 **0장**이었다.

    이유는 모집단 착각이다. 17/252 는 medium 까지 실패한 이미지 기준이고,
    홀드아웃2 에서 그 단계까지 내려오는 이미지는 9장뿐이다. 기댓값이 0.6장이라
    0장은 예상 범위다. '17장 더 회수' 라는 표현 자체가 잘못된 프레이밍이었다.

    비용은 모델 84MB 와 장당 3.01 → 3.15초. 관측되지 않은 이득에 지불할 값이 아니다.
    """
    det = os.path.join(weights_dir, RETRY2_DET)
    need = {k: os.path.join(weights_dir, v) for k, v in RAPID_FILES.items()}
    need["Det.model_path"] = det
    missing = [os.path.basename(p) for p in need.values() if not os.path.exists(p)]
    if missing:
        return None, "가중치 없음: " + ", ".join(missing)
    try:
        from rapidocr import RapidOCR
        return RapidOCR(params=need), "bundled"
    except Exception as e:
        return None, "로딩 실패: %s" % e
