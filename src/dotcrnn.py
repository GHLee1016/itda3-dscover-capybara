"""
도트 인쇄 판독기 (CTC 시퀀스). 글자를 자르지 않고 줄 단위로 통째로 읽는다.

왜 분할을 버렸나 (EDA 110)
  글자 분류기는 숫자를 잘 맞혔다(`2022.07` 의 `2207` 을 확신 1.00). 그런데
  최종 정답은 0장이었다. 날짜를 맞히려면 여덟 글자가 **전부** 정확히 잘려야
  하고, 하나만 어긋나면 전체가 틀리기 때문이다.

  CTC 는 입력 폭을 따라 문자열을 내보내므로 자를 필요가 없다.

추론에 torch 를 싣지 않는다. onnxruntime 만 쓴다(EDA 52~53).
"""
import os

import cv2
import numpy as np

import dotseg

H, W = 32, 192
CHARS = "0123456789./- "
VOCAB = [""] + list(CHARS)


def _decode(seq):
    """CTC 그리디: 반복을 접고 blank(0)를 뺀다."""
    out, prev = [], -1
    for i in seq:
        if i != prev and i != 0:
            out.append(VOCAB[i])
        prev = i
    return "".join(out)


def _prep(gray):
    """
    합성 학습과 같은 모양으로 맞춘다. 학습 데이터는 이진화된 흰 글자였으므로
    실제 크롭도 같은 규격으로 넣어야 한다.
    """
    bw = dotseg._binarize(gray)
    ys, xs = np.where(bw > 0)
    if len(ys) < 10:
        return None
    bw = bw[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = bw.shape
    s = (H - 6) / max(h, 1)
    nw = max(8, min(W - 6, int(w * s)))
    r = cv2.resize(bw, (nw, H - 6), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((H, W), np.uint8)
    canvas[3:3 + (H - 6), 3:3 + nw] = r
    return canvas


class DotCRNN:
    def __init__(self, onnx_path, threads=1):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(onnx_path, so,
                                         providers=["CPUExecutionProvider"])
        self.iname = self.sess.get_inputs()[0].name

    def read(self, gray):
        """
        도트 크롭(그레이) -> 줄별 문자열.

        줄은 dotseg 가 쓰는 것과 같은 수평 투영으로 가른다. 글자를 자르는 것이
        아니라 **줄만** 나누는 것이라 110단계의 분할 문제를 겪지 않는다.
        """
        if gray is None or gray.size == 0:
            return []
        big = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        bw = dotseg._binarize(big)
        r = dotseg._stroke_width(bw)
        k = int(np.clip(round(r * 2.2), 3, 15))
        hker = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, k), 1))
        rows = dotseg._rows(cv2.morphologyEx(bw, cv2.MORPH_CLOSE, hker))

        bands = [big[y1:y2] for y1, y2 in rows] or [big]
        tiles, keep = [], []
        for b in bands:
            t = _prep(b)
            if t is not None:
                tiles.append(t)
                keep.append(b)
        if not tiles:
            return []

        # ⚠️ 한 장씩 넣는다. GRU 가 ONNX 로 나가면서 배치 차원이 1 로 굳어져,
        #    여러 장을 한 번에 넣으면 Reshape 에서 죽는다. 줄 수가 한둘이라
        #    낱개로 돌려도 비용 차이가 없다.
        out = []
        for t in tiles:
            x = t.astype(np.float32)[None, None] / 255.0
            try:
                logits = self.sess.run(None, {self.iname: x})[0]
            except Exception:
                continue
            s = _decode(logits[0].argmax(axis=1)).strip()
            if sum(c.isdigit() for c in s) >= 4:
                out.append(s)
        return out


def build_crnn(weights_dir, threads=1):
    """(DotCRNN|None, 상태문자열). 파일이 없으면 만들지 않는다."""
    p = os.path.join(weights_dir, "dot", "dotcrnn_v1.onnx")
    if not os.path.exists(p):
        return None, "가중치 없음: dot/dotcrnn_v1.onnx"
    try:
        return DotCRNN(p, threads=threads), "bundled"
    except Exception as e:
        return None, "로딩 실패: %s" % e
