"""
도트 매트릭스 인쇄 전용 분할기.

왜 필요한가 (EDA 107~108)
  남은 미검출의 주된 원인이 도트 인쇄다. YOLO 는 날짜 영역을 정확히 잡는데
  인식기가 그 글꼴을 못 읽는다. `26.08.13` 을 `2:88:13` 으로 읽는다.
  이진화·모폴로지·색분리·확대를 다 해도 같다. **인식기가 도트 글꼴을 학습한
  적이 없는** 것이지 분리가 안 되는 것이 아니다.

  그래서 글자를 하나씩 떼어내 전용 분류기에 넣는다. 이 파일은 그 앞단이다.

세 가지를 이미지마다 맞춰야 한다 (108단계에서 확인)
  · 도트 크기·간격이 제각각이다. 고정 커널로는 어떤 것은 안 이어지고
    어떤 것은 옆 글자까지 붙는다  →  **점 크기를 먼저 재서 커널을 정한다**
  · 날짜가 두 줄인 경우가 많고, 확대하면 위아래가 뭉갠다
                                  →  **수평 투영으로 줄을 먼저 가른다**
  · 글자가 여전히 붙어 있을 수 있다
                                  →  줄 안에서 **수직 투영 골짜기**로 자른다

torch 를 쓰지 않는다. 추론 경로에 torch 가 실리면 파이프라인이 2.4배 느려진다
(EDA 52~53).
"""
import cv2
import numpy as np


def _binarize(gray):
    """도트는 배경보다 어둡다. 글자가 흰색이 되도록 반전 이진화."""
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # 잉크가 배경보다 밝은 경우(어두운 포장에 흰 잉크)를 대비해 뒤집어 본다
    if bw.mean() > 127:
        bw = 255 - bw
    return bw


def _stroke_width(bw):
    """
    획(=점) 두께를 거리 변환으로 잰다.

    ⚠️ 처음에는 연결요소 **높이의 중앙값**을 점 크기로 썼는데 크게 틀렸다.
       이미 획이 이어진 덩어리가 중앙값을 차지해 46~61 같은 값이 나왔고,
       커널이 최대치로 고정돼 글자가 통째로 뭉갰다(EDA 108).

    거리 변환은 전경 픽셀마다 배경까지의 거리를 준다. 그 최댓값이 획 반지름이다.
    점이 격자로 찍힌 인쇄에서는 곧 점 반지름이 된다. 덩어리별 최댓값을 모아
    중앙값을 취하면 이어진 획이 섞여 있어도 잘 버틴다.
    """
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    if n <= 1:
        return 2.0
    dist = cv2.distanceTransform(bw, cv2.DIST_L2, 3)
    rs = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 4:
            continue
        r = dist[lab == i].max()
        if r > 0:
            rs.append(float(r))
    if not rs:
        return 2.0
    rs.sort()
    return max(1.0, rs[len(rs) // 2])


def _rows(closed, min_frac=0.06):
    """수평 투영으로 글자 줄을 가른다. -> [(y1, y2), ...]"""
    prof = (closed > 0).sum(axis=1).astype(np.float32)
    if prof.max() <= 0:
        return []
    thr = max(1.0, prof.max() * min_frac)
    on = prof >= thr
    out, s = [], None
    for i, v in enumerate(on):
        if v and s is None:
            s = i
        elif not v and s is not None:
            out.append((s, i))
            s = None
    if s is not None:
        out.append((s, len(on)))
    H = closed.shape[0]
    return [(a, b) for a, b in out if (b - a) >= H * 0.12]


def _split_wide(band, x, w, h):
    """
    한 덩어리가 글자 여러 개인 경우 수직 투영 골짜기에서 자른다.
    글자는 대체로 높이의 0.45~0.9배 폭이므로 그보다 넓으면 나눈다.
    """
    exp = max(1, int(round(w / max(h * 0.62, 1))))
    if exp <= 1:
        return [(x, w)]
    sub = band[:, x:x + w]
    prof = (sub > 0).sum(axis=0).astype(np.float32)
    cuts = []
    step = w / exp
    for i in range(1, exp):
        c = int(step * i)
        lo, hi = max(1, c - int(step * 0.3)), min(w - 1, c + int(step * 0.3))
        if hi <= lo:
            cuts.append(c)
            continue
        cuts.append(lo + int(np.argmin(prof[lo:hi])))
    out, prev = [], 0
    for c in cuts + [w]:
        if c - prev >= 3:
            out.append((x + prev, c - prev))
        prev = c
    return out or [(x, w)]


def segment(gray, scale=3):
    """
    도트 크롭(그레이스케일) -> 글자 후보 목록.

    반환: [{"img": 48x48 uint8(글자 흰색), "x": 원본기준 x, "y":, "w":, "h":,
            "row": 줄 번호}, ...]  왼쪽→오른쪽, 위→아래 순.
    """
    if gray is None or gray.size == 0:
        return []
    big = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    bw = _binarize(big)

    r = _stroke_width(bw)
    k = int(np.clip(round(r * 2.2), 3, 15))
    if k % 2 == 0:
        k += 1

    # ⚠️ 줄을 **닫기 전에** 갈라야 한다. 닫기부터 하면 위아래 줄이 세로로
    #    이어져 `20` 위에 `20` 이 한 덩어리가 된다(EDA 108에서 실제로 그랬다).
    #    투영을 재려면 점이 어느 정도 이어져 있어야 하므로, 가로로만 살짝
    #    이어 붙인 뒤 수평 투영을 본다. 세로로는 건드리지 않는다.
    hker = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, k), 1))
    for_rows = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, hker)

    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    out = []
    for ri, (y1, y2) in enumerate(_rows(for_rows)):
        # 줄을 먼저 떼어낸 다음 그 안에서만 점을 잇는다
        band = cv2.morphologyEx(bw[y1:y2], cv2.MORPH_CLOSE, ker)
        bh = y2 - y1
        n, _, stats, _ = cv2.connectedComponentsWithStats(band, 8)
        raw = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if h < bh * 0.35:            # 마침표·얼룩
                continue
            if area < (bh * bh) * 0.04:
                continue
            raw.append((x, w, h))
        raw.sort()
        for x, w, h in raw:
            for xx, ww in _split_wide(band, x, w, bh):
                ch = band[:, xx:xx + ww]
                if ch.size == 0 or ch.max() == 0:
                    continue
                ys = np.where(ch.any(axis=1))[0]
                ch = ch[ys[0]:ys[-1] + 1]
                out.append({"img": _fit48(ch), "x": xx, "y": y1,
                            "w": ww, "h": ch.shape[0], "row": ri})
    return out


def _fit48(ch, side=48):
    """종횡비를 지키며 48x48 한가운데에 놓는다. 학습·추론이 같은 규격을 쓴다."""
    h, w = ch.shape[:2]
    s = (side - 8) / max(h, w, 1)
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    r = cv2.resize(ch, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((side, side), np.uint8)
    y0, x0 = (side - nh) // 2, (side - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = r
    return canvas


def debug_draw(gray, scale=3):
    """분할 결과를 눈으로 확인할 시각화 이미지."""
    big = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    bw = _binarize(big)
    r = _stroke_width(bw)
    k = int(np.clip(round(r * 2.2), 3, 15))
    if k % 2 == 0:
        k += 1
    hker = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, k), 1))
    for_rows = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, hker)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = np.zeros_like(bw)
    for y1, y2 in _rows(for_rows):
        closed[y1:y2] = cv2.morphologyEx(bw[y1:y2], cv2.MORPH_CLOSE, ker)
    vis = cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)
    COL = [(0, 0, 255), (0, 200, 0), (255, 128, 0)]
    for c in segment(gray, scale):
        cv2.rectangle(vis, (c["x"], c["y"]), (c["x"] + c["w"], c["y"] + c["h"]),
                      COL[c["row"] % 3], 1)
    return vis, r, k
