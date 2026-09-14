"""
전처리 변형 모음.

06~09단계에서 확인한 실패 원인 세 갈래를 각각 겨냥한다.

  (A) 세로/저대비 잉크젯 날짜의 **검출 누락**
      → 잉크젯은 점(dot) 배열로 찍힌다. 점이 떨어져 있으면 det 가 하나의 텍스트줄로
        묶지 못한다. 모폴로지 closing 으로 점을 이어 획을 만든다.
  (B) 곡면(병·캔)의 **불균일 조명**
      → 강한 블러로 만든 배경으로 나눠 조명을 평탄화(flat-field)한다.
  (C) 전반적 **저대비**
      → CLAHE 로 국소 대비를 올리고 언샵으로 경계를 세운다.
"""
import cv2
import numpy as np


def _to_bgr(g):
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def none(a):
    return a


def clahe(a, clip=2.0, tile=8):
    lab = cv2.cvtColor(a, cv2.COLOR_BGR2LAB)
    l, u, v = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(l)
    return cv2.cvtColor(cv2.merge([l, u, v]), cv2.COLOR_LAB2BGR)


def unsharp(a, sigma=2.0, amt=0.6):
    blur = cv2.GaussianBlur(a, (0, 0), sigma)
    return cv2.addWeighted(a, 1 + amt, blur, -amt, 0)


def clahe_unsharp(a):
    return unsharp(clahe(a))


def dot_close(a, k=3):
    """(A) 잉크젯 점 배열을 이어 붙인다. 어두운 글자를 대상으로 closing."""
    g = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(g, cv2.MORPH_CLOSE, ker)   # 밝은 배경의 어두운 점을 메움
    return _to_bgr(closed)


def dot_close5(a):
    return dot_close(a, k=5)


def illum_norm(a, sigma=25):
    """(B) 조명 평탄화: 원본 / 강블러 → 곡면 그림자 제거."""
    g = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32) + 1.0
    bg = cv2.GaussianBlur(g, (0, 0), sigma) + 1.0
    n = np.clip(g / bg * 128.0, 0, 255).astype(np.uint8)
    return _to_bgr(n)


def illum_dot(a):
    return dot_close(illum_norm(a))


def illum_clahe(a):
    return clahe(illum_norm(a))


def blackhat(a, k=9):
    """(A) 밝은 배경 위 어두운 가는 글자를 강조 (black-hat)."""
    g = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, ker)
    out = cv2.normalize(bh, None, 0, 255, cv2.NORM_MINMAX)
    return _to_bgr(255 - out)                              # 다시 '밝은 배경/어두운 글자'로


def adapt_bin(a, block=31, C=10):
    """(C) 국소 이진화. det 에는 도움될 수 있으나 rec 에는 해로울 수 있다."""
    g = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    b = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY, block, C)
    return _to_bgr(b)


def gamma_clahe(a, gamma=0.7):
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], np.uint8)
    return clahe(cv2.LUT(a, lut))


def bilat_unsharp(a):
    b = cv2.bilateralFilter(a, 7, 60, 60)
    return unsharp(b, sigma=1.5, amt=0.8)


VARIANTS = {
    "none": none,
    "clahe": clahe,
    "unsharp": unsharp,
    "clahe+unsharp": clahe_unsharp,
    "dot_close3": dot_close,
    "dot_close5": dot_close5,
    "illum_norm": illum_norm,
    "illum+dot": illum_dot,
    "illum+clahe": illum_clahe,
    "blackhat": blackhat,
    "adapt_bin": adapt_bin,
    "gamma+clahe": gamma_clahe,
    "bilat+unsharp": bilat_unsharp,
}
