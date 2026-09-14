"""OCR 파이프라인 공용 유틸 (여러 실험 스크립트에서 재사용)."""
import os
import numpy as np
import cv2

MODELS = r"C:\Users\Public\ocr_models"


def build_ocr(det_side=960, threads=1, angle_cls=False):
    from paddleocr import PaddleOCR
    return PaddleOCR(
        use_angle_cls=angle_cls, lang="korean", show_log=False, use_gpu=False,
        det_limit_side_len=det_side, det_limit_type="max", cpu_threads=threads,
        det_model_dir=os.path.join(MODELS, "det", "ml", "Multilingual_PP-OCRv3_det_infer"),
        rec_model_dir=os.path.join(MODELS, "rec", "korean", "korean_PP-OCRv4_rec_infer"),
        cls_model_dir=os.path.join(MODELS, "cls", "ch_ppocr_mobile_v2.0_cls_infer"),
    )


def imread(path):
    """한글 경로 대응. cv2 는 EXIF orientation 을 자동 적용한다."""
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)


def imwrite(path, img, q=90):
    ext = os.path.splitext(path)[1] or ".jpg"
    ok, buf = cv2.imencode(ext, img, [cv2.IMWRITE_JPEG_QUALITY, q] if ext in (".jpg", ".jpeg") else [])
    if ok:
        buf.tofile(path)
    return ok


def downscale(a, long_side=960):
    h, w = a.shape[:2]
    s = long_side / max(h, w)
    if s < 1.0:
        a = cv2.resize(a, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    return a


def crop_quad(img, quad, pad=0.0):
    q = np.array(quad, dtype=np.float32)
    w = int(max(np.linalg.norm(q[0] - q[1]), np.linalg.norm(q[2] - q[3])))
    h = int(max(np.linalg.norm(q[0] - q[3]), np.linalg.norm(q[1] - q[2])))
    if w < 4 or h < 4:
        return None
    if pad:
        c = q.mean(axis=0)
        q = c + (q - c) * (1.0 + pad)
        w, h = int(w * (1 + pad)), int(h * (1 + pad))
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    out = cv2.warpPerspective(img, cv2.getPerspectiveTransform(q, dst), (w, h),
                              borderMode=cv2.BORDER_REPLICATE)
    if h > w * 1.5:
        out = cv2.rotate(out, cv2.ROTATE_90_CLOCKWISE)
    return out


def fit_rec_height(crop, target=48, max_w=1600):
    h, w = crop.shape[:2]
    if h < 1:
        return None
    s = target / h
    nw = max(8, min(max_w, int(w * s)))
    return cv2.resize(crop, (nw, target),
                      interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)


ROT = {0: None, 90: cv2.ROTATE_90_CLOCKWISE,
       180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def rotate(a, deg):
    return a if deg == 0 else cv2.rotate(a, ROT[deg])
