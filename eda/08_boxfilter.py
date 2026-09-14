"""
EDA 8단계 — 박스 사전 필터 (CPU 예산의 핵심)

07단계 결과: rec 가 전체 비용의 80.9%, 이미지당 박스 41개 × 16.6ms.
그런데 우리가 필요한 박스는 '날짜가 든 것' 1~2개뿐이다.
→ **rec 를 돌리기 전에** 날짜일 리 없는 박스를 버리면 비용이 그만큼 줄어든다.

문제: 무엇이 날짜인지는 rec 를 해봐야 안다. 그래서 rec 없이 쓸 수 있는
     싸구려 영상 특징만으로 '숫자 줄일 가능성'을 판별해야 한다.

측정 방법 (사람 라벨 불필요):
  1. 이미지 N장에서 det → 모든 박스에 rec 수행 (정답 생성용, 느려도 됨)
  2. rec 결과로 날짜가 파싱되는 박스 = 양성(positive)
  3. 각 크롭에서 rec 없이 계산 가능한 특징 추출
  4. 분류기를 학습해 '양성 박스를 놓치지 않으면서 몇 %를 버릴 수 있는가' 측정

핵심 지표: **양성 박스 재현율 100% 근처를 유지하면서 rec 대상을 몇 배 줄이는가**
"""
import os
import sys
import time
import json
import argparse
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "eda"))
from datex import find_dates

IMG_DIR = os.path.join(ROOT, "images", "상품사진입니다")
OUT_DIR = os.path.join(ROOT, "eda", "out")
MODELS = r"C:\Users\Public\ocr_models"


def build():
    from paddleocr import PaddleOCR
    return PaddleOCR(
        use_angle_cls=False, lang="korean", show_log=False, use_gpu=False,
        det_limit_side_len=960, det_limit_type="max",
        det_model_dir=os.path.join(MODELS, "det", "ml", "Multilingual_PP-OCRv3_det_infer"),
        rec_model_dir=os.path.join(MODELS, "rec", "korean", "korean_PP-OCRv4_rec_infer"),
        cls_model_dir=os.path.join(MODELS, "cls", "ch_ppocr_mobile_v2.0_cls_infer"),
    )


def crop_quad(img, quad):
    q = np.array(quad, dtype=np.float32)
    w = int(max(np.linalg.norm(q[0] - q[1]), np.linalg.norm(q[2] - q[3])))
    h = int(max(np.linalg.norm(q[0] - q[3]), np.linalg.norm(q[1] - q[2])))
    if w < 4 or h < 4:
        return None
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    out = cv2.warpPerspective(img, cv2.getPerspectiveTransform(q, dst), (w, h),
                              borderMode=cv2.BORDER_REPLICATE)
    if h > w * 1.5:
        out = cv2.rotate(out, cv2.ROTATE_90_CLOCKWISE)
    return out


# ------------------------------------------------------------------ 특징
FEATS = ["aspect", "h_norm", "fill", "n_cc", "cc_per_w", "cc_ar_mean", "cc_ar_std",
         "cc_fill_mean", "cc_h_ratio", "cc_h_std", "gap_ratio", "stroke", "edge_dens",
         "sat_mean", "val_std"]


def features(crop, img_h):
    """rec 없이 계산 가능한 싸구려 특징. 한글 음절 블록 vs 숫자열을 가른다."""
    h, w = crop.shape[:2]
    if h < 3 or w < 3:
        return None
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # 높이 32로 정규화해 스케일 영향 제거
    s = 32.0 / h
    gn = cv2.resize(g, (max(4, int(w * s)), 32), interpolation=cv2.INTER_AREA)
    H, W = gn.shape
    _, bw = cv2.threshold(gn, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if bw.mean() > 127:                      # 글자가 어두운 쪽이 되도록 통일
        bw = 255 - bw
    fill = float((bw > 0).mean())

    n, lab, stats, _ = cv2.connectedComponentsWithStats((bw > 0).astype(np.uint8), 8)
    comps = [stats[i] for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 4]
    if comps:
        cw = np.array([c[cv2.CC_STAT_WIDTH] for c in comps], float)
        ch = np.array([c[cv2.CC_STAT_HEIGHT] for c in comps], float)
        ca = np.array([c[cv2.CC_STAT_AREA] for c in comps], float)
        ar = cw / np.maximum(ch, 1)
        ccf = ca / np.maximum(cw * ch, 1)
        f_ncc, f_arm, f_ars = len(comps), float(ar.mean()), float(ar.std())
        f_ccf = float(ccf.mean())
        f_chr = float((ch / H).mean())
        f_chs = float((ch / H).std())
    else:
        f_ncc = f_arm = f_ars = f_ccf = f_chr = f_chs = 0.0

    col = (bw > 0).any(axis=0)
    gaps = int(np.sum(np.diff(col.astype(int)) != 0))

    dist = cv2.distanceTransform((bw > 0).astype(np.uint8), cv2.DIST_L2, 3)
    stroke = float(dist[bw > 0].mean() * 2) if (bw > 0).any() else 0.0

    edges = cv2.Canny(gn, 60, 160)
    hsv = cv2.cvtColor(cv2.resize(crop, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)

    return {
        "aspect": w / max(h, 1),
        "h_norm": h / max(img_h, 1),
        "fill": fill,
        "n_cc": f_ncc,
        "cc_per_w": f_ncc / max(W / 32.0, 0.1),
        "cc_ar_mean": f_arm,
        "cc_ar_std": f_ars,
        "cc_fill_mean": f_ccf,
        "cc_h_ratio": f_chr,
        "cc_h_std": f_chs,
        "gap_ratio": gaps / max(W / 32.0, 0.1),
        "stroke": stroke,
        "edge_dens": float((edges > 0).mean()),
        "sat_mean": float(hsv[:, :, 1].mean()),
        "val_std": float(hsv[:, :, 2].std()),
    }


def has_date(txt):
    return any(not d["in_long_digits"] and not d["day_missing"] for d in find_dates(txt or ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--cache", default="boxfeat.csv")
    ap.add_argument("--out", default="08_boxfilter.md")
    a = ap.parse_args()

    cache = os.path.join(OUT_DIR, a.cache)
    if os.path.exists(cache):
        df = pd.read_csv(cache)
        print(f"캐시 로드: {len(df):,} 박스")
    else:
        meta = pd.read_csv(os.path.join(OUT_DIR, "metadata.csv"))
        sam = meta.sample(min(a.n, len(meta)), random_state=11)
        ocr = build()
        det, rec = ocr.text_detector, ocr.text_recognizer
        rows, t0 = [], time.time()
        for k, f in enumerate(sam["file"], 1):
            img = cv2.imdecode(np.fromfile(os.path.join(IMG_DIR, f), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            hh, ww = img.shape[:2]
            s = min(1.0, 960 / max(hh, ww))
            if s < 1:
                img = cv2.resize(img, (int(ww * s), int(hh * s)), interpolation=cv2.INTER_AREA)
            for rotname, im in (("0", img), ("90", cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))):
                boxes, _ = det(im)
                if boxes is None or not len(boxes):
                    continue
                crops, keep = [], []
                for b in boxes:
                    c = crop_quad(im, b)
                    if c is None:
                        continue
                    ft = features(c, im.shape[0])
                    if ft is None:
                        continue
                    ch, cw = c.shape[:2]
                    crops.append(cv2.resize(c, (max(8, int(cw * 48 / ch)), 48)))
                    keep.append(ft)
                if not crops:
                    continue
                rr = rec(crops)
                rr = rr[0] if isinstance(rr, tuple) else rr
                for ft, (txt, conf) in zip(keep, rr):
                    ft["file"] = f
                    ft["rot"] = rotname
                    ft["text"] = txt
                    ft["conf"] = float(conf)
                    ft["y"] = int(has_date(txt))
                    rows.append(ft)
            if k % 50 == 0:
                print(f"  {k}/{len(sam)}  박스 {len(rows):,}  {time.time()-t0:.0f}s", flush=True)
        df = pd.DataFrame(rows)
        df.to_csv(cache, index=False, encoding="utf-8-sig")
        print(f"수집 완료: {len(df):,} 박스, {time.time()-t0:.0f}s")

    pos = df["y"].sum()
    print(f"양성(날짜 파싱됨) 박스 {pos:,} / 전체 {len(df):,} = {pos/len(df)*100:.2f}%")

    # ---------------- 분류기 ----------------
    from sklearn.model_selection import GroupKFold
    from sklearn.ensemble import HistGradientBoostingClassifier

    X = df[FEATS].fillna(0).values
    y = df["y"].values
    groups = df["file"].values          # 같은 이미지가 학습/평가에 섞이지 않게

    oof = np.zeros(len(df))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                           max_leaf_nodes=31, random_state=0)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    df["score"] = oof

    L = ["# EDA 8 — 박스 사전 필터 (rec 비용 절감)\n",
         f"- 표본: 이미지 {df['file'].nunique():,}장 → 박스 **{len(df):,}개** (0°/90° 합산)",
         f"- 양성(rec 결과에서 날짜가 파싱된 박스): **{pos:,}개 ({pos/len(df)*100:.2f}%)**",
         "- 검증: 이미지 단위 GroupKFold 5-fold (같은 이미지가 학습/평가에 섞이지 않음)\n",
         "> 07단계에서 rec 는 전체 비용의 80.9% 였다. 박스를 미리 걸러내면 그만큼 직접 줄어든다.\n",
         "## 임계값별 트레이드오프\n",
         "| 유지 비율 | 양성 재현율 | rec 호출 감소 | 1회 통과 예상 |",
         "| ---: | ---: | ---: | ---: |"]

    base_rec_ms = 684.0     # 07단계 실측: rec 평균 684 ms/장
    base_other = 845.0 - base_rec_ms
    for keep in (1.0, 0.5, 0.3, 0.2, 0.15, 0.1, 0.05):
        thr = np.quantile(oof, 1 - keep)
        sel = oof >= thr
        recall = df.loc[sel, "y"].sum() / max(pos, 1)
        cost = base_other + base_rec_ms * keep
        L.append(f"| {keep*100:.0f}% | **{recall*100:.1f}%** | "
                 f"{(1-keep)*100:.0f}% | {cost:.0f} ms |")
    L.append("")

    # 재현율 기준으로 역산
    L.append("## 목표 재현율을 지키는 최소 유지 비율\n")
    L.append("| 목표 양성 재현율 | 필요한 유지 비율 | rec 비용 | 1회 통과 |")
    L.append("| ---: | ---: | ---: | ---: |")
    order = np.argsort(-oof)
    ycum = np.cumsum(y[order])
    for target in (1.00, 0.99, 0.98, 0.95, 0.90):
        need = int(np.searchsorted(ycum, target * pos) + 1)
        keep = need / len(df)
        cost = base_other + base_rec_ms * keep
        L.append(f"| {target*100:.0f}% | {keep*100:.1f}% | {base_rec_ms*keep:.0f} ms | {cost:.0f} ms |")
    L.append("")

    # 100% 재현율 지점
    need100 = int(np.searchsorted(ycum, pos) + 1)
    keep100 = need100 / len(df)
    cost100 = base_other + base_rec_ms * keep100
    L.append(f"> **양성 박스를 하나도 놓치지 않으려면 상위 {keep100*100:.1f}% 만 rec 하면 된다.** "
             f"1회 통과 비용 845 ms → **{cost100:.0f} ms ({845/cost100:.1f}배 단축)**")
    L.append("")

    L.append("## 특징 중요도 (permutation)\n")
    from sklearn.inspection import permutation_importance
    m = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(X, y)
    pi = permutation_importance(m, X, y, n_repeats=5, random_state=0,
                                scoring="average_precision")
    imp = sorted(zip(FEATS, pi.importances_mean), key=lambda t: -t[1])
    L.append("| 특징 | 중요도 | 의미 |")
    L.append("| --- | ---: | --- |")
    desc = {
        "cc_ar_mean": "연결요소 가로/세로비 — 한글 음절은 정사각, 숫자는 세로로 김",
        "cc_fill_mean": "연결요소 내부 채움율 — 한글이 훨씬 빽빽함",
        "n_cc": "연결요소 개수",
        "cc_per_w": "폭 대비 연결요소 밀도",
        "aspect": "박스 가로/세로비 — 날짜줄은 가늘고 김",
        "gap_ratio": "글자 사이 공백 전환 횟수",
        "stroke": "획 두께",
        "fill": "전체 잉크 비율",
        "cc_h_std": "연결요소 높이 편차 — 숫자는 높이가 균일",
        "edge_dens": "에지 밀도",
        "sat_mean": "채도 — 잉크젯 날짜는 무채색인 경우가 많음",
    }
    for f, v in imp[:10]:
        L.append(f"| `{f}` | {v:.4f} | {desc.get(f,'')} |")
    L.append("")
    L.append("> ⚠️ 이 필터는 **rec 결과로 만든 자기지도 라벨**로 학습·평가했다. "
             "즉 'rec 가 날짜로 읽어낸 박스'를 재현하는 것이지, "
             "'rec 가 놓친 진짜 날짜'까지 살려내지는 못한다. "
             "recall 을 올리는 수단이 아니라 **비용을 줄이는 수단**으로만 쓸 것.")

    with open(os.path.join(OUT_DIR, a.out), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
