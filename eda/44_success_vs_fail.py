"""
EDA 44단계 — 성공 vs 실패 대조 분석

실패만 들여다보면 원인을 특정할 수 없다. 성공한 것에도 같은 특징이 있으면
그건 원인이 아니기 때문이다. 홀드아웃 200장 전부에서 측정 가능한 지표를 뽑아
**두 집단을 대조**한다.

측정 지표
  이미지 전역   해상도, 밝기, 대비, 선명도(Laplacian)
  검출 단계     박스 수, 박스 높이(글자 크기 대용), 박스 면적 비율
  인식 단계     rec 신뢰도 분포, 숫자를 포함한 박스 수
  날짜 영역     정답 날짜를 담은 박스의 높이·신뢰도 (성공 케이스만)

'글자가 몇 픽셀인가' 가 가장 직접적인 후보다. 640px 로 줄인 뒤 글자 높이가
인식 한계(대략 10px) 아래로 내려가면 무엇을 해도 못 읽는다.
"""
import os
import sys
import json
import time
import argparse
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "eda"))
import pipeline as P
import ocrlib

IMG_DIR = os.path.join(ROOT, "images", "상품사진입니다")
OUT_DIR = os.path.join(ROOT, "eda", "out")

_OCR = None


def get_ocr():
    global _OCR
    if _OCR is None:
        _OCR = ocrlib.build_ocr(det_side=640, threads=1)
    return _OCR


def box_h(b):
    q = np.asarray(b, dtype=float).reshape(4, 2)
    return float(min(np.linalg.norm(q[0] - q[3]), np.linalg.norm(q[1] - q[2])))


def work(args):
    f, truth = args
    ocr = get_ocr()
    img = P.imread(os.path.join(IMG_DIR, f))
    r = {"file": f, "truth": truth}
    if img is None:
        return r
    H, W = img.shape[:2]
    r["mp"] = H * W / 1e6

    small = P.downscale(img, 640)
    g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    r["brightness"] = float(g.mean())
    r["contrast"] = float(g.std())
    r["sharpness"] = float(cv2.Laplacian(g, cv2.CV_64F).var())

    # 검출 + 인식 (0° 만 — 지표 비교가 목적)
    res = ocr.ocr(small, cls=False)
    lines = res[0] if res and res[0] else []
    r["n_box"] = len(lines)
    if lines:
        hs = [box_h(ln[0]) for ln in lines]
        cf = [float(ln[1][1]) for ln in lines]
        r["box_h_med"] = float(np.median(hs))
        r["box_h_max"] = float(np.max(hs))
        r["conf_med"] = float(np.median(cf))
        r["conf_max"] = float(np.max(cf))
        r["n_digit_box"] = sum(1 for ln in lines
                               if sum(c.isdigit() for c in ln[1][0]) >= 4)
        # 정답 날짜를 담은 박스 찾기
        gy, gm, gd = (truth.split("-") + ["NONE"] * 3)[:3] if truth != "NONE" \
            else ("NONE",) * 3
        pats = []
        if gy != "NONE":
            pats += [gy + gm + gd, gy[2:] + gm + gd, gd + gm + gy[2:]]
        if gm != "NONE" and gd != "NONE":
            pats.append(gm + gd)
        hit_h = hit_c = None
        for ln in lines:
            dd = "".join(c for c in ln[1][0] if c.isdigit())
            if any(p and p in dd for p in pats):
                hit_h, hit_c = box_h(ln[0]), float(ln[1][1])
                break
        r["date_box_h"] = hit_h
        r["date_box_conf"] = hit_c
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    lab = json.load(open(os.path.join(OUT_DIR, "truth_holdout2.json"),
                         encoding="utf-8"))["labels"]
    err = pd.read_csv(os.path.join(OUT_DIR, "holdout2_errors.csv"))
    err_set = set(err["file"])
    kind_of = dict(zip(err["file"], err["kind"]))

    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rs = list(ex.map(work, [(f, v) for f, v in lab.items()], chunksize=2))
    print(f"측정 {time.time()-t0:.0f}s", flush=True)

    df = pd.DataFrame(rs)
    df["ok"] = ~df["file"].isin(err_set)
    df["kind"] = df["file"].map(lambda f: kind_of.get(f, "정답"))
    df.to_csv(os.path.join(OUT_DIR, "success_vs_fail.csv"), index=False,
              encoding="utf-8-sig")

    ok, ng = df[df["ok"]], df[~df["ok"]]
    unread = df[df["kind"] == "미검출-못읽음"]

    def row(col, label, fmt="{:.2f}"):
        a_, b_, c_ = ok[col].dropna(), ng[col].dropna(), unread[col].dropna()
        if not len(a_) or not len(b_):
            return None
        return (f"| {label} | {fmt.format(a_.median())} | {fmt.format(b_.median())} | "
                f"{fmt.format(c_.median()) if len(c_) else '-'} |")

    L = ["# EDA 44 — 성공 vs 실패 대조 분석\n",
         f"- 홀드아웃 200장 (성공 {len(ok)} / 실패 {len(ng)}, 그중 '못 읽음' {len(unread)})",
         "- 값은 **중앙값**. 640px 로 줄인 뒤 측정 (실제 파이프라인과 동일 조건)",
         "",
         "## 지표 비교\n",
         "| 지표 | 성공 | 실패 | 못 읽음 |",
         "| --- | ---: | ---: | ---: |"]
    for col, lab_, fmt in [
        ("mp", "원본 해상도 (MP)", "{:.2f}"),
        ("brightness", "밝기 (0-255)", "{:.0f}"),
        ("contrast", "대비 (std)", "{:.1f}"),
        ("sharpness", "선명도 (Laplacian)", "{:.0f}"),
        ("n_box", "검출 박스 수", "{:.0f}"),
        ("box_h_med", "박스 높이 중앙값 (px)", "{:.1f}"),
        ("box_h_max", "박스 높이 최대 (px)", "{:.1f}"),
        ("n_digit_box", "숫자 4개+ 박스 수", "{:.0f}"),
        ("conf_med", "rec 신뢰도 중앙값", "{:.3f}"),
        ("conf_max", "rec 신뢰도 최대", "{:.3f}"),
    ]:
        r = row(col, lab_, fmt)
        if r:
            L.append(r)
    L.append("")

    # 날짜 박스가 잡힌 비율
    for name, sub in [("성공", ok), ("실패", ng), ("못 읽음", unread)]:
        if len(sub):
            got = sub["date_box_h"].notna().sum()
            L.append(f"- **{name}**: 정답 날짜를 담은 박스를 찾은 비율 "
                     f"{got}/{len(sub)} ({got/len(sub)*100:.0f}%)")
    L.append("")
    dh = ok["date_box_h"].dropna()
    if len(dh):
        L.append(f"- 성공 케이스에서 **날짜 박스 높이 중앙값 {dh.median():.1f}px**, "
                 f"하위 10% {dh.quantile(.1):.1f}px, 최소 {dh.min():.1f}px")
    L.append("")

    # 글자 크기 구간별 성공률
    L.append("## 박스 높이(글자 크기)별 성공률\n")
    L.append("640px 로 줄인 뒤 검출 박스의 **최대 높이**로 구간을 나눴다.")
    L.append("날짜는 보통 작게 인쇄되므로 이 값이 '이 사진의 글자가 얼마나 큰가'의 대용이다.\n")
    L.append("| 박스 높이 최대 | 장수 | 성공률 |")
    L.append("| --- | ---: | ---: |")
    bins = [0, 15, 20, 25, 35, 1000]
    labels = ["<15px", "15-20px", "20-25px", "25-35px", ">35px"]
    df["hbin"] = pd.cut(df["box_h_max"], bins, labels=labels)
    for b, g in df.groupby("hbin", observed=True):
        L.append(f"| {b} | {len(g)} | {g['ok'].mean()*100:.0f}% |")
    L.append("")

    L.append("## rec 신뢰도별 성공률\n")
    L.append("| 최대 신뢰도 | 장수 | 성공률 |")
    L.append("| --- | ---: | ---: |")
    df["cbin"] = pd.cut(df["conf_max"], [0, .90, .95, .98, 1.01],
                        labels=["<0.90", "0.90-0.95", "0.95-0.98", ">0.98"])
    for b, g in df.groupby("cbin", observed=True):
        L.append(f"| {b} | {len(g)} | {g['ok'].mean()*100:.0f}% |")

    with open(os.path.join(OUT_DIR, "44_success_vs_fail.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
