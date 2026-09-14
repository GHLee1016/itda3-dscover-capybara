"""
EDA 58단계 — 팀원 파이프라인(classical CV + EasyOCR)을 우리 정답으로 측정

팀원 기록에는 "500장 938초 / 값이 채워진 행 171장(34.2%)" 까지만 있다. 정답이 없어
**채워졌는지**만 셌지 **맞았는지**는 못 쟀다. 우리에겐 라벨 383장이 있으니 대신 잰다.

합치기 판단에 필요한 것은 두 가지다.
  1) 팀원 파이프라인의 실제 정확도 (우리 66.8% 와 비교)
  2) **우리가 놓친 이미지를 팀원 쪽이 건지는가** — 합쳐서 이득이 나는 유일한 경로

⚠️ 반드시 별도 venv(.venv-team)에서 실행한다. easyocr 는 torch 와
   opencv-python-headless 를 끌고 오는데, 둘 다 우리 제출 환경을 망가뜨린다
   (torch: 2.4배 저하 — EDA 52. opencv 혼재: cv2 버전이 바뀜 — requirements.txt 경고).

   .venv-team\\Scripts\\python.exe eda\\58_team_eval.py
"""
import os
import sys
import json
import time
import argparse
import warnings

warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(ROOT, "images", "상품사진입니다")
OUT_DIR = os.path.join(ROOT, "eda", "out")
NB = os.path.join(ROOT, "팀원 기록", "1", "itda3-dscover-capybara", "predict.ipynb")
ZIP_NB = None


def _imread_korean(path, flags=cv2.IMREAD_COLOR):
    """팀원 코드는 cv2.imread 를 쓰는데 한글 경로에서 None 을 돌려준다.
    로직은 그대로 두고 입출력만 바꿔 끼운다."""
    try:
        a = np.fromfile(path, np.uint8)
        return cv2.imdecode(a, flags)
    except Exception:
        return None


def load_team_functions():
    """팀원 노트북의 함수 정의 셀만 실행해 그대로 가져온다 (재구현하지 않는다)."""
    import json as _json
    import io
    nbp = NB
    if not os.path.exists(nbp):
        # zip 을 풀어둔 경로를 쓴다
        alt = os.environ.get("ITDA_TEAM_NB")
        if alt and os.path.exists(alt):
            nbp = alt
        else:
            raise SystemExit(f"팀원 노트북을 찾지 못했다: {nbp}\n"
                             "ITDA_TEAM_NB 환경변수로 경로를 넘겨라.")
    nb = _json.load(io.open(nbp, encoding="utf-8"))
    cells = [("".join(c["source"]) if isinstance(c["source"], list) else c["source"])
             for c in nb["cells"] if c["cell_type"] == "code"]
    g = {"__name__": "team"}
    # 0번은 CONFIG, 마지막 두 셀은 실제 실행 루프라 제외한다.
    # 함수·상수 정의 셀(3,6,8,10)만 올린다.
    for src in cells:
        if "INPUT_DIR" in src and "os.environ.get" in src:
            continue
        if "ThreadPoolExecutor(" in src and "submit(" in src:
            continue
        if "to_csv" in src:
            continue
        try:
            exec(compile(src, "<team>", "exec"), g)
        except Exception as e:
            print(f"  (셀 건너뜀: {type(e).__name__}: {e})")
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canvas", type=int, default=550,
                    help="3단계 canvas. 능력 상한을 보려고 가장 큰 값을 기본으로 둔다")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    cv2.imread = _imread_korean          # 한글 경로 대응

    print("팀원 노트북에서 함수 로딩", flush=True)
    g = load_team_functions()
    for need in ("process_one_image", "extract_best_date", "propose_line_boxes"):
        if need not in g:
            raise SystemExit(f"필요한 함수가 없다: {need}")

    import easyocr
    import torch
    torch.set_num_threads(4)
    print("EasyOCR Reader 로딩 (최초 1회 가중치 다운로드 약 99MB)", flush=True)
    reader = easyocr.Reader(["ko", "en"], gpu=False, verbose=False)

    h2 = json.load(open(os.path.join(OUT_DIR, "truth_holdout2.json"),
                        encoding="utf-8"))["labels"]
    files = list(h2)
    if a.limit:
        files = files[:a.limit]

    pred, t0 = {}, time.time()
    for i, f in enumerate(files, 1):
        try:
            y, mo, d = g["process_one_image"](reader, os.path.join(IMG_DIR, f), a.canvas)
        except Exception:
            y = mo = d = None
        pred[f] = "-".join(str(v) if v else "NONE" for v in (y, mo, d))
        if i % 25 == 0:
            print(f"  {i}/{len(files)}  {time.time()-t0:.0f}s", flush=True)
    wall = time.time() - t0

    json.dump({"canvas": a.canvas, "sec": wall, "pred": pred},
              open(os.path.join(OUT_DIR, "58_team_pred.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    def split3(v):
        if not v or v == "NONE":
            return ("NONE",) * 3
        p = str(v).split("-")
        return tuple(p) if len(p) == 3 else ("NONE",) * 3

    ok = tot = filled = 0
    for f in files:
        gt, pp = split3(h2[f]), split3(pred[f])
        ok += sum(gt[i] == pp[i] for i in range(3))
        tot += 3
        filled += pp != ("NONE",) * 3
    print(f"\n팀원 파이프라인 — 부분점수 {ok/tot*100:.1f}%, "
          f"값이 채워진 행 {filled}/{len(files)} ({filled/len(files)*100:.1f}%), "
          f"{wall:.0f}초 ({wall/len(files):.2f}초/장)")
    print(f"→ eda/out/58_team_pred.json  (우리 결과와의 비교는 59단계에서)")


if __name__ == "__main__":
    main()
