"""
EDA 63단계 — 팀원2(박안젤라, RapidOCR) 파이프라인을 우리 정답으로 측정

팀원1(EasyOCR)은 엔진 합치기가 이득 0%p 로 기각됐다(EDA 59). 팀원2 는 조건이 다르다.

  · RapidOCR 은 **onnxruntime 기반이라 torch 를 안 쓴다.** 팀원1 을 기각한 가장 큰
    이유(torch 임포트만으로 2.4배 저하)가 여기엔 없다. 즉 합치는 비용이 훨씬 싸다.
  · 인식 엔진 계열도 다르다. 우리 PP-OCRv4(paddle) vs RapidOCR(PP-OCR 계열 ONNX).
  · 규칙이 우리와 독립적으로 짜였다 — 인접 줄 2~3개 이어붙이기, 넓은 키워드 사전.

그래서 다시 묻는다: **우리가 놓친 이미지를 이쪽이 건지는가.**

⚠️ 별도 venv(.venv-team2)에서 실행한다. rapidocr 는 opencv-python-headless 5.x 를
   끌고 오는데, 우리 제출 환경은 opencv 4.10.0.84 로 고정돼 있다(EDA 57 에서
   실제로 깨뜨린 적이 있다).

   .venv-team2\\Scripts\\python.exe eda\\63_team2_eval.py
"""
import os
import sys
import json
import time
import argparse
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(ROOT, "images", "상품사진입니다")
OUT_DIR = os.path.join(ROOT, "eda", "out")
NB = os.path.join(ROOT, "팀원 기록", "2", "박안젤라_predict.ipynb")


def _imread_korean(path, flags=cv2.IMREAD_COLOR):
    """팀원 코드의 cv2.imread 는 한글 경로에서 None 을 준다. 입출력만 바꿔 끼운다."""
    try:
        return cv2.imdecode(np.fromfile(path, np.uint8), flags)
    except Exception:
        return None


def load_team_functions():
    """팀원 노트북의 정의 셀만 실행해 그대로 쓴다 (재구현하지 않는다)."""
    import io
    nb = json.load(io.open(NB, encoding="utf-8"))
    cells = [("".join(c["source"]) if isinstance(c["source"], list) else c["source"])
             for c in nb["cells"] if c["cell_type"] == "code"]
    g = {"__name__": "team2"}
    for src in cells:
        if "INPUT_DIR" in src and "os.environ.get" in src:
            continue
        if "glob.glob(" in src or "to_csv" in src:
            continue          # 실제 실행 루프·저장 셀은 건너뛴다
        try:
            exec(compile(src, "<team2>", "exec"), g)
        except Exception as e:
            print(f"  (셀 건너뜀: {type(e).__name__}: {e})")
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    cv2.imread = _imread_korean

    print("팀원2 노트북에서 함수 로딩 (RapidOCR 최초 로딩 시 모델 준비)", flush=True)
    g = load_team_functions()
    if "predict_expiration_fields" not in g:
        raise SystemExit("predict_expiration_fields 를 찾지 못했다")

    h2 = json.load(open(os.path.join(OUT_DIR, "truth_holdout2.json"),
                        encoding="utf-8"))["labels"]
    files = list(h2)
    if a.limit:
        files = files[:a.limit]

    pred, t0 = {}, time.time()
    for i, f in enumerate(files, 1):
        try:
            r = g["predict_expiration_fields"](os.path.join(IMG_DIR, f))
            pred[f] = "-".join(str(r.get(k, "NONE")) for k in ("year", "month", "day"))
        except Exception:
            pred[f] = "NONE-NONE-NONE"
        if i % 25 == 0:
            print(f"  {i}/{len(files)}  {time.time()-t0:.0f}s", flush=True)
    wall = time.time() - t0

    json.dump({"sec": wall, "pred": pred},
              open(os.path.join(OUT_DIR, "63_team2_pred.json"), "w", encoding="utf-8"),
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
    print(f"\n팀원2 (RapidOCR) — 부분점수 {ok/tot*100:.1f}%, "
          f"값이 채워진 행 {filled}/{len(files)} ({filled/len(files)*100:.1f}%), "
          f"{wall:.0f}초 ({wall/len(files):.2f}초/장)")
    print("→ eda/out/63_team2_pred.json  (합치기 판정은 64단계)")


if __name__ == "__main__":
    main()
