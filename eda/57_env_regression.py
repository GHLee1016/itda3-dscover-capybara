"""
EDA 57단계 — 환경 회귀 검사

팀원 파이프라인(easyocr)을 설치했다가 opencv 계열이 섞였다.
  opencv-python 4.10.0.84 + opencv-contrib-python 4.10.0.84 (우리 고정)
  + opencv-python-headless 4.11.0.86 (easyocr 가 끌고 옴)
→ cv2 가 4.11.0 으로 바뀌었고, 되돌리는 과정에서 numpy 도 2.2.6 으로 올라갔다.

requirements.txt 가 경고한 바로 그 조합이다. 지금 환경이 측정 당시와 같은지
**숫자로** 확인해야 한다. 라이브러리 버전이 같아 보여도 결과가 같다는 보장은 없다.

홀드아웃2 앞 60장을 돌려 56단계 기록과 대조한다.
"""
import os
import sys
import json
import time
import argparse
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "eda"))
import pipeline as P
import ocrlib

IMG_DIR = os.path.join(ROOT, "images", "상품사진입니다")
OUT_DIR = os.path.join(ROOT, "eda", "out")
ONNX = os.path.join(ROOT, "weights", "yolo", "date_v1.onnx")

_PIPE = None


def work(f):
    global _PIPE
    if _PIPE is None:
        pipe = P.DatePipeline(ocrlib.build_ocr(det_side=640, threads=1),
                              long_side=640, rotations=(0, 90),
                              adaptive=True, keep=None)
        pipe.yolo = P.OnnxDateDetector(ONNX, threads=1)
        pipe.yolo_conf = 0.25
        pipe.yolo_mode = "always"
        _PIPE = pipe
    try:
        fin, _, _ = _PIPE.predict_path(os.path.join(IMG_DIR, f))
    except Exception:
        fin = None
    r = P.to_row(os.path.splitext(f)[0], fin)
    r["file"] = f
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    import cv2
    import numpy as np
    print(f"cv2 {cv2.__version__} / numpy {np.__version__}", flush=True)

    h2 = json.load(open(os.path.join(OUT_DIR, "truth_holdout2.json"),
                        encoding="utf-8"))["labels"]
    files = list(h2)[:a.n]

    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(work, files, chunksize=2))
    wall = time.time() - t0

    def split3(v):
        if v is None or v == "NONE":
            return ("NONE",) * 3
        p = str(v).split("-")
        return tuple(p) if len(p) == 3 else ("NONE",) * 3

    ok = tot = 0
    for r in rows:
        g, p = split3(h2[r["file"]]), (r["year"], r["month"], r["day"])
        ok += sum(g[i] == p[i] for i in range(3))
        tot += 3
    score = ok / tot

    # 56단계 기준값과 비교할 수 있도록 예측을 저장해 둔다
    snap = os.path.join(OUT_DIR, "57_env_snapshot.json")
    cur = {r["file"]: "-".join((r["year"], r["month"], r["day"])) for r in rows}
    prev = None
    if os.path.exists(snap):
        prev = json.load(open(snap, encoding="utf-8"))
    json.dump({"cv2": cv2.__version__, "numpy": np.__version__,
               "n": len(files), "score": score, "pred": cur},
              open(snap, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print(f"\n{len(files)}장 부분점수 {score*100:.1f}%  ({wall:.0f}s)")
    if prev:
        diff = [f for f in cur if prev["pred"].get(f) != cur[f]]
        print(f"이전 스냅샷({prev['cv2']}/{prev['numpy']}, {prev['score']*100:.1f}%) 대비 "
              f"예측이 달라진 이미지 {len(diff)}장")
        for f in diff[:10]:
            print(f"  {f}: {prev['pred'].get(f)} → {cur[f]}")
    else:
        print("이전 스냅샷이 없다. 이번 결과를 기준으로 저장했다.")
        print("56단계 기록: 홀드아웃2 198장 전체에서 66.8%")


if __name__ == "__main__":
    main()
