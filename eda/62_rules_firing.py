"""
EDA 62단계 — 이식한 규칙이 애초에 발동은 하는가

61단계에서 두 규칙 모두 부분점수를 0.0%p 바꿨고 예측이 달라진 이미지가 0장이었다.
해석이 두 갈래인데 안전성이 크게 다르다.

  (A) 아예 발동하지 않았다  → 검증된 것이 없다. 채점 때 처음 발동하면 그건
                              **검증 안 된 코드가 본선에서 처음 도는 것**이다.
  (B) 발동했지만 결과를 못 바꿨다 → 무해함이 실제로 확인된 것이다.

둘을 구분해야 켜고 끌지 결정할 수 있다. 카운터를 걸어 실제 발동 횟수를 센다.
"""
import os
import sys
import json
import time
import argparse
import warnings
from collections import Counter

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
    """이미지 1장에서 두 규칙이 몇 번 발동할 조건이 되는지 센다."""
    global _PIPE
    if _PIPE is None:
        pipe = P.DatePipeline(ocrlib.build_ocr(det_side=640, threads=1),
                              long_side=640, rotations=(0, 90),
                              adaptive=True, keep=None)
        pipe.yolo = P.OnnxDateDetector(ONNX, threads=1)
        pipe.yolo_conf = 0.25
        pipe.yolo_mode = "always"
        _PIPE = pipe

    # predict_image 가 모아둔 전체 텍스트를 그대로 받아 규칙 조건을 검사한다.
    # (파이프라인을 고치지 않고 같은 입력에 대해 규칙만 다시 돌린다)
    img = P.imread(os.path.join(IMG_DIR, f))
    if img is None:
        return {"file": f, "bc": None, "time_hits": 0, "y2": 0, "ambig": 0}

    texts = []
    for side in [640] + [s for s in (512, 448, 384)]:
        c, a, m, y, t = _PIPE._scan(img, side)
        texts.append(t)
        if c:
            break
    text = "\n".join(texts)

    bc = P.barcode_country(text)

    # 2자리 연도 매치가 몇 개이고, 그중 뒤에 시각이 붙은 것이 몇 개인가
    y2 = ambig = time_hits = 0
    for kind, rx in P.PATTERNS:
        if kind != "Y2_sep":
            continue
        for mm in rx.finditer(text):
            y2 += 1
            s, e = mm.span()
            if P._TIME_AFTER.match(text, e):
                time_hits += 1
            gd = mm.groupdict()
            try:
                a_, c_ = int(gd["y"]), int(gd["d"])
            except (TypeError, ValueError):
                continue
            ok_kr = 2015 <= 2000 + a_ <= 2035 and 1 <= c_ <= 31
            ok_eu = 2015 <= 2000 + c_ <= 2035 and 1 <= a_ <= 31
            if ok_kr and ok_eu:
                ambig += 1
    return {"file": f, "bc": bc, "time_hits": time_hits, "y2": y2, "ambig": ambig}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    h2 = json.load(open(os.path.join(OUT_DIR, "truth_holdout2.json"),
                        encoding="utf-8"))["labels"]
    files = list(h2)

    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(work, files, chunksize=2))
    wall = time.time() - t0

    n = len(rows)
    bc_found = [r for r in rows if r["bc"]]
    bc_counter = Counter(r["bc"] for r in bc_found)
    with_y2 = [r for r in rows if r["y2"]]
    with_ambig = [r for r in rows if r["ambig"]]
    with_time = [r for r in rows if r["time_hits"]]
    # 바코드가 실제로 판단을 바꿀 수 있는 조건: 힌트가 있고 + 모호한 2자리 연도가 있다
    actionable = [r for r in rows if r["bc"] and r["ambig"]]

    L = ["# EDA 62 — 이식한 규칙이 애초에 발동은 하는가\n",
         "61단계에서 두 규칙 모두 예측을 0장 바꿨다. '발동 자체를 안 한 것'과",
         "'발동했지만 결과를 못 바꾼 것'은 안전성이 다르므로 구분한다.",
         "",
         f"홀드아웃2 {n}장, {wall:.0f}초.",
         "",
         "| 조건 | 해당 이미지 | 비율 |",
         "| --- | ---: | ---: |",
         f"| 2자리 연도 날짜가 검출됨 | {len(with_y2)} | {len(with_y2)/n*100:.1f}% |",
         f"| └ 그중 한국식/유럽식이 **모두 타당**(모호) | {len(with_ambig)} | {len(with_ambig)/n*100:.1f}% |",
         f"| └ 그중 뒤에 '시:분' 이 붙음 | {len(with_time)} | {len(with_time)/n*100:.1f}% |",
         f"| 체크섬 유효한 EAN-13 바코드를 읽음 | {len(bc_found)} | {len(bc_found)/n*100:.1f}% |",
         f"| **바코드 힌트가 실제로 판단을 바꿀 수 있는 경우** | **{len(actionable)}** | "
         f"**{len(actionable)/n*100:.1f}%** |",
         ""]
    if bc_counter:
        L.append("바코드 힌트 내역: " + ", ".join(f"{k} {v}장" for k, v in bc_counter.items()))
        L.append("")

    L.append("## 판정\n")
    if not actionable and not with_time:
        L.append("**두 규칙 모두 이 표본에서 한 번도 발동하지 않았다.** 61단계의 '변화 0장'은")
        L.append("무해함이 확인된 것이 아니라 **아무것도 검증되지 않았다**는 뜻이다.")
        L.append("")
        L.append("채점 때 처음 발동하면 검증 안 된 코드가 본선에서 처음 도는 셈이고,")
        L.append("특히 바코드 규칙은 잘못 걸리면 한국 날짜를 유럽식으로 뒤집어 연·일 두 필드를")
        L.append("한꺼번에 틀린다. 기대 이득이 0 인데 꼬리 위험만 있으므로 **기본값을 끈다.**")
        L.append("")
        L.append("코드와 근거는 남겨 둔다 — 표본이 더 커지면 다시 재볼 수 있다.")
    else:
        L.append(f"규칙이 실제로 발동하는 이미지가 있다 "
                 f"(바코드 {len(actionable)}장, 시각 {len(with_time)}장).")
        L.append("61단계에서 결과가 안 바뀐 것은 **발동했지만 최종 선택을 못 바꾼 것**이므로")
        L.append("무해함이 확인된 셈이다. 켜 두어도 된다.")

    with open(os.path.join(OUT_DIR, "62_rules_firing.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
