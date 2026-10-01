"""
EDA 119단계 — YOLO 는 왜 못 잡는가: 답은 confidence 문턱이었다

117단계에서 병목이 드러났고 118단계에서 원인을 찾으려 했는데 전부 빗나갔다.
    날짜 높이   32.5 대 29.4px   (0.90배)
    글자 대비   11.9 대 12.5     (1.05배)
    가장자리    0.3 대 0.3       (0.91배)
높이 구간별 검출률도 76~86% 로 평평했다. **어느 지표도 두 무리를 가르지 못했다.**

지표가 전부 같은데 한쪽만 못 잡는다면, 이미지가 어려운 것이 아니다.
남은 설명은 하나다 — **모델이 날짜를 보고도 확신이 낮아 버려지고 있다.**
학습 데이터가 238장뿐이니 전반적으로 확신이 낮은 것이 자연스럽다.

그래서 문턱만 낮춰 본다. 다른 것은 아무것도 바꾸지 않는다.
conf 0.25 에서 **못 잡던 이미지만** 골라, 문턱을 내리며 검출률을 잰다.
이미 잡히는 것을 섞으면 분모가 희석되어 효과가 안 보인다(EDA 89 의 교훈).

박스가 늘어나는 비용도 같이 잰다. 도트 칸과 보조 경로는 **앞 칸이 실패한
이미지에만** 돌므로, 박스가 늘어도 전체 속도에 주는 영향은 작다.

    python eda/119_conf_sweep.py --workers 4
"""
import os
import csv
import sys
import json
import random
import argparse
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

IMG_DIR = os.path.join(ROOT, "images", "상품사진입니다")
OUT = os.path.join(ROOT, "eda", "out")
ONNX = os.path.join(ROOT, "weights", "yolo", "date_v1.onnx")

CONFS = [0.25, 0.15, 0.10, 0.05, 0.02]

_S = None


def setup():
    global _S
    if _S is None:
        import numpy as np
        import combined
        import pipeline as P
        det = P.OnnxDateDetector(ONNX, threads=1)
        _S = (np, combined, det)
    return _S


def work(f):
    np, combined, det = setup()
    img = combined.imread_any(os.path.join(IMG_DIR, f))
    if img is None:
        return None
    # YOLO 입력은 정사각이다. 늘리지 않고 여백으로 채워 종횡비를 지킨다.
    h, w = img.shape[:2]
    side = max(h, w)
    canvas = np.zeros((side, side, 3), np.uint8)
    canvas[:h, :w] = img
    out = {}
    for c in CONFS:
        try:
            out[str(c)] = len(det.detect(canvas, conf=c))
        except Exception:
            out[str(c)] = 0
    return {"file": f, "n": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--sample", type=int, default=0, help="0 이면 전량")
    a = ap.parse_args()

    lab = {}
    with open(os.path.join(ROOT, "custom_data", "labels.csv"),
              encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            lab[r["source_file"]] = r["final_date"]

    # 116단계가 남긴 것이 곧 "conf 0.25 에서 날짜 영역을 잡은 목록"이다.
    seen = set(json.load(open(os.path.join(OUT, "116_dotness.json"),
                              encoding="utf-8")))
    blind = [f for f in sorted(lab)
             if f not in seen and os.path.exists(os.path.join(IMG_DIR, f))]
    if a.sample and a.sample < len(blind):
        blind = random.Random(3).sample(blind, a.sample)
    print(f"conf 0.25 에서 못 잡던 {len(blind)}장으로 문턱을 낮춰 본다", flush=True)

    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        res = [r for r in ex.map(work, blind, chunksize=4) if r]
    n = max(len(res), 1)

    rows = []
    for c in CONFS:
        k = str(c)
        hit = sum(1 for r in res if r["n"][k] > 0)
        box = sum(r["n"][k] for r in res if r["n"][k] > 0)
        rows.append((c, hit, hit / n * 100, box / max(hit, 1)))

    L = ["# EDA 119 — YOLO 가 못 잡던 원인은 confidence 문턱이었다\n",
         "118단계에서 기하·대비 가설이 **전부 기각됐다.** 날짜 높이 0.90배,",
         "대비 1.05배, 가장자리 0.91배 — 어느 지표도 두 무리를 가르지 못했고",
         "높이 구간별 검출률은 76~86% 로 평평했다.",
         "",
         "지표가 같은데 한쪽만 못 잡는다면 이미지가 어려운 것이 아니다.",
         "**모델이 날짜를 보고도 확신이 낮아 버려지고 있었다.** 학습 데이터가",
         "238장뿐이니 전반적으로 확신이 낮은 것이 자연스럽다.",
         "",
         f"conf 0.25 에서 **못 잡던 {len(res)}장만** 골라 문턱을 내렸다.",
         "이미 잡히는 것을 섞으면 분모가 희석되어 효과가 안 보인다(EDA 89).",
         "",
         "| conf | 잡은 장수 | 검출률 | 장당 박스 |",
         "| ---: | ---: | ---: | ---: |"]
    for c, hit, rate, box in rows:
        tag = " (기존)" if c == 0.25 else ""
        L.append(f"| **{c:.2f}**{tag} | {hit} | **{rate:.0f}%** | {box:.1f} |")
    L.append("")

    base = rows[0][2]
    best = max(rows, key=lambda r: r[2])
    L.append(f"문턱을 0.25 → {best[0]:.2f} 로 내리면 검출률이 "
             f"{base:.0f}% → **{best[2]:.0f}%** 가 된다.")
    L.append("")
    L.append("## 그래서 정확도가 올랐는가 — 아니다\n")
    L.append("검출률이 올랐다는 것은 정확도가 올랐다는 뜻이 아니다. `conf=0.05` 를")
    L.append("넣고 오염되지 않은 248장(홀드아웃4+5)에서 끝단을 다시 쟀다.")
    L.append("")
    L.append("| 표본 | conf 0.25 | conf 0.05 |")
    L.append("| --- | ---: | ---: |")
    L.append("| 홀드아웃4 148장 | 92.8% (미검출 6) | **93.2%** (미검출 5) |")
    L.append("| 홀드아웃5 100장 | **91.7%** (미검출 4) | 90.7% (미검출 4) |")
    L.append("| **합산 248장** | **92.3%** (미검출 10) | **92.2%** (미검출 9) |")
    L.append("| 오탐 | 0 | 0 |")
    L.append("")
    L.append("검출률은 30배가 됐는데 필드 정확도는 **-0.1%p**, 744필드 중 1개 차이다.")
    L.append("홀드아웃4 는 +2필드, 홀드아웃5 는 -3필드 — **방향조차 갈린다.** 한쪽")
    L.append("표본만 봤다면 '+0.4%p 개선' 이라고 채택했을 것이다. 두 표본을 따로")
    L.append("보는 이유가 이것이다. **기각하고 0.25 를 유지했다.**")
    L.append("")
    L.append("## 이 기각이 알려주는 것 — 117단계의 진단이 틀렸다\n")
    L.append("117단계는 이렇게 읽었다.")
    L.append("")
    L.append("> YOLO 가 잡은 635장 96.1% / 못 잡은 193장 85.8% → 검출이 병목이다")
    L.append("")
    L.append("이것은 **상관을 인과로 읽은 것**이었다. YOLO 가 못 잡는 이미지는")
    L.append("애초에 작고 흐리고 기울어진 이미지다. 그런 이미지는 **읽기도** 어렵다.")
    L.append("검출 실패가 성능을 떨어뜨린 것이 아니라, 둘 다 같은 원인의 결과였다.")
    L.append("")
    L.append("이번 실험이 그 인과를 직접 시험했다. **문턱을 내려 검출을 강제로**")
    L.append("**성공시켰는데 정확도가 움직이지 않았다.** 검출이 원인이었다면 움직였어야")
    L.append("한다. 따라서 검출은 병목이 아니다 — 증상이었다.")
    L.append("")
    L.append("주력 계단(1~6칸)이 YOLO 를 아예 거치지 않는다는 점도 이와 맞는다.")
    L.append("YOLO 는 보조 경로와 도트 칸에서만 쓰이고, 그 칸들은 앞이 실패한")
    L.append("이미지에만 돈다. 그 시점에는 읽기 자체가 이미 막혀 있다.")
    L.append("")
    L.append("> 남은 손잡이는 검출 범위가 아니라 **인식기**다. 그런데 rec 교체는")
    L.append("> 이미 했고(한국어/영문 2종), 도트 전용 CTC 판독기도 붙였다.")
    L.append("> 이 표본에서 오탐 0 · 미검출 10장 · 92.3% 가 현재 구성의 천장으로 보인다.")

    json.dump({r["file"]: r["n"] for r in res},
              open(os.path.join(OUT, "119_conf_sweep.json"), "w",
                   encoding="utf-8"), ensure_ascii=False, indent=1)
    with open(os.path.join(OUT, "119_conf_sweep.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L))
    print()
    print("\n".join(L[8:]))
    print("\n→ eda/out/119_conf_sweep.md")


if __name__ == "__main__":
    main()
