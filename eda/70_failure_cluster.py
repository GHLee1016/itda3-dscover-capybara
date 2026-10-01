"""
EDA 70단계 — 실패 이미지 군집화와 라벨링 대상 선발 (능동학습 2단계)

69단계가 모은 실패 이미지 중 **어느 것을 라벨링할지** 고른다.

왜 무작위로 뽑지 않는가 (문서 §9.1, Gupte 외 arXiv 2401.14555):
    파운데이션 모델 임베딩에서 k-means 중심에 가까운 샘플을 초기 라벨로 쓰면
    무작위 초기화보다 첫 라운드 정확도가 22~26%p 높았다. 저예산일수록 효과가 크다.
    우리가 라벨링할 수 있는 양(150장 내외)은 명백한 저예산 구간이다.

왜 불확실성을 쓰지 않는가:
    우리는 이미 위원회 불일치 기반 능동학습을 시도해 실패했다(EDA 29).
    불일치가 큰 집합의 오답률이 0%, 전원 일치 집합이 50% 로 신호가 거꾸로였다.
    문서 §10.1 한계 1번이 이유를 설명한다 — "안정적으로 틀리는 오류는 잡지 못한다".
    우리 실패는 검출기가 날짜 영역을 아예 안 보는 경우라 흔들리지 않는다.
    그래서 불확실성 대신 **실패 사실 자체**를 신호로 쓰고, 그 안에서 다양성을 확보한다.

⚠️ 정보 누수 차단 (§3.5): 선발에 정답을 쓰지 않는다. 임베딩과 실패 여부만 쓴다.
⚠️ 재현성 (§3.4): 시드를 고정하고 보고한다.

    .venv-team\\Scripts\\python.exe eda\\70_failure_cluster.py --budget 150
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

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(ROOT, "images", "상품사진입니다")
OUT_DIR = os.path.join(ROOT, "eda", "out")
SEED = 42


def imread_rgb(path):
    import cv2
    a = np.fromfile(path, np.uint8)
    img = cv2.imdecode(a, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def embed_all(files, model_name, batch=16, side=224):
    """DINOv2 임베딩. CPU 로 돈다. side 는 patch14 배수여야 한다(224 = 16×14)."""
    import torch
    import timm
    from PIL import Image

    torch.set_num_threads(os.cpu_count() or 4)
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model.eval()
    cfg = timm.data.resolve_data_config({}, model=model)
    cfg["input_size"] = (3, side, side)
    tf = timm.data.create_transform(**cfg, is_training=False)

    out, t0 = [], time.time()
    for i in range(0, len(files), batch):
        chunk = files[i:i + batch]
        ims = []
        for f in chunk:
            a = imread_rgb(os.path.join(IMG_DIR, f))
            if a is None:
                a = np.zeros((side, side, 3), np.uint8)
            ims.append(tf(Image.fromarray(a)))
        with torch.no_grad():
            v = model(torch.stack(ims))
        out.append(v.numpy())
        if (i // batch) % 5 == 0:
            done = min(i + batch, len(files))
            el = time.time() - t0
            print(f"    임베딩 {done}/{len(files)}  {el:.0f}s", flush=True)
    E = np.concatenate(out, 0).astype(np.float64)
    E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)   # L2 정규화
    return E


def coverage(sel_idx, labels, k):
    """선택 집합이 몇 개 군집을 덮는가 (문서 §4.4 선택 집합 자체의 특성화)."""
    return len(set(labels[i] for i in sel_idx)) / k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=150, help="라벨링할 장수")
    ap.add_argument("--model", default="vit_base_patch14_dinov2.lvd142m")
    ap.add_argument("--side", type=int, default=224)
    a = ap.parse_args()

    rows = json.load(open(os.path.join(OUT_DIR, "69_failures.json"), encoding="utf-8"))
    fails = [r for r in rows if r["stage"] == "none" and r["cause"] != "load_fail"]
    files = [r["file"] for r in fails]
    cause = {r["file"]: r["cause"] for r in fails}
    print(f"실패 {len(files)}장 / 전체 {len(rows)}장", flush=True)
    if not files:
        print("실패가 없다. 69단계를 먼저 돌릴 것.")
        return

    B = min(a.budget, len(files))
    print(f"임베딩 추출 ({a.model}, {a.side}px)", flush=True)
    E = embed_all(files, a.model, side=a.side)

    from sklearn.cluster import KMeans
    print(f"k-means k={B}, seed={SEED}", flush=True)
    km = KMeans(n_clusters=B, random_state=SEED, n_init=10).fit(E)
    lab = km.labels_

    # 각 군집에서 중심에 가장 가까운 1장 (§9.1 중심 기반 선택)
    sel = []
    for c in range(B):
        idx = np.where(lab == c)[0]
        if not len(idx):
            continue
        d = np.linalg.norm(E[idx] - km.cluster_centers_[c], axis=1)
        sel.append(int(idx[int(np.argmin(d))]))
    sel = sorted(set(sel))

    # 무작위 선택과 다양성 비교 (라벨 없이도 잴 수 있는 지표)
    rng = np.random.default_rng(SEED)
    rnd = sorted(rng.choice(len(files), size=len(sel), replace=False).tolist())
    cov_sel = coverage(sel, lab, B)
    cov_rnd = coverage(rnd, lab, B)

    # 군집별 성격
    csize = Counter(lab.tolist())
    big = csize.most_common(8)

    # 산출물
    work = os.path.join(ROOT, "custom_data", "al_round1_worklist.csv")
    os.makedirs(os.path.dirname(work), exist_ok=True)
    import csv as _csv
    with open(work, "w", encoding="utf-8-sig", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["order", "file", "cluster", "cluster_size", "cause"])
        for o, i in enumerate(sel, 1):
            w.writerow([o, files[i], int(lab[i]), csize[lab[i]], cause[files[i]]])

    np.save(os.path.join(OUT_DIR, "70_fail_embed.npy"), E.astype(np.float32))
    json.dump({"files": files, "labels": lab.tolist(), "selected": sel,
               "random_baseline": rnd, "seed": SEED, "model": a.model},
              open(os.path.join(OUT_DIR, "70_cluster.json"), "w", encoding="utf-8"))

    cause_all = Counter(cause.values())
    cause_sel = Counter(cause[files[i]] for i in sel)
    KO = {"no_text": "글자 못 찾음", "text_no_digits": "숫자 줄 없음",
          "digits_no_date": "숫자는 있으나 날짜 아님"}

    L = ["# EDA 70 — 실패 이미지 군집화와 라벨링 대상 선발\n",
         f"- 실패 모집단 **{len(files)}장** (69단계 산출, 홀드아웃2 제외)",
         f"- 임베딩 `{a.model}` {a.side}px, L2 정규화",
         f"- k-means k={B}, seed={SEED} → 각 군집 중심에서 1장씩 **{len(sel)}장** 선발",
         "",
         "선발 방식의 근거는 문서 §9.1 이다. 파운데이션 모델 임베딩에서 중심 기반",
         "선택이 무작위보다 저예산 구간에서 크게 앞섰다는 결과를 따랐다.",
         "불확실성 기반 선발은 우리 데이터에서 이미 실패했으므로(EDA 29) 쓰지 않았다.",
         "",
         "## 선택 집합의 다양성 (라벨 없이 측정)\n",
         "| 선택 방식 | 덮은 군집 수 | 군집 커버리지 |",
         "| --- | ---: | ---: |",
         f"| **중심 기반 (채택)** | {int(cov_sel*B)} / {B} | **{cov_sel*100:.1f}%** |",
         f"| 무작위 (같은 장수) | {int(cov_rnd*B)} / {B} | {cov_rnd*100:.1f}% |",
         "",
         "커버리지가 높을수록 서로 다른 실패 유형을 고르게 담았다는 뜻이다.",
         "라벨링 전에 선택 자체의 품질을 볼 수 있는 유일한 지표다(문서 §4.4).",
         "",
         "## 실패 원인 구성\n",
         "| 원인 | 모집단 | 선발 집합 |",
         "| --- | ---: | ---: |"]
    for c, v in cause_all.most_common():
        L.append(f"| {KO.get(c,c)} | {v} ({v/len(files)*100:.1f}%) | "
                 f"{cause_sel.get(c,0)} ({cause_sel.get(c,0)/max(len(sel),1)*100:.1f}%) |")
    L.append("")
    L.append("## 큰 군집 (같은 실패가 반복되는 유형)\n")
    L.append("| 군집 | 장수 | 대표 파일 | 원인 |")
    L.append("| ---: | ---: | --- | --- |")
    for c, sz in big:
        idx = np.where(lab == c)[0]
        d = np.linalg.norm(E[idx] - km.cluster_centers_[c], axis=1)
        rep = files[int(idx[int(np.argmin(d))])]
        L.append(f"| {c} | {sz} | `{rep}` | {KO.get(cause[rep], cause[rep])} |")
    L.append("")
    L.append("## 산출물\n")
    L.append(f"- `custom_data/al_round1_worklist.csv` — 라벨링 대상 {len(sel)}장")
    L.append("- `eda/out/70_cluster.json` — 군집 배정과 선발 인덱스 (재현용)")
    L.append("- `eda/out/70_fail_embed.npy` — 임베딩 (재사용)")
    L.append("")
    L.append("## 다음 단계\n")
    L.append("1. 선발 이미지에 **날짜 영역 박스**를 라벨링한다 (도구 필요)")
    L.append("2. 기존 238박스 + 신규 박스로 YOLO 재학습")
    L.append("3. **신규 홀드아웃**으로 측정 — 홀드아웃2 로 재면 개선폭을 믿을 수 없다")

    with open(os.path.join(OUT_DIR, "70_failure_cluster.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L))
    print()
    print("\n".join(L[:22]))
    print(f"\n→ {work}")
    print("→ eda/out/70_failure_cluster.md")


if __name__ == "__main__":
    main()
