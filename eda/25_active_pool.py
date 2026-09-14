"""
EDA 25단계 — 액티브 러닝: 위원회 불일치로 라벨링 대상 고르기

문제: 라벨링 예산이 병목이다(정답 34장, 95% CI ±16%p). 게다가 그 34장이
      id<700 고해상도 폰 사진에 편향되어, 전체의 33%인 640×640 웹 수집
      클러스터는 **한 장도 검증되지 않았다**.

방법: Query-by-Committee. 같은 이미지를 설정만 바꿔 여러 번 돌리면 결과가 갈린다
      (EDA 12/14: 전처리 합집합 52.8% vs 단일 최고 39.8%). 이 **불일치**가
      곧 "모델이 헷갈려하는 정도" 이므로 정보량 지표로 쓴다.

⚠️ 측정용과 개선용을 반드시 분리한다.
   - 정확도 측정 → **무작위 대표 표본**. 액티브로 뽑으면 어려운 것만 모여 왜곡된다.
   - 규칙 개선 → 액티브 표본.
   같은 집합으로 둘 다 하면 성능을 스스로 속이게 된다.
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
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "eda"))
import pipeline as P
import ocrlib

IMG_DIR = os.path.join(ROOT, "images", "상품사진입니다")
OUT_DIR = os.path.join(ROOT, "eda", "out")

# 위원회 — 설정이 서로 충분히 다르되 전부 현실적인 것들
COMMITTEE = [
    {"name": "s640", "side": 640, "rot": (0, 90)},
    {"name": "s512", "side": 512, "rot": (0, 90)},
    {"name": "s448", "side": 448, "rot": (0, 90)},
    {"name": "s640_r0", "side": 640, "rot": (0,)},
]

_OCR = None


def get_ocr():
    global _OCR
    if _OCR is None:
        _OCR = ocrlib.build_ocr(det_side=640, threads=1)
    return _OCR


def work(path):
    ocr = get_ocr()
    img = P.imread(path)
    out = {"file": os.path.basename(path)}
    if img is None:
        out["err"] = True
        return out
    for m in COMMITTEE:
        pipe = P.DatePipeline(ocr, long_side=m["side"], rotations=m["rot"],
                              adaptive=True, keep=None)
        try:
            final, cands, anc = pipe.predict_image(img)
        except Exception:
            final, cands, anc = None, {}, set()
        out[m["name"]] = final
        out[m["name"] + "_n"] = len(cands)
        if m["name"] == "s640":
            out["anchors"] = sorted(anc)
            out["cands"] = sorted(cands)
    return out


def category(votes):
    """위원회 투표 패턴을 정보량 관점에서 분류."""
    dates = [v for v in votes if v]
    uniq = set(dates)
    if not dates:
        return "C_전원실패"
    if len(uniq) > 1:
        return "A_날짜불일치"         # 가장 정보량이 큼
    if len(dates) < len(votes):
        return "B_검출경계"           # 일부만 찾음
    return "D_전원일치"               # 쉬움


def cluster_of(row):
    if row["w"] == 640 and row["h"] == 640:
        return "sq640"
    if row["mp"] > 5:
        return "hires"
    return "mid"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=900, help="위원회를 돌릴 후보 풀")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-active", type=int, default=60, help="액티브 선발 수")
    ap.add_argument("--n-random", type=int, default=45, help="무작위 측정셋 수")
    ap.add_argument("--out", default="25_active_pool.md")
    a = ap.parse_args()

    meta = pd.read_csv(os.path.join(OUT_DIR, "metadata.csv"))
    meta["cluster"] = meta.apply(cluster_of, axis=1)
    truth = json.load(open(os.path.join(OUT_DIR, "truth_v2.json"), encoding="utf-8"))

    # 이미 라벨링된 것은 풀에서 제외
    pool_df = meta[~meta["file"].isin(truth)]
    # 클러스터 비율을 유지해 표집 (640×640 이 33% 이므로 자연히 많이 들어온다)
    rng = np.random.default_rng(20260912)
    frac = a.pool / len(pool_df)
    pool = (pool_df.groupby("cluster", group_keys=False)
                   .apply(lambda g: g.sample(max(1, int(round(len(g) * frac))),
                                             random_state=1)))
    files = [os.path.join(IMG_DIR, f) for f in pool["file"]]
    print(f"풀 {len(files)}장 × 위원회 {len(COMMITTEE)}명", flush=True)
    print("  클러스터 분포:", dict(Counter(pool["cluster"])), flush=True)

    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    res = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(work, files, chunksize=2), 1):
            res.append(r)
            if i % 100 == 0:
                print(f"  {i}/{len(files)}  {time.time()-t0:.0f}s", flush=True)
    print(f"위원회 실행 완료 {time.time()-t0:.0f}s", flush=True)

    df = pd.DataFrame(res)
    df = df.merge(meta[["file", "cluster", "w", "h", "mp"]], on="file", how="left")
    names = [m["name"] for m in COMMITTEE]
    df["votes"] = df[names].values.tolist()
    df["cat"] = df["votes"].map(category)
    # 불일치도: 최빈 표의 비율이 낮을수록 크다
    def disagree(v):
        c = Counter(x if x else "NONE" for x in v)
        return 1.0 - c.most_common(1)[0][1] / len(v)
    df["disagree"] = df["votes"].map(disagree)
    df.to_csv(os.path.join(OUT_DIR, "active_pool.csv"), index=False, encoding="utf-8-sig")

    # ---------------- 선발 ----------------
    # 측정셋: 순수 무작위 (대표성 유지)
    eval_set = df.sample(min(a.n_random, len(df)), random_state=7)
    rest = df[~df["file"].isin(eval_set["file"])]

    # 액티브셋: 정보량 높은 것 우선 + 클러스터 균형
    #   A(날짜불일치) > B(검출경계) > C(전원실패) 순, D(전원일치)는 제외
    prio = {"A_날짜불일치": 0, "B_검출경계": 1, "C_전원실패": 2, "D_전원일치": 9}
    rest = rest.assign(_p=rest["cat"].map(prio))
    cand = rest[rest["_p"] < 9].sort_values(["_p", "disagree"], ascending=[True, False])

    # 클러스터별로 골고루 뽑는다 (640×640 이 미검증이므로 최소 비율 보장)
    per = max(1, a.n_active // 3)
    picks = []
    for cl in ("sq640", "hires", "mid"):
        picks.append(cand[cand["cluster"] == cl].head(per))
    active = pd.concat(picks).head(a.n_active)
    if len(active) < a.n_active:
        extra = cand[~cand["file"].isin(active["file"])].head(a.n_active - len(active))
        active = pd.concat([active, extra])

    # ---------------- 저장 ----------------
    def dump(frame, path, kind):
        items = []
        for _, r in frame.iterrows():
            items.append({
                "file": r["file"], "kind": kind, "cluster": r["cluster"],
                "size": f"{int(r['w'])}x{int(r['h'])}",
                "cat": r["cat"], "disagree": round(float(r["disagree"]), 2),
                "votes": {n: r[n] for n in names},
                "cands": r.get("cands") if isinstance(r.get("cands"), list) else [],
                "anchors": r.get("anchors") if isinstance(r.get("anchors"), list) else [],
            })
        json.dump(items, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        return items

    ev = dump(eval_set, os.path.join(OUT_DIR, "label_eval_random.json"), "eval")
    ac = dump(active, os.path.join(OUT_DIR, "label_active.json"), "active")

    L = ["# EDA 25 — 액티브 러닝: 라벨링 대상 선발\n",
         f"- 후보 풀 **{len(df)}장** × 위원회 **{len(COMMITTEE)}명** "
         f"(해상도 640/512/448 + 0°단독)",
         f"- 위원회 실행 {time.time()-t0:.0f}초",
         "",
         "> 같은 이미지를 설정만 바꿔 돌리면 결과가 갈린다(EDA 12/14). "
         "그 **불일치**를 모델이 헷갈려하는 정도로 보고 정보량 지표로 쓴다.",
         "",
         "## 위원회 투표 패턴\n",
         "| 유형 | 장수 | 비율 | 뜻 |",
         "| --- | ---: | ---: | --- |"]
    desc = {"A_날짜불일치": "위원마다 **다른 날짜** — 인식·선택 모호. 정보량 최대",
            "B_검출경계": "일부만 날짜를 찾음 — 검출 경계선",
            "C_전원실패": "전원 NONE — 근본적 실패. 원인 파악용",
            "D_전원일치": "전원 같은 날짜 — 쉬움. 라벨링 가치 낮음"}
    for k, v in df["cat"].value_counts().items():
        L.append(f"| {k} | {v:,} | {v/len(df)*100:.1f}% | {desc[k]} |")
    L.append("")

    L.append("## 클러스터 × 투표 패턴\n")
    ct = pd.crosstab(df["cluster"], df["cat"])
    L.append("| 클러스터 | " + " | ".join(ct.columns) + " | 합계 |")
    L.append("| --- | " + " | ".join(["---:"] * len(ct.columns)) + " | ---: |")
    for cl, row in ct.iterrows():
        L.append(f"| {cl} | " + " | ".join(str(v) for v in row) + f" | {row.sum()} |")
    L.append("")
    sq = df[df["cluster"] == "sq640"]
    L.append(f"> 640×640 클러스터 {len(sq)}장 중 전원실패가 "
             f"**{(sq['cat'] == 'C_전원실패').mean()*100:.1f}%** "
             f"(전체 평균 {(df['cat'] == 'C_전원실패').mean()*100:.1f}%)")
    L.append("")

    L.append("## 선발 결과\n")
    L.append("| 집합 | 장수 | 표집 방식 | 용도 |")
    L.append("| --- | ---: | --- | --- |")
    L.append(f"| `label_eval_random.json` | {len(ev)} | **순수 무작위** | "
             "정확도 측정 (대표성 필수) |")
    L.append(f"| `label_active.json` | {len(ac)} | 불일치 상위 + 클러스터 균형 | "
             "규칙 개선·실패 원인 분석 |")
    L.append("")
    L.append("### 액티브셋 구성\n")
    L.append("| 클러스터 | 장수 | | 유형 | 장수 |")
    L.append("| --- | ---: | --- | --- | ---: |")
    cc = Counter(x["cluster"] for x in ac)
    kk = Counter(x["cat"] for x in ac)
    keys = list(cc) + [""] * max(0, len(kk) - len(cc))
    vals = list(kk) + [""] * max(0, len(cc) - len(kk))
    for i in range(max(len(cc), len(kk))):
        a1 = f"{keys[i]} | {cc[keys[i]]}" if i < len(cc) and keys[i] else " | "
        a2 = f"{vals[i]} | {kk[vals[i]]}" if i < len(kk) and vals[i] else " | "
        L.append(f"| {a1} | | {a2} |")
    L.append("")
    L.append("> ⚠️ 두 집합을 섞어 정확도를 재면 안 된다. 액티브셋은 어려운 것만 "
             "모아놓은 것이라 실제 성능보다 나쁘게 나온다.")

    with open(os.path.join(OUT_DIR, a.out), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
