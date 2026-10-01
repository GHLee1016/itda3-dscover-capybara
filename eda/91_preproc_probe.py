"""
EDA 91단계 — 점선·저대비 인쇄를 살릴 전처리 찾기

90단계 결과: 오답 51장 중 규칙 문제는 2장뿐이고 49장이 OCR 문제다.
그 49장은 성격이 둘로 갈린다.

    ㉠ 어느 칸도 날짜를 못 만든 23장
       칸을 하나 더 붙이면 **악화가 구조적으로 0** 이다. 이미 NONE 이므로.
    ㉡ 틀린 답을 내고 멈춘 26장
       파이프라인은 온전한 날짜가 나오면 거기서 끝낸다. 칸을 더 붙여도
       **돌지 않는다.** 이 26장을 건드리려면 구조를 바꿔야 한다.

여기서는 "어떤 전처리가 실제로 그 글자를 살리는가"만 잰다. 붙일지 말지는
㉠㉡ 별 회수량을 보고 정한다.

번들된 가중치는 det small / det medium / rec small 뿐이다. 인식기를 키우는
선택지는 없으므로(자동 내려받기는 0점 사유) 전처리와 해상도로만 겨룬다.

    python eda/91_preproc_probe.py --workers 4
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

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

IMG_DIR = os.path.join(ROOT, "images", "상품사진입니다")
OUT = os.path.join(ROOT, "eda", "out")
RAPID_DIR = os.path.join(ROOT, "weights", "rapidocr")

_S = None


def setup():
    global _S
    if _S is None:
        import cv2
        import numpy as np
        import combined
        import team2_rules
        rapid, _ = combined.build_rapid_ocr(RAPID_DIR)
        retry, _ = combined.build_rapid_retry(RAPID_DIR)
        _S = (cv2, np, combined, team2_rules, rapid, retry)
    return _S


def variants(cv2, np, combined, img):
    """이름 -> (이미지, 엔진키). 엔진키는 'small' 또는 'medium'."""
    s1024 = combined.downscale_max(img, 1024)
    s1600 = combined.downscale_max(img, 1600)
    g = cv2.cvtColor(s1024, cv2.COLOR_BGR2GRAY)

    # 도트 매트릭스: 글자가 점으로 끊겨 있다. 닫기 연산으로 점을 이어 붙인다.
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    close2 = cv2.morphologyEx(g, cv2.MORPH_CLOSE, k2)
    close3 = cv2.morphologyEx(g, cv2.MORPH_CLOSE, k3)

    # 잉크가 옅은 인쇄: 국소 대비를 올린다
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(g)

    # 언샤프 마스킹 — 윤곽을 세운다
    blur = cv2.GaussianBlur(g, (0, 0), 2.0)
    unsharp = cv2.addWeighted(g, 1.7, blur, -0.7, 0)

    def bgr(x):
        return cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)

    return [
        ("현재 small@1024",      s1024,          "small"),
        ("det medium@1024",      s1024,          "medium"),
        ("small@1600",           s1600,          "small"),
        ("medium@1600",          s1600,          "medium"),
        ("닫기2x2 + small",      bgr(close2),    "small"),
        ("닫기3x3 + small",      bgr(close3),    "small"),
        ("닫기2x2 + medium",     bgr(close2),    "medium"),
        ("CLAHE + small",        bgr(clahe),     "small"),
        ("CLAHE + medium",       bgr(clahe),     "medium"),
        ("언샤프 + small",       bgr(unsharp),   "small"),
        ("언샤프 + medium",      bgr(unsharp),   "medium"),
    ]


def work(args):
    f, truth = args
    cv2, np, combined, R, rapid, retry = setup()
    img = combined.imread_any(os.path.join(IMG_DIR, f))
    if img is None:
        return None
    eng = {"small": rapid, "medium": retry}
    out = {}
    for name, arr, key in variants(cv2, np, combined, img):
        e = eng.get(key)
        if e is None:
            continue
        try:
            r = e(arr)
            txt = list(r.txts) if getattr(r, "txts", None) else []
        except Exception:
            txt = []
        got = R.choose_expiration_date(txt) if txt else None
        out[name] = got
    return {"file": f, "truth": truth, "got": out}


def s3(v):
    if not v or v == "NONE":
        return ("NONE",) * 3
    q = str(v).split("-")
    return tuple(q) if len(q) == 3 else ("NONE",) * 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    D = json.load(open(os.path.join(OUT, "88_all_pred.json"), encoding="utf-8"))
    lab, pred = D["label"], D["pred"]
    trace = {r["f"]: r for r in json.load(
        open(os.path.join(OUT, "90_stage_trace.json"), encoding="utf-8"))}

    wrong = [f for f in lab if s3(lab[f]) != s3(pred[f])]
    # 90단계의 분류로 두 무리를 나눈다
    g_none = [f for f in wrong if trace.get(f, {}).get("why") == "어느 칸도 날짜를 못 만듦"]
    g_bad = [f for f in wrong if trace.get(f, {}).get("why") == "그 칸이 정답을 못 읽음"]
    print(f"㉠ 아무 답도 없던 {len(g_none)}장 / ㉡ 틀린 답을 낸 {len(g_bad)}장", flush=True)

    targets = [(f, lab[f]) for f in g_none + g_bad]
    t0 = time.time()
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        res = {r["file"]: r for r in ex.map(work, targets, chunksize=1) if r}
    print(f"  {time.time()-t0:.0f}초", flush=True)

    names = []
    for r in res.values():
        for k in r["got"]:
            if k not in names:
                names.append(k)

    def count(group, name):
        return sum(1 for f in group
                   if f in res and res[f]["got"].get(name) == lab[f])

    L = ["# EDA 91 — 점선·저대비 인쇄를 살릴 전처리\n",
         "90단계에서 오답 51장 중 49장이 OCR 문제로 드러났다. 그 49장을",
         "성격별로 나눠, 어떤 전처리가 실제로 글자를 살리는지 잰다.",
         "",
         f"- ㉠ 어느 칸도 날짜를 못 만든 {len(g_none)}장 — 칸을 붙이면 악화가 0",
         f"- ㉡ 틀린 답을 내고 멈춘 {len(g_bad)}장 — 칸을 붙여도 돌지 않는다",
         "",
         "## 변형별 회수량\n",
         "| 변형 | ㉠ 회수 | ㉡ 회수 | 합계 |",
         "| --- | ---: | ---: | ---: |"]
    rows = []
    for n in names:
        a1, a2 = count(g_none, n), count(g_bad, n)
        rows.append((n, a1, a2))
    rows.sort(key=lambda x: -(x[1] + x[2]))
    for n, a1, a2 in rows:
        L.append(f"| {n} | {a1} | {a2} | **{a1+a2}** |")
    L.append("")
    L.append("> ㉡ 은 지금 구조로는 얻을 수 없다. 첫 성공에서 멈추기 때문이다.")
    L.append("> ㉡ 을 가져오려면 여러 칸을 돌려 고르는 구조로 바꿔야 하고, 그만큼 느려진다.")
    L.append("")

    # 현재 쓰는 두 칸이 못 잡는 것을 새 전처리가 잡는지(=붙일 값어치)
    base = {"현재 small@1024", "det medium@1024"}
    L.append("## 지금 칸들이 못 잡는데 새 전처리가 잡는 것 (㉠ 한정)\n")
    L.append("| 변형 | 새로 잡는 장수 | 파일 |")
    L.append("| --- | ---: | --- |")
    for n, _, _ in rows:
        if n in base:
            continue
        got = [f for f in g_none
               if f in res and res[f]["got"].get(n) == lab[f]
               and not any(res[f]["got"].get(b) == lab[f] for b in base)]
        if got:
            L.append(f"| {n} | {len(got)} | {', '.join('`%s`' % x for x in got[:8])} |")
    L.append("")

    json.dump({f: r["got"] for f, r in res.items()},
              open(os.path.join(OUT, "91_preproc_probe.json"), "w",
                   encoding="utf-8"), ensure_ascii=False, indent=1)
    with open(os.path.join(OUT, "91_preproc_probe.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print()
    print("\n".join(L[7:7 + len(rows) + 4]))
    print("\n→ eda/out/91_preproc_probe.md")


if __name__ == "__main__":
    main()
