"""
EDA 99단계 — 속도를 내주고 정확도를 살 수 있는가

현재 0.81초/장은 장당 예산 4.8초의 17% 다. 여섯 배 가까이 남는다.
본선 배점은 정확성 20점 · 속도 및 효율 검증 15점이므로, 속도를 조금 내주고
정확도를 얻는 거래가 이득일 수 있다. 다만 거래가 성립하려면 **얻을 것이
실제로 있어야** 한다. 그것부터 잰다.

여기서 내는 수치
  ① 칸별 단독 정확도 — 어느 칸이 얼마나 맞히나
  ② **오라클 상한** — 어느 칸 하나라도 맞히면 정답으로 칠 때의 한계.
     이것이 현재와 거의 같으면 여러 칸을 돌려도 얻을 것이 없다는 뜻이고,
     이 방향을 여기서 접는다.
  ③ 다수결 — 같은 답을 낸 칸이 많은 쪽을 고를 때
  ④ 칸별 소요 시간 — 거래의 가격표

⚠️ 라벨 580장은 이미 오답 검토에 사용해 오염됐다. 여기서는 **어떤 구조가
   유망한지 고르는 데만** 쓰고, 채택 여부의 최종 판정은 홀드아웃4 에서 한 번만 한다.

    python eda/99_speed_for_accuracy.py --workers 4
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
        import combined
        import team2_rules
        small, _ = combined.build_rapid_ocr(RAPID_DIR)
        med, _ = combined.build_rapid_retry(RAPID_DIR)
        _S = (cv2, combined, team2_rules, small, med)
    return _S


def rungs(cv2, combined, img):
    """(이름, 이미지, 엔진키) 목록. 엔진키는 small / medium."""
    s1024 = combined.downscale_max(img, 1024)
    s1600 = combined.downscale_max(img, 1600)
    g = cv2.cvtColor(s1024, cv2.COLOR_BGR2GRAY)

    adap = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 31, 11)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(g)
    blur = cv2.GaussianBlur(g, (0, 0), 2.0)
    unsharp = cv2.addWeighted(g, 1.7, blur, -0.7, 0)

    def bgr(x):
        return cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)

    return [
        ("1 small@1024",    s1024,         "small"),
        ("2 adaptive",      adap,          "small"),
        ("3 medium@1024",   s1024,         "medium"),
        ("4 CLAHE+med",     bgr(clahe),    "medium"),
        ("5 small@1600",    s1600,         "small"),
        ("6 medium@1600",   s1600,         "medium"),
        ("7 unsharp+small", bgr(unsharp),  "small"),
        ("8 CLAHE+small",   bgr(clahe),    "small"),
    ]


def work(f):
    cv2, combined, R, small, med = setup()
    img = combined.imread_any(os.path.join(IMG_DIR, f))
    if img is None:
        return None
    eng = {"small": small, "medium": med}
    got, secs = {}, {}
    for name, arr, key in rungs(cv2, combined, img):
        e = eng.get(key)
        if e is None:
            continue
        t0 = time.time()
        try:
            r = e(arr)
            txt = list(r.txts) if getattr(r, "txts", None) else []
        except Exception:
            txt = []
        d = R.choose_expiration_date(txt) if txt else None
        if d is None and txt:
            p = R.extract_partial_date(txt)
            if p:
                d = "-".join(p[k] for k in ("year", "month", "day"))
        secs[name] = round(time.time() - t0, 3)
        got[name] = d
    return {"file": f, "got": got, "sec": secs}


def s3(v):
    if not v or v == "NONE":
        return ("NONE",) * 3
    q = str(v).split("-")
    return tuple(q) if len(q) == 3 else ("NONE",) * 3


def hits(g, p):
    return sum(g[i] == p[i] for i in range(3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n", type=int, default=0, help="0이면 전량")
    a = ap.parse_args()

    D = json.load(open(os.path.join(OUT, "88_all_pred.json"), encoding="utf-8"))
    lab = D["label"]
    files = [f for f in sorted(lab) if os.path.exists(os.path.join(IMG_DIR, f))]
    if a.n:
        import random
        files = random.Random(5).sample(files, min(a.n, len(files)))
    print(f"{len(files)}장 × 8칸", flush=True)

    t0 = time.time()
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        res = {r["file"]: r for r in ex.map(work, files, chunksize=2) if r}
    print(f"  {time.time()-t0:.0f}초", flush=True)

    json.dump({f: {"got": r["got"], "sec": r["sec"]} for f, r in res.items()},
              open(os.path.join(OUT, "99_rungs.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    names = []
    for r in res.values():
        for k in r["got"]:
            if k not in names:
                names.append(k)
    names.sort()

    n = len(res)
    tot = n * 3

    def acc(pick):
        ok = exact = 0
        for f, r in res.items():
            g = s3(lab[f])
            p = s3(pick(f, r))
            ok += hits(g, p)
            exact += (g == p)
        return ok / tot * 100, exact / n * 100

    L = ["# EDA 99 — 속도를 내주고 정확도를 살 수 있는가\n",
         "현재 0.81초/장은 장당 예산 4.8초의 17% 다. 본선 배점은 정확성 20점 ·",
         "속도 및 효율 검증 15점이므로 거래가 이득일 수 있다. 얻을 것이 있는지 먼저 잰다.",
         "",
         f"- 라벨 {n}장 (오답 검토에 이미 사용한 표본이다. **구조를 고르는 데만** 쓴다)",
         "",
         "## ① 칸별 단독 성능\n",
         "| 칸 | 필드 정확도 | 완전일치 | 장당(병렬) |",
         "| --- | ---: | ---: | ---: |"]
    for nm in names:
        fa, ex = acc(lambda f, r, k=nm: r["got"].get(k))
        sec = sum(r["sec"].get(nm, 0) for r in res.values()) / n
        L.append(f"| {nm} | {fa:.1f}% | {ex:.1f}% | {sec:.2f}초 |")
    L.append("")

    # ② 오라클 상한
    def oracle(f, r):
        g = s3(lab[f])
        best, bh = None, -1
        for v in r["got"].values():
            h = hits(g, s3(v))
            if h > bh:
                best, bh = v, h
        return best
    ofa, oex = acc(oracle)
    L.append("## ② 오라클 상한 — 어느 칸이든 하나만 맞으면\n")
    L.append(f"**필드 정확도 {ofa:.1f}% · 완전일치 {oex:.1f}%**")
    L.append("")
    L.append("여덟 칸을 모두 돌리고 **정답을 아는 사람이 고를 때**의 한계다.")
    L.append("실제로는 라벨 없이 골라야 하므로 이보다 낮아진다.")
    L.append("")

    # ③ 다수결
    def vote(f, r):
        c = Counter(v for v in r["got"].values() if v and v != "NONE-NONE-NONE")
        if not c:
            return None
        top = c.most_common()
        best = top[0][1]
        tied = [v for v, k in top if k == best]
        if len(tied) == 1:
            return tied[0]
        # 동점이면 앞선 칸(더 비용이 저렴한 칸)의 답을 따른다
        for nm in names:
            if r["got"].get(nm) in tied:
                return r["got"][nm]
        return tied[0]
    vfa, vex = acc(vote)

    # ③-b 우선순위: 먼저 온전한 날짜를 낸 칸
    def firstfull(f, r):
        for nm in names:
            v = r["got"].get(nm)
            if v and "NONE" not in v:
                return v
        for nm in names:
            if r["got"].get(nm):
                return r["got"][nm]
        return None
    ffa, fex = acc(firstfull)

    L.append("## ③ 라벨 없이 고르는 규칙\n")
    L.append("| 고르는 방법 | 필드 정확도 | 완전일치 |")
    L.append("| --- | ---: | ---: |")
    b1fa, b1ex = acc(lambda f, r: r["got"].get(names[0]))
    L.append(f"| 1칸만 사용 (현재 주력) | {b1fa:.1f}% | {b1ex:.1f}% |")
    L.append(f"| 먼저 온전한 날짜를 낸 칸 (현재 계단) | {ffa:.1f}% | {fex:.1f}% |")
    L.append(f"| **여덟 칸 다수결** | **{vfa:.1f}%** | **{vex:.1f}%** |")
    L.append(f"| 오라클 상한 | {ofa:.1f}% | {oex:.1f}% |")
    L.append("")

    allsec = sum(sum(r["sec"].values()) for r in res.values()) / n
    L.append("## ④ 가격표\n")
    L.append(f"- 여덟 칸 전부 실행: 장당 **{allsec:.2f}초**(병렬 기준). "
             "단일 프로세스는 이보다 빠르다 — 92단계 참고")
    L.append(f"- 현재 계단: 장당 0.81초(단일 실측), 제한의 17%")
    L.append("")
    L.append("## 판단\n")
    gain = vfa - ffa
    if ofa - ffa < 0.5:
        L.append("**오라클 상한이 현재와 거의 같다. 여러 칸을 돌려도 얻을 것이 없다.**")
        L.append("이 방향은 여기서 접고 다른 손잡이를 찾아야 한다.")
    elif gain < 0.3:
        L.append(f"오라클 상한은 {ofa-ffa:+.1f}%p 남아 있지만, 라벨 없이 고르는 규칙으로는")
        L.append(f"{gain:+.1f}%p 밖에 못 가져온다. **고르는 방법이 병목이다.**")
    else:
        L.append(f"다수결이 {gain:+.1f}%p 를 가져온다. 비용과 견줘 판단할 값어치가 있다.")

    with open(os.path.join(OUT, "99_speed_for_accuracy.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L))
    print()
    print("\n".join(L[6:]))
    print("\n→ eda/out/99_speed_for_accuracy.md")


if __name__ == "__main__":
    main()
