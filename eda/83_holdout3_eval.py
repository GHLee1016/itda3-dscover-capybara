"""
EDA 83단계 — 홀드아웃3 최종 판정

이번 라운드(69~82)의 모든 변경을 **한 번도 쓰지 않은 표본**에서 잰다.

왜 필요한가
  홀드아웃2 198장은 이번 라운드에서 오염됐다. 78·79단계에서 오답을 직접 들여다보고
  원인을 찾았고, 81단계에서 규칙 선택에도 참고했다. 거기서 나온 90.1% 는
  그 표본에 맞춘 정도가 섞인 값이라 본선 보고서에 쓸 수 없다.

집합 분리 (80단계에서 나눠 뒀다)
  대표 150장   무작위 표집. **성능 수치는 여기서만 낸다**
  표적  50장   2자리 연도가 뒤집힐 수 있는 이미지. 이번 수정의 효과를 보는 용도.
               난이도가 편향돼 있으므로 헤드라인 수치에 넣지 않는다

비교 구성
  A_예선본     1차 예선 제출 구성 (재시도 없음 + 유럽식 기본)
  B_계단       + det medium 재시도
  C_최종       + det medium 재시도 + 2자리 연도 한국식 기본   ← 현재

A 를 재현하려면 규칙의 기본값을 되돌려야 하므로, team2_rules 의 동작을
런타임에 바꿔 끼운다(원본 파일은 건드리지 않는다).

    python eda/83_holdout3_eval.py --workers 4
"""
import os
import sys
import json
import time
import argparse
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "eda"))

IMG_DIR = os.path.join(ROOT, "images", "상품사진입니다")
OUT_DIR = os.path.join(ROOT, "eda", "out")
ONNX = os.path.join(ROOT, "weights", "yolo", "date_v1.onnx")
RAPID_DIR = os.path.join(ROOT, "weights", "rapidocr")

TRUTH3 = os.path.join(OUT_DIR, "truth_holdout3.json")
SPLIT = os.path.join(OUT_DIR, "80_holdout3_split.json")

_P = None
_K = None


def build(mode):
    import pipeline as P
    import team2_rules
    import combined
    import ocrlib

    rapid, _ = combined.build_rapid_ocr(RAPID_DIR)
    fb = P.DatePipeline(ocrlib.build_ocr(det_side=640, threads=1),
                        long_side=640, rotations=(0, 90), adaptive=True, keep=None)
    if os.path.exists(ONNX):
        fb.yolo = P.OnnxDateDetector(ONNX, threads=1)
        fb.yolo_conf = 0.25
        fb.yolo_mode = "always"

    retry = None
    if mode in ("B_계단", "C_최종"):
        retry, _ = combined.build_rapid_retry(RAPID_DIR)

    # B 는 재시도만 더한 것이므로 연도 규칙은 예선본 그대로 둬야 한다.
    # 여기서 B 에 현재 규칙을 주면 B 와 C 가 같은 구성이 되고,
    # "연도 규칙만의 기여"가 언제나 0 으로 나온다(실제로 그렇게 나왔었다).
    rules = team2_rules
    if mode in ("A_예선본", "B_계단"):
        rules = _legacy_rules(team2_rules)

    return combined.CombinedPipeline(rapid, rules, fb, rapid_retry=retry)


def _legacy_rules(mod):
    """
    1차 예선 당시 동작(2자리 연도가 모호하면 유럽식)을 되살린 얇은 껍데기.

    소스를 고쳐 다시 import 하는 대신, 원문 소스를 문자열 치환해 별도 모듈로
    올린다. 원본 파일은 건드리지 않는다.
    """
    import types
    src = open(os.path.join(ROOT, "src", "team2_rules.py"), encoding="utf-8").read()
    old = "elif ymd_date is not None and dmy_date is not None:\n                  found.append(ymd_date)"
    new = "elif ymd_date is not None and dmy_date is not None:\n                  found.append(dmy_date)"
    if old not in src:
        raise SystemExit("예선본 재현 실패: 치환 지점을 찾지 못했다. 81단계 수정을 확인할 것.")
    m = types.ModuleType("team2_rules_legacy")
    exec(compile(src.replace(old, new), "<legacy>", "exec"), m.__dict__)
    return m


def work(args):
    global _P, _K
    f, mode = args
    if _P is None or _K != mode:
        _P, _K = build(mode), mode
    t0 = time.time()
    fields = _P.predict(os.path.join(IMG_DIR, f))
    return {"file": f, "sec": round(time.time() - t0, 3),
            "pred": "-".join(fields[k] for k in ("year", "month", "day"))}


def s3(v):
    if not v or v == "NONE":
        return ("NONE",) * 3
    q = str(v).split("-")
    return tuple(q) if len(q) == 3 else ("NONE",) * 3


def mcnemar(pa, pb):
    up = sum(1 for k in pa if pb.get(k, 0) > pa[k])
    dn = sum(1 for k in pa if pb.get(k, 0) < pa[k])
    if up + dn == 0:
        return up, dn, 1.0
    from math import comb
    n, k = up + dn, min(up, dn)
    return up, dn, min(1.0, sum(comb(n, i) for i in range(k + 1)) / 2 ** n * 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    if not os.path.exists(TRUTH3):
        print("홀드아웃3 라벨이 없다: %s" % TRUTH3)
        print("custom_data/tool 에서 라벨링을 마치고 truth_labeled.json 을")
        print("이 경로로 옮긴 뒤 다시 실행할 것.")
        return

    lab = json.load(open(TRUTH3, encoding="utf-8"))["labels"]
    split = json.load(open(SPLIT, encoding="utf-8"))
    rep = [f for f in split["representative"] if f in lab]
    tgt = [f for f in split["targeted"] if f in lab]
    files = rep + tgt
    files = [f for f in files if os.path.exists(os.path.join(IMG_DIR, f))]
    print(f"홀드아웃3 라벨 {len(lab)}장 → 대표 {len(rep)} / 표적 {len(tgt)}", flush=True)

    MODES = ["A_예선본", "B_계단", "C_최종"]
    from concurrent.futures import ProcessPoolExecutor
    out = {}
    for mode in MODES:
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            rows = list(ex.map(work, [(f, mode) for f in files], chunksize=2))
        out[mode] = {r["file"]: r for r in rows}
        print(f"  {mode:10s} {time.time()-t0:.0f}초", flush=True)

    json.dump({m: {f: r["pred"] for f, r in d.items()} for m, d in out.items()},
              open(os.path.join(OUT_DIR, "83_holdout3.json"), "w",
                   encoding="utf-8"), ensure_ascii=False, indent=1)

    def score(mode, subset):
        d = out[mode]
        ok = tot = miss = fp = 0
        fy = fm = fd = exact = 0
        per = {}
        for f in subset:
            g, p = s3(lab[f]), s3(d[f]["pred"])
            hit = sum(g[i] == p[i] for i in range(3))
            per[f] = hit
            fy += g[0] == p[0]; fm += g[1] == p[1]; fd += g[2] == p[2]
            ok += hit; tot += 3
            gn, pn = g == ("NONE",) * 3, p == ("NONE",) * 3
            miss += (pn and not gn)
            fp += (gn and not pn)
            exact += g == p
        n = max(len(subset), 1)
        return {"score": ok / max(tot, 1), "y": fy / n, "m": fm / n, "d": fd / n,
                "exact": exact / n, "miss": miss, "fp": fp, "per": per,
                "sec": sum(d[f]["sec"] for f in subset) / n}

    DESC = {"A_예선본": "1차 예선 제출 구성 (재시도 없음 · 유럽식)",
            "B_계단": "+ det medium 재시도 (연도 규칙은 예선본 그대로)",
            "C_최종": "**+ 2자리 연도 한국식 (현재)**"}

    L = ["# EDA 83 — 홀드아웃3 최종 판정\n",
         "이번 라운드(69~82)의 모든 변경을 **한 번도 쓰지 않은 표본**에서 쟀다.",
         "홀드아웃2 는 78·79·81단계에서 오답을 직접 들여다봤으므로 더 이상",
         "개선폭의 근거가 될 수 없다.",
         "",
         f"- 대표 {len(rep)}장 (무작위) · 표적 {len(tgt)}장 (2자리 연도 후보)",
         ""]

    for title, subset, note in (
            ("대표 표본 — 성능 수치는 여기서만", rep,
             "무작위 표집이라 이 수치가 실제 평가셋에 대한 추정이다."),
            ("표적 표본 — 2자리 연도 수정의 효과", tgt,
             "일부러 어려운 것만 모았다. **헤드라인 수치에 쓰지 않는다.**")):
        if not subset:
            continue
        L.append(f"## {title}\n")
        L.append(note)
        L.append("")
        L.append("| 구성 | **필드 정확도** | year | month | day | 완전일치 | 미검출 | 오탐 |")
        L.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        S = {m: score(m, subset) for m in MODES}
        for m in MODES:
            s = S[m]
            L.append(f"| {DESC[m]} | **{s['score']*100:.1f}%** | {s['y']*100:.1f}% | "
                     f"{s['m']*100:.1f}% | {s['d']*100:.1f}% | {s['exact']*100:.1f}% | "
                     f"{s['miss']} | {s['fp']} |")
        L.append("")
        up, dn, pv = mcnemar(S["A_예선본"]["per"], S["C_최종"]["per"])
        d = (S["C_최종"]["score"] - S["A_예선본"]["score"]) * 100
        L.append(f"예선본 → 현재: **{d:+.1f}%p** (개선 {up}장 / 악화 {dn}장, p={pv:.4f})")
        up2, dn2, pv2 = mcnemar(S["B_계단"]["per"], S["C_최종"]["per"])
        d2 = (S["C_최종"]["score"] - S["B_계단"]["score"]) * 100
        L.append(f"그중 2자리 연도 수정만의 기여: {d2:+.1f}%p "
                 f"(개선 {up2} / 악화 {dn2}, p={pv2:.4f})")
        L.append("")

    S = {m: score(m, rep) for m in MODES} if rep else {}
    if S:
        L.append("## 홀드아웃2 와의 비교\n")
        L.append("| 표본 | 예선본 | 현재 | 차이 |")
        L.append("| --- | ---: | ---: | ---: |")
        L.append(f"| 홀드아웃2 198장 (오염됨) | 83.5% | 90.1% | +6.6%p |")
        L.append(f"| **홀드아웃3 대표 {len(rep)}장 (깨끗)** | "
                 f"**{S['A_예선본']['score']*100:.1f}%** | "
                 f"**{S['C_최종']['score']*100:.1f}%** | "
                 f"**{(S['C_최종']['score']-S['A_예선본']['score'])*100:+.1f}%p** |")
        L.append("")
        L.append("두 수치의 차이가 홀드아웃2 에 누적된 낙관 편향의 크기다.")
        L.append("")
        L.append("## 속도\n")
        sec = S["C_최종"]["sec"]
        L.append(f"- 장당 {sec:.2f}초, 500장 환산 {sec*500:.0f}초 "
                 f"(제한 2,400초의 {sec*500/2400*100:.0f}%)")

    with open(os.path.join(OUT_DIR, "83_holdout3_eval.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L))
    print()
    print("\n".join(L[5:26]))
    print("\n→ eda/out/83_holdout3_eval.md")


if __name__ == "__main__":
    main()
