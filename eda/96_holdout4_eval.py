"""
EDA 96단계 — 홀드아웃4 최종 판정

이번 라운드(88~95)의 모든 변경을 **한 번도 쓰지 않은 표본**에서 잰다.

왜 또 필요한가
  홀드아웃3 까지 포함한 580장은 88~91단계에서 오답을 한 장씩 들여다보며
  고쳤다. 거기서 나온 94.1% 는 그 표본에 맞춘 정도가 섞인 값이다.
  홀드아웃4 150장은 94단계에서 나머지 2,769장에서 새로 뽑았다.

비교 구성
  A_예선본   1차 예선 제출 구성 — 재시도 없음 · 유럽식 기본 · 이번에 더한 패턴 없음
  D_현재     계단 4칸 + 이번 수정 전부

  ⚠️ A 는 완전한 재현이 아니다. 당시에는 부분 결과가 나오면 계단이 거기서
     멈췄는데(그게 이번에 고친 버그다), 여기서는 재시도를 아예 빼는 것으로
     근사했다. 그래서 A 가 실제 예선본보다 아주 조금 유리할 수 있다.
     즉 아래 개선폭은 **보수적으로 낮게** 잡힌 값이다.

    python eda/96_holdout4_eval.py --workers 4
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

IMG_DIR = os.path.join(ROOT, "images", "상품사진입니다")
OUT = os.path.join(ROOT, "eda", "out")
ONNX = os.path.join(ROOT, "weights", "yolo", "date_v1.onnx")
RAPID_DIR = os.path.join(ROOT, "weights", "rapidocr")

_P = None
_K = None


def _legacy_rules():
    """
    예선 당시의 규칙으로 되돌린 껍데기. 원본 파일은 건드리지 않고,
    소스를 문자열로 읽어 이번에 더한 것들을 되돌린 뒤 별도 모듈로 올린다.
    """
    import types
    src = open(os.path.join(ROOT, "src", "team2_rules.py"), encoding="utf-8").read()
    n = 0

    # ① 2자리 연도 기본값을 유럽식으로 되돌린다 (81단계에서 바꾼 것)
    a = ("elif ymd_date is not None and dmy_date is not None:\n"
         "                found.append(ymd_date)")
    b = ("elif ymd_date is not None and dmy_date is not None:\n"
         "                found.append(dmy_date)")
    if a in src:
        src = src.replace(a, b)
        n += 1

    # ② 이번에 더한 패턴들을 무력화한다. 지우는 대신 결과를 버리게 한다.
    for var in ("pattern5b", "pattern4b", "pattern_yearmonth_day"):
        marker = f"        for {var[0].upper() if False else ''}"
        old = f"{var} = re.findall("
        if old in src:
            # 리스트를 빈 것으로 만들어 뒤의 for 루프가 돌지 않게 한다
            src = src.replace(f"        for d, mon, y in {var}:",
                              f"        for d, mon, y in []:")
            src = src.replace(f"        for mon, d, yy in {var}:",
                              f"        for mon, d, yy in []:")
            src = src.replace(f"        for year, month, day in {var}:",
                              f"        for year, month, day in []:")
            n += 1

    # ③ pattern1 의 시각 차단을 되돌린다
    if "r'(?!\\s*:\\s*\\d)'," in src:
        src = src.replace("            r'(?!\\d)'\n            r'(?!\\s*:\\s*\\d)',",
                          "            r'(?!\\d)',")
        n += 1

    if n < 4:
        raise SystemExit(f"예선본 재현 실패: {n}/4 만 되돌렸다. 소스 변경을 확인할 것.")

    m = types.ModuleType("team2_rules_legacy")
    exec(compile(src, "<legacy>", "exec"), m.__dict__)
    return m


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

    if mode == "A_예선본":
        # 1600px 칸은 기각돼 최종 combined.py 에서 제거됐다. 예선 재현은
        # 재시도 칸 없이 주력 엔진 + 당시 규칙만으로 구성하면 된다.
        return combined.CombinedPipeline(rapid, _legacy_rules(), fb)
    retry, _ = combined.build_rapid_retry(RAPID_DIR)
    ko, _ = combined.build_rapid_rec(RAPID_DIR, combined.REC_KOREAN)
    en, _ = combined.build_rapid_rec(RAPID_DIR, combined.REC_EN)
    return combined.CombinedPipeline(rapid, team2_rules, fb, rapid_retry=retry,
                                     rec_korean=ko, rec_en=en)


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


def wilson(k, n):
    """이항 비율의 95% 신뢰구간. 148장짜리 표본이라 오차범위를 같이 적는다."""
    if n == 0:
        return 0.0, 0.0
    from math import sqrt
    z = 1.96
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    lab = json.load(open(os.path.join(OUT, "truth_holdout4.json"),
                         encoding="utf-8"))["labels"]
    files = [f for f in sorted(lab) if os.path.exists(os.path.join(IMG_DIR, f))]
    print(f"홀드아웃4 {len(files)}장 × 2구성", flush=True)

    MODES = ["A_예선본", "D_현재"]
    from concurrent.futures import ProcessPoolExecutor
    out = {}
    for mode in MODES:
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            rows = list(ex.map(work, [(f, mode) for f in files], chunksize=2))
        out[mode] = {r["file"]: r for r in rows}
        print(f"  {mode:8s} {time.time()-t0:.0f}초", flush=True)

    json.dump({m: {f: r["pred"] for f, r in d.items()} for m, d in out.items()},
              open(os.path.join(OUT, "96_holdout4.json"), "w",
                   encoding="utf-8"), ensure_ascii=False, indent=1)

    def score(mode):
        d = out[mode]
        ok = tot = miss = fp = exact = 0
        fy = fm = fd = 0
        per = {}
        for f in files:
            g, p = s3(lab[f]), s3(d[f]["pred"])
            hit = sum(g[i] == p[i] for i in range(3))
            per[f] = hit
            fy += g[0] == p[0]; fm += g[1] == p[1]; fd += g[2] == p[2]
            ok += hit; tot += 3
            gn, pn = g == ("NONE",) * 3, p == ("NONE",) * 3
            miss += (pn and not gn)
            fp += (gn and not pn)
            exact += g == p
        n = len(files)
        return {"score": ok / tot, "ok": ok, "tot": tot,
                "y": fy / n, "m": fm / n, "d": fd / n,
                "exact": exact, "exact_r": exact / n, "miss": miss, "fp": fp,
                "per": per, "sec": sum(d[f]["sec"] for f in files) / n}

    S = {m: score(m) for m in MODES}
    lo, hi = wilson(S["D_현재"]["ok"], S["D_현재"]["tot"])
    elo, ehi = wilson(S["D_현재"]["exact"], len(files))

    L = ["# EDA 96 — 홀드아웃4 최종 판정\n",
         "이번 라운드(88~95)의 모든 변경을 **한 번도 쓰지 않은 표본**에서 쟀다.",
         "라벨이 있던 580장은 88~91단계에서 오답을 한 장씩 들여다봤으므로",
         "더 이상 성능의 근거가 될 수 없다.",
         "",
         f"- 무작위 {len(files)}장 (94단계에서 나머지 2,769장에서 뽑음)",
         "",
         "| 구성 | **필드 정확도** | year | month | day | 완전일치 | 미검출 | 오탐 |",
         "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    NAME = {"A_예선본": "1차 예선 제출 구성", "D_현재": "**현재**"}
    for m in MODES:
        s = S[m]
        L.append(f"| {NAME[m]} | **{s['score']*100:.1f}%** | {s['y']*100:.1f}% | "
                 f"{s['m']*100:.1f}% | {s['d']*100:.1f}% | {s['exact_r']*100:.1f}% | "
                 f"{s['miss']} | {s['fp']} |")
    L.append("")
    up, dn, pv = mcnemar(S["A_예선본"]["per"], S["D_현재"]["per"])
    diff = (S["D_현재"]["score"] - S["A_예선본"]["score"]) * 100
    L.append(f"예선본 → 현재: **{diff:+.1f}%p** (개선 {up}장 / 악화 {dn}장, p={pv:.4f})")
    L.append("")
    L.append("## 오차범위\n")
    L.append(f"표본이 {len(files)}장이므로 수치에 폭이 있다. 95% 신뢰구간(Wilson):")
    L.append("")
    L.append(f"- 필드 정확도 **{S['D_현재']['score']*100:.1f}%** "
             f"(**{lo*100:.1f}% ~ {hi*100:.1f}%**)")
    L.append(f"- 완전일치 **{S['D_현재']['exact_r']*100:.1f}%** "
             f"({elo*100:.1f}% ~ {ehi*100:.1f}%)")
    L.append("")
    L.append("## 오염된 수치와의 비교\n")
    L.append("| 표본 | 필드 정확도 | 성격 |")
    L.append("| --- | ---: | --- |")
    L.append("| 라벨 580장 | 94.1% | 오답을 다 들여다보고 고친 뒤 잰 값 |")
    L.append(f"| **홀드아웃4 {len(files)}장** | **{S['D_현재']['score']*100:.1f}%** | "
             "한 번도 쓰지 않은 표본 |")
    L.append("")
    L.append("두 값의 차이가 580장에 누적된 낙관 편향의 크기다.")
    L.append("")
    L.append("## 속도\n")
    L.append("> 아래는 4워커 병렬에서 잰 개별 지연이라 채점 조건(순차)과 다르다.")
    L.append("> 단일 프로세스 실측은 92단계를 볼 것 (0.81초/장, 500장 404초, 제한의 17%).")
    L.append("")
    L.append(f"- 병렬 기준 장당 {S['D_현재']['sec']:.2f}초")

    with open(os.path.join(OUT, "96_holdout4_eval.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print()
    print("\n".join(L[5:]))
    print("\n→ eda/out/96_holdout4_eval.md")


if __name__ == "__main__":
    main()
