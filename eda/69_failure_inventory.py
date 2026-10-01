"""
EDA 69단계 — 전량 실패 목록과 원인 분류 (능동학습 1단계)

목적: 최종 파이프라인을 홀드아웃2를 제외한 전량에 돌려, 실패 이미지를 모으고
      **실패 원인을 단계별로 분류**한다. 다음 단계(임베딩 군집 → 박스 라벨링)의
      입력이 된다.

⚠️ 정보 누수 차단 (문서 §3.5)
   선발에 정답을 쓰지 않는다. 이 스크립트는 '파이프라인이 무엇을 출력했는가'와
   '어느 단계에서 멈췄는가'만 기록한다. 정답은 라벨이 있는 이미지의 정확도를
   참고로 찍을 때만 쓰고, 실패 선발 기준에는 넣지 않는다.

⚠️ 평가셋 동결
   홀드아웃2 198장은 제외한다. 최종 측정에만 쓰기 위해서다. 이미 설정 선택에
   여러 번 노출돼 낙관 편향이 있으므로, 본선용으로는 신규 홀드아웃을 따로
   만드는 것이 맞다(70단계 예정).

실패 원인 분류
  no_text      RapidOCR 이 글자를 한 조각도 못 찾음      → 검출 문제. YOLO 재학습의 표적
  text_no_date 글자는 찾았는데 날짜 후보가 전혀 없음     → 검출 또는 인식 문제
  date_dropped 날짜 비슷한 문자열은 있는데 규칙이 버림   → 규칙 문제. 라벨링 말고 규칙 수정
  fallback_fail 보조 경로까지 갔는데 실패                → 가장 어려운 집합

    python eda/69_failure_inventory.py --workers 4
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

IMG_DIR = os.path.join(ROOT, "images", "상품사진입니다")
OUT_DIR = os.path.join(ROOT, "eda", "out")
ONNX = os.path.join(ROOT, "weights", "yolo", "date_v1.onnx")
RAPID_DIR = os.path.join(ROOT, "weights", "rapidocr")

_S = None


def setup():
    """워커마다 한 번. 파이프라인 부품을 직접 들고 있어야 단계별 계측이 된다."""
    global _S
    if _S is not None:
        return _S
    import pipeline as P
    import team2_rules
    import combined
    import ocrlib
    rapid, src = combined.build_rapid_ocr(RAPID_DIR)
    fb = P.DatePipeline(ocrlib.build_ocr(det_side=640, threads=1),
                        long_side=640, rotations=(0, 90), adaptive=True, keep=None)
    if os.path.exists(ONNX):
        fb.yolo = P.OnnxDateDetector(ONNX, threads=1)
        fb.yolo_conf = 0.25
        fb.yolo_mode = "always"
    _S = (P, team2_rules, combined, rapid, fb)
    return _S


def digits(s):
    return sum(c.isdigit() for c in str(s))


def work(fname):
    import cv2
    P, rules, combined, rapid, fb = setup()
    t0 = time.time()
    rec = {"file": fname, "stage": "none", "cause": "unknown",
           "year": "NONE", "month": "NONE", "day": "NONE",
           "n_frag": 0, "n_digit_frag": 0, "w": 0, "h": 0, "sec": 0.0}

    img = combined.imread_any(os.path.join(IMG_DIR, fname))
    if img is None:
        rec["cause"] = "load_fail"
        return rec
    rec["h"], rec["w"] = img.shape[:2]

    def texts_of(a):
        try:
            r = rapid(a)
        except Exception:
            return []
        return list(r.txts) if getattr(r, "txts", None) else []

    small = combined.downscale_max(img, 1024)

    # --- 주력 1차 ---
    tx = texts_of(small)
    rec["n_frag"] = len(tx)
    rec["n_digit_frag"] = sum(1 for t in tx if digits(t) >= 3)
    got = None
    if tx:
        full = rules.choose_expiration_date(tx)
        if full:
            y, m, d = full.split("-")
            got = {"year": y, "month": m, "day": d}
        else:
            got = rules.extract_partial_date(tx)
    if got:
        rec.update(got)
        rec["stage"] = "rapid"
        rec["cause"] = "ok"
        rec["sec"] = time.time() - t0
        return rec

    # --- 주력 2차: adaptive threshold ---
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    adp = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 11)
    tx2 = texts_of(adp)
    rec["n_frag"] = max(rec["n_frag"], len(tx2))
    rec["n_digit_frag"] = max(rec["n_digit_frag"],
                              sum(1 for t in tx2 if digits(t) >= 3))
    if tx2:
        full = rules.choose_expiration_date(tx2)
        if full:
            y, m, d = full.split("-")
            got = {"year": y, "month": m, "day": d}
        else:
            got = rules.extract_partial_date(tx2)
    if got:
        rec.update(got)
        rec["stage"] = "adaptive"
        rec["cause"] = "ok"
        rec["sec"] = time.time() - t0
        return rec

    # --- 보조 경로 ---
    try:
        fin, _, _ = fb.predict_image(img)
    except Exception:
        fin = None
    if fin:
        out = {}
        for k in ("year", "month", "day"):
            v = fin.get(k[0])
            out[k] = "NONE" if v is None else ("%04d" % v if k == "year" else "%02d" % v)
        if any(v != "NONE" for v in out.values()):
            rec.update(out)
            rec["stage"] = "fallback"
            rec["cause"] = "ok"
            rec["sec"] = time.time() - t0
            return rec

    # --- 실패. 어디서 막혔는지 분류 ---
    allt = tx + tx2
    if not allt:
        rec["cause"] = "no_text"            # 글자를 한 조각도 못 찾음
    elif rec["n_digit_frag"] == 0:
        rec["cause"] = "text_no_digits"     # 글자는 있는데 숫자 줄이 없음
    else:
        # 숫자 3개 이상인 조각이 있는데도 날짜가 안 나왔다 = 인식 오류 또는 규칙 탈락
        rec["cause"] = "digits_no_date"
    rec["stage"] = "none"
    rec["sec"] = time.time() - t0
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
    allf = sorted(f for f in os.listdir(IMG_DIR)
                  if f.lower().endswith(EXTS) and not f.startswith("."))

    # 홀드아웃2 는 최종 측정용으로 동결한다
    hold2 = set(json.load(open(os.path.join(OUT_DIR, "truth_holdout2.json"),
                               encoding="utf-8"))["labels"])
    files = [f for f in allf if f not in hold2]
    if a.limit:
        files = files[:a.limit]

    print(f"전체 {len(allf)}장 중 홀드아웃2 {len(hold2)}장 제외 → {len(files)}장 처리",
          flush=True)

    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(work, files, chunksize=4), 1):
            rows.append(r)
            if i % 250 == 0:
                el = time.time() - t0
                print(f"  {i}/{len(files)}  {el:.0f}s  ({el/i:.2f}s/장, "
                      f"남은 {(len(files)-i)*el/i/60:.0f}분)", flush=True)
    wall = time.time() - t0

    json.dump(rows, open(os.path.join(OUT_DIR, "69_failures.json"), "w",
                         encoding="utf-8"), ensure_ascii=False)

    n = len(rows)
    stage = Counter(r["stage"] for r in rows)
    fails = [r for r in rows if r["stage"] == "none"]
    cause = Counter(r["cause"] for r in fails)

    # 라벨이 있는 이미지에 한해 정확도를 참고로만 본다 (선발 기준에는 쓰지 않음)
    lab = {}
    for fn in ("truth_v4.json", "truth_holdout.json"):
        p = os.path.join(OUT_DIR, fn)
        if os.path.exists(p):
            lab.update(json.load(open(p, encoding="utf-8"))["labels"])

    def s3(v):
        if not v or v == "NONE":
            return ("NONE",) * 3
        q = str(v).split("-")
        return tuple(q) if len(q) == 3 else ("NONE",) * 3

    ok = tot = 0
    for r in rows:
        if r["file"] not in lab:
            continue
        g, p = s3(lab[r["file"]]), (r["year"], r["month"], r["day"])
        ok += sum(g[i] == p[i] for i in range(3))
        tot += 3

    CAUSE_KO = {
        "no_text": "글자를 한 조각도 못 찾음 (검출 실패)",
        "text_no_digits": "글자는 찾았으나 숫자 줄이 없음",
        "digits_no_date": "숫자 줄은 있으나 날짜로 파싱 안 됨",
        "load_fail": "이미지 로드 실패",
        "unknown": "분류 불가",
    }

    L = ["# EDA 69 — 전량 실패 목록과 원인 분류 (능동학습 1단계)\n",
         "최종 파이프라인을 **홀드아웃2를 제외한 전량**에 돌렸다. 홀드아웃2 198장은",
         "최종 측정용으로 동결한다. 선발에 정답을 쓰지 않았다(문서 §3.5 정보 누수 차단).",
         "",
         f"- 처리 {n}장, {wall/60:.0f}분 ({wall/n:.2f}초/장, {a.workers}워커)",
         "",
         "## 어느 경로가 처리했나\n",
         "| 경로 | 장수 | 비율 |",
         "| --- | ---: | ---: |"]
    NAME = {"rapid": "주력 1차 (RapidOCR)", "adaptive": "주력 2차 (adaptive)",
            "fallback": "보조 경로 (PP-OCR + YOLO)", "none": "**실패**"}
    for k in ("rapid", "adaptive", "fallback", "none"):
        L.append(f"| {NAME[k]} | {stage.get(k,0)} | {stage.get(k,0)/n*100:.1f}% |")
    L.append("")
    L.append(f"**실패 {len(fails)}장 ({len(fails)/n*100:.1f}%)** 이 다음 단계의 모집단이다.")
    L.append("")

    L.append("## 실패 원인 분류\n")
    L.append("| 원인 | 장수 | 실패 중 비율 | 대응 |")
    L.append("| --- | ---: | ---: | --- |")
    PLAN = {
        "no_text": "**박스 라벨링 → YOLO 재학습**. 가장 직접적인 표적",
        "text_no_digits": "박스 라벨링 대상. 날짜 영역만 못 본 경우",
        "digits_no_date": "규칙 수정으로 접근. 라벨링보다 값이 쌀 수 있다",
        "load_fail": "파일 문제. 별도 확인",
        "unknown": "재확인 필요",
    }
    for c, v in cause.most_common():
        L.append(f"| {CAUSE_KO.get(c,c)} | {v} | {v/max(len(fails),1)*100:.1f}% | "
                 f"{PLAN.get(c,'')} |")
    L.append("")

    if tot:
        L.append(f"> 참고: 라벨이 있는 {tot//3}장에서의 필드 정확도 **{ok/tot*100:.1f}%**. "
                 "이 수치는 선발에 쓰지 않았고 경향 확인용이다.")
        L.append("")

    # 실패 이미지 해상도 분포 — 군집화 전에 대략의 성격을 본다
    if fails:
        px = sorted(r["w"] * r["h"] for r in fails)
        med = px[len(px) // 2]
        small_n = sum(1 for r in fails if max(r["w"], r["h"]) <= 800)
        L.append("## 실패 이미지의 성격\n")
        L.append(f"- 화소수 중앙값 {med/1e6:.1f}MP")
        L.append(f"- 긴 변 800px 이하인 저해상도 이미지 {small_n}장 "
                 f"({small_n/len(fails)*100:.1f}%)")
        L.append("")

    L.append("## 다음 단계\n")
    L.append("1. 실패 이미지를 임베딩으로 군집화해 **서로 다른 실패 유형**의 대표를 뽑는다")
    L.append("   (문서 §9.1: 중심 기반 선택이 무작위보다 크게 앞섰다)")
    L.append("2. 뽑힌 이미지에 **날짜 영역 박스**를 사람이 라벨링한다")
    L.append("3. 기존 238박스 + 신규 박스로 YOLO 재학습")
    L.append("4. **신규 홀드아웃**으로 측정한다 — 홀드아웃2 로 재면 개선폭을 믿을 수 없다")
    L.append("")
    L.append("> `eda/out/69_failures.json` 에 전량 예측과 단계·원인이 들어 있다.")

    with open(os.path.join(OUT_DIR, "69_failure_inventory.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L))
    print()
    print("\n".join(L[5:30]))
    print("\n→ eda/out/69_failure_inventory.md")


if __name__ == "__main__":
    main()
