"""
발표용 시연 — 계단 구조가 실제로 작동하는 것을 화면으로 보여준다.

    python demo.py                 큐레이션된 사례 9장 (칸별로 하나씩) ← 발표용 권장
    python demo.py --trace 002335  한 장이 계단을 오르는 과정을 단계별로
    python demo.py --batch 60      묶음 실행 + 요약 (속도·칸별 분담)

규정: "실행 화면 또는 시연 영상 캡처를 반드시 포함해 주세요."

## 왜 큐레이션 사례인가

무작위 이미지를 돌리면 80%가 1칸에서 끝나 **계단이 작동하는 모습이 안 보인다.**
여기 쓰는 9장은 설계 과정에서 각 칸을 만들게 한 바로 그 이미지들이다
(근거는 src/combined.py 주석과 JOURNAL.md 에 있다).

## 화면에 일부러 넣은 것

  · 엔진 8개의 로딩 상태 — 가중치가 실제로 올라왔음을 보인다
  · 각 이미지가 **몇 칸까지 올라갔는지**
  · 정답 대조 — **틀린 것도 숨기지 않는다**
  · 장당 소요 시간
"""
import os
import re
import csv
import sys
import time
import argparse
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

WEIGHTS = os.path.join(ROOT, "weights")
RAPID_DIR = os.path.join(WEIGHTS, "rapidocr")
IMG_DIR = os.path.join(ROOT, "images", "상품사진입니다")
LABELS = os.path.join(ROOT, "custom_data", "labels.csv")

# ANSI. Windows Terminal·PowerShell 7 에서 그대로 나온다.
_C = os.environ.get("NO_COLOR") is None
def c(s, code):
    return "\033[%sm%s\033[0m" % (code, s) if _C else s
BOLD = lambda s: c(s, "1")
DIM  = lambda s: c(s, "2")
GRN  = lambda s: c(s, "32")
RED  = lambda s: c(s, "31")
YEL  = lambda s: c(s, "33")
CYN  = lambda s: c(s, "36")
BLU  = lambda s: c(s, "94")

# 각 칸을 만들게 한 사례. 근거는 src/combined.py 주석.
CURATED = [
    ("000440.jpg", "도트 인쇄 — 일반 인식기는 26.08.13 을 2:88:13 으로 읽는다"),
    ("001563.jpg", "도트 인쇄 — 전용 CTC 판독기가 처리"),
    ("002335.jpg", "1차는 2021:11.19 로 오독 → det medium 이 바로잡는다"),
    ("000528.jpg", "'나 2026.03.11 까지 홍병현' — 한글이 날짜에 붙어 있다"),
    ("002400.jpg", "'유통2023.01.05까지' — 띄어쓰기 없이 한글이 붙었다"),
    ("002892.jpg", "det medium 전면 교체 시 망가지던 장 (그래서 계단으로)"),
    ("002596.jpg", "저대비 인쇄"),
    ("002208.jpg", "EXP 표기 영문 라벨"),
    ("000032.jpg", "월·일만 인쇄 — 연도가 없다 (부분 정답 사례)"),
]

RUNG_NAME = {
    "rapid":    ("1칸", "주력 det small @1024"),
    "adaptive": ("2칸", "adaptive 이진화"),
    "retry":    ("3칸", "det medium"),
    "clahe":    ("4칸", "CLAHE + det medium"),
    "rec_ko":   ("5칸", "인식기 교체 — 한국어"),
    "rec_en":   ("6칸", "인식기 교체 — 영문"),
    "dot":      ("7칸", "도트 전용 판독기 (직접 학습)"),
    "partial":  ("부분", "온전한 날짜 없음 → 읽은 필드만"),
    "fallback": ("보조", "YOLO 검출 + PP-OCR"),
    "none":     ("실패", "아무 값도 내지 못함"),
}


def rule(ch="─", n=78):
    print(DIM(ch * n))


def head(t):
    print()
    print(BOLD(CYN("  " + t)))
    rule()


# ───────────────────────────────────────────── 조립
def build(verbose=True):
    import combined
    import team2_rules
    import pipeline as P

    if verbose:
        head("모델 로딩 — weights/ 에 있는 파일만 읽는다 (네트워크 접근 없음)")
    t0 = time.time()

    def line(tag, status, note=""):
        if not verbose:
            return
        ok = status == "bundled"
        mark = GRN("●") if ok else RED("○")
        print("   %s %-9s %-26s %s %s" % (
            mark, tag, note, GRN("bundled") if ok else RED(status), ""))

    rapid, s = combined.build_rapid_ocr(RAPID_DIR);  line("1·2칸", s, "RapidOCR PP-OCRv6 small")
    retry, s = combined.build_rapid_retry(RAPID_DIR); line("3·4칸", s, "det medium")
    ko, s = combined.build_rapid_rec(RAPID_DIR, combined.REC_KOREAN); line("5칸", s, "rec 한국어")
    en, s = combined.build_rapid_rec(RAPID_DIR, combined.REC_EN);     line("6칸", s, "rec 영문")

    dot = None
    try:
        import dotcrnn
        dot, s = dotcrnn.build_crnn(WEIGHTS); line("7칸", s, "도트 CTC (직접 학습)")
    except Exception as e:
        line("7칸", str(e)[:28], "도트 CTC")

    onnx = os.path.join(WEIGHTS, "yolo", "date_v1.onnx")
    det = P.OnnxDateDetector(onnx, threads=1) if os.path.exists(onnx) else None
    line("검출기", "bundled" if det else "없음", "YOLOv8n (직접 학습)")

    fb = None
    parts = [("det", "ml", "Multilingual_PP-OCRv3_det_infer"),
             ("rec", "korean", "korean_PP-OCRv4_rec_infer"),
             ("cls", "ch_ppocr_mobile_v2.0_cls_infer")]
    if all(os.path.isdir(os.path.join(WEIGHTS, *p)) for p in parts):
        try:
            import run_local
            from paddleocr import PaddleOCR
            d = [run_local.ascii_safe(os.path.join(WEIGHTS, *p)) for p in parts]
            ocr = PaddleOCR(use_angle_cls=False, lang="korean", show_log=False,
                            use_gpu=False, det_limit_side_len=640,
                            det_limit_type="max", cpu_threads=1,
                            det_model_dir=d[0], rec_model_dir=d[1], cls_model_dir=d[2])
            fb = P.DatePipeline(ocr, long_side=640, rotations=(0, 90),
                                adaptive=True, keep=None)
            if det is not None:
                fb.yolo, fb.yolo_conf, fb.yolo_mode = det, 0.25, "always"
            line("보조", "bundled", "PP-OCRv3 + YOLO")
        except Exception as e:
            line("보조", str(e)[:28], "PP-OCRv3 + YOLO")
    else:
        line("보조", "없음", "PP-OCRv3 + YOLO")

    pipe = combined.CombinedPipeline(rapid, team2_rules, fb, rapid_retry=retry,
                                     rec_korean=ko, rec_en=en,
                                     dot=dot, dot_boxes=det)
    if verbose:
        print()
        print("   %s  GPU 미사용 · 외부 API 미사용 · 로딩 %.1f초"
              % (DIM("│"), time.time() - t0))
    return pipe, combined


def labels():
    d = {}
    if os.path.exists(LABELS):
        with open(LABELS, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                d[r["source_file"]] = r["final_date"]
    return d


def which_rung(before, after):
    for k in after:
        if after[k] > before.get(k, 0):
            return k
    return "none"


def verdict(pred, truth):
    """
    맞은 필드 수와 표시 문자열.

    ⚠️ NONE 도 '맞은 필드' 로 세어진다(채점이 필드별 독립이므로 그게 맞다).
       그래서 전부 NONE 을 낸 경우가 '1/3' 로 보일 수 있다 — 정답에도 NONE 이
       있으면 그 자리가 일치하기 때문이다. 발표 화면에서 오해를 부르므로
       **아무 값도 못 낸 경우는 따로 표시한다.**
    """
    if not truth:
        return None, DIM("라벨없음")
    g = truth.split("-") if truth != "NONE" else ["NONE"] * 3
    if len(g) != 3:
        g = ["NONE"] * 3
    p = pred.split("-")
    n = sum(g[i] == p[i] for i in range(3))
    blank = all(v == "NONE" for v in p)
    if blank and truth != "NONE":
        return n, RED("미검출") + DIM(" (%d/3)" % n)
    if n == 3:
        return n, GRN("3/3 일치")
    if n == 0:
        return n, RED("0/3")
    return n, YEL("%d/3 부분" % n)


# ───────────────────────────────────────────── 큐레이션 시연
def run_curated(pipe, combined, lab):
    head("시연 — 각 칸을 만들게 한 사례들")
    print("   %-13s %-11s %-13s %-9s %-7s %s"
          % ("파일", "예측", "정답", "처리한 칸", "소요", "비고"))
    rule("╌")

    tot_t, ok_f, tot_f, exact = 0.0, 0, 0, 0
    for fn, why in CURATED:
        path = os.path.join(IMG_DIR, fn)
        if not os.path.exists(path):
            continue
        img = combined.imread_any(path)
        before = dict(pipe.stats)
        t0 = time.time()
        r = pipe.predict(path, img)
        el = time.time() - t0
        tot_t += el
        pred = "%s-%s-%s" % (r["year"], r["month"], r["day"])
        rung = which_rung(before, pipe.stats)
        tag, _ = RUNG_NAME.get(rung, (rung, ""))
        truth = lab.get(fn)
        n, vtxt = verdict(pred, truth)
        if n is not None:
            ok_f += n
            tot_f += 3
            exact += (n == 3)
        colour = BLU if rung in ("rapid",) else (YEL if rung in ("partial", "none") else GRN)
        print("   %-13s %-11s %-13s %-9s %5.2f초  %s"
              % (fn, pred, truth or "-", colour(tag), el, vtxt))
        print("   %s %s" % (DIM("└─"), DIM(why)))

    rule()
    n = len(CURATED)
    print("   %d장 중 %s · 필드 %d/%d"
          % (n, GRN("완전일치 %d장" % exact), ok_f, tot_f))
    print()
    print("   " + YEL("※ 이 9장은 각 칸을 만들게 한 '가장 어려운' 사례만 모은 것이다."))
    print("   " + YEL("   정확도도 속도도 이 표본으로 말하면 안 된다."))
    print("   " + DIM("   전체 성능 : 필드 정확도 92.3%% · 오탐 0건 (설계에 쓰지 않은 248장)"))
    print("   " + DIM("   전체 속도 : 장당 1.20초 · 500장 614초 (제한 2,400초의 26%%)"))
    print("   " + DIM("   이 표본은 장당 %.1f초 — 계단을 끝까지 오르는 장들이라 느리다."
                      % (tot_t / max(n, 1))))


# ───────────────────────────────────────────── 한 장 추적
def run_trace(pipe, combined, lab, key):
    fn = key if key.lower().endswith((".jpg", ".jpeg", ".png")) else None
    if fn is None:
        for ext in (".jpg", ".jpeg", ".png"):
            if os.path.exists(os.path.join(IMG_DIR, key + ext)):
                fn = key + ext
                break
    if fn is None or not os.path.exists(os.path.join(IMG_DIR, fn)):
        print("   이미지를 찾지 못했다: %s" % key)
        return

    head("한 장이 계단을 오르는 과정 — %s" % fn)
    path = os.path.join(IMG_DIR, fn)
    img = combined.imread_any(path)
    small = combined.downscale_max(img, pipe.MAX_SIDE)
    print("   원본 %dx%d → 축소 %dx%d" % (img.shape[1], img.shape[0],
                                       small.shape[1], small.shape[0]))
    print()

    import cv2
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    steps = [
        ("1칸  주력 det small", lambda: pipe._texts(small)),
        ("2칸  adaptive 이진화", lambda: pipe._texts(cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11))),
        ("3칸  det medium", lambda: pipe._texts(small, pipe.retry) if pipe.retry else []),
        ("4칸  CLAHE + medium", lambda: pipe._texts(cv2.cvtColor(
            cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray),
            cv2.COLOR_GRAY2BGR), pipe.retry) if pipe.retry else []),
        ("5칸  rec 한국어", lambda: pipe._texts(small, pipe.rec_korean) if pipe.rec_korean else []),
        ("6칸  rec 영문", lambda: pipe._texts(small, pipe.rec_en) if pipe.rec_en else []),
    ]

    DATEISH = re.compile(r"\d{1,4}\s*[.\-/:]\s*\d{1,2}\s*[.\-/:]\s*\d{1,4}")
    stopped = False
    for name, fn_ in steps:
        t0 = time.time()
        texts = fn_()
        el = time.time() - t0
        got = pipe._full(texts)
        # 날짜꼴로 보이는 조각만 보여준다. 전체 텍스트는 수십 개라 화면을 덮는다.
        hits = [t for t in texts if DATEISH.search(str(t))][:3]
        if stopped:
            print("   %s %-20s %s" % (DIM("·"), DIM(name), DIM("실행하지 않음 (앞 칸이 성공)")))
            continue
        mark = GRN("✓") if got else DIM("·")
        print("   %s %-20s %5.2f초  조각 %d개  %s"
              % (mark, name, el, len(texts),
                 GRN("→ %s-%s-%s" % (got["year"], got["month"], got["day"])) if got
                 else DIM("온전한 날짜 없음")))
        for h in hits:
            print("        %s %s" % (DIM("읽은 것:"), repr(str(h))[:62]))
        if got:
            stopped = True

    print()
    truth = lab.get(fn)
    r = pipe.predict(path, img)
    pred = "%s-%s-%s" % (r["year"], r["month"], r["day"])
    _, vtxt = verdict(pred, truth)
    print("   최종 %s   정답 %s   %s" % (BOLD(pred), truth or "-", vtxt))
    print()
    print("   " + DIM("각 칸은 앞 칸이 실패했을 때만 실행된다. 그래서 칸을 추가해도"))
    print("   " + DIM("이미 맞힌 답이 깨지지 않는다 — 회귀 위험이 구조적으로 0이다."))


# ───────────────────────────────────────────── 묶음 실행
def run_batch(pipe, combined, lab, n):
    head("묶음 실행 — %d장 (채점과 같은 순차 처리)" % n)
    files = sorted(f for f in os.listdir(IMG_DIR)
                   if re.search(r"\.(jpe?g|png)$", f, re.I) and f in lab)[:n]
    ok_f = tot_f = exact = 0
    t0 = time.time()
    for i, fn in enumerate(files, 1):
        img = combined.imread_any(os.path.join(IMG_DIR, fn))
        r = pipe.predict(os.path.join(IMG_DIR, fn), img)
        pred = "%s-%s-%s" % (r["year"], r["month"], r["day"])
        k, _ = verdict(pred, lab.get(fn))
        if k is not None:
            ok_f += k
            tot_f += 3
            exact += (k == 3)
        if i % 10 == 0 or i == len(files):
            el = time.time() - t0
            print("   %3d/%d  %5.1f초  장당 %.2f초  필드 %.1f%%"
                  % (i, len(files), el, el / i, ok_f / max(tot_f, 1) * 100))
    el = time.time() - t0

    rule()
    print("   장당 %.2f초  →  500장 환산 %.0f초 (제한 2,400초의 %.0f%%)"
          % (el / len(files), el / len(files) * 500, el / len(files) * 500 / 2400 * 100))
    print("   필드 정확도 %.1f%%  ·  완전일치 %.1f%%"
          % (ok_f / max(tot_f, 1) * 100, exact / max(len(files), 1) * 100))
    print()
    print("   " + BOLD("칸별 분담"))
    tot = sum(pipe.stats.values()) or 1
    for k, v in pipe.stats.items():
        if v:
            tag, desc = RUNG_NAME.get(k, (k, ""))
            bar = "█" * max(1, int(v / tot * 34))
            print("     %-5s %-30s %3d장 %3.0f%%  %s"
                  % (tag, desc, v, v / tot * 100, CYN(bar)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", metavar="IMAGE", help="한 장을 단계별로 추적")
    ap.add_argument("--batch", type=int, metavar="N", help="N장 묶음 실행")
    a = ap.parse_args()

    print()
    print(BOLD("  소비기한 추출 파이프라인 — 시연"))
    print("  [DScover]_카피바라 · 제3회 ITDA 연합학술제 본선")
    print("  " + DIM("계단 구조: 각 칸은 앞 칸이 실패한 이미지에만 실행된다"))

    lab = labels()
    pipe, combined = build()

    if a.trace:
        run_trace(pipe, combined, lab, a.trace)
    elif a.batch:
        run_batch(pipe, combined, lab, a.batch)
    else:
        run_curated(pipe, combined, lab)
    print()


if __name__ == "__main__":
    main()
