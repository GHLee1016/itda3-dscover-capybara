"""
로컬 실행 점검 — 제출 노트북과 같은 구성으로 돌려 본다.

    python run_local.py              # 30장 (빠른 확인)
    python run_local.py --n 200      # 200장
    python run_local.py --all        # 전량 3,352장 (약 1시간)

## 왜 이 스크립트가 필요한가

제출 노트북은 `weights/` 안의 파일만 읽는다. 그런데 실험 스크립트(`eda/`)는
`C:\\Users\\Public\\ocr_models` 라는 **다른 경로**에서 보조 엔진 가중치를 읽고
있었다. 그래서 두 가지가 어긋나 있었다.

    측정에 쓴 구성   보조 엔진 O  →  92.3%
    노트북을 그냥 돌리면  보조 엔진 X  →  더 낮다

`weights/det|rec|cls` 가 비어 있으면 보조 경로가 조용히 꺼지고, 아무 에러도
나지 않는다. 이 스크립트는 **그 상태를 먼저 드러내고 고친 뒤** 실행한다.

## 하는 일

    1. 가중치 점검        무엇이 있고 무엇이 없는지 표로 보여 준다
    2. 보조 가중치 복구    Public\\ocr_models 에 있으면 weights/ 로 복사
    3. 파이프라인 조립     노트북과 같은 방식 (weights/ 에서만 읽는다)
    4. 실행               submission.csv 생성 + 라벨이 있는 장은 정답 대조
"""
import os
import re
import csv
import sys
import time
import shutil
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
OUT_CSV = os.path.join(ROOT, "submission.csv")

# 보조 엔진(PaddleOCR) 가중치가 로컬 어딘가에 이미 있으면 여기서 가져온다.
# download_weights.sh 가 받아 주는 것과 같은 파일이다.
PADDLE_CACHE = r"C:\Users\Public\ocr_models"
PADDLE_PARTS = [
    ("det", "ml", "Multilingual_PP-OCRv3_det_infer"),
    ("rec", "korean", "korean_PP-OCRv4_rec_infer"),
    ("cls", "ch_ppocr_mobile_v2.0_cls_infer"),
]


_ASCII_TMP = None


def ascii_safe(path):
    """
    PaddleOCR 의 C++ 백엔드는 **비ASCII 경로의 모델을 열지 못한다.**
    같은 파일을 두 경로에 두고 확인했다.
        C:\\Users\\Public\\ocr_models   로딩 성공
        ...\\연합 학술제\\weights       (NotFound) Cannot open file ...
    게다가 조용히 실패한다 — 보조 경로만 꺼지고 에러가 안 난다.

    경로가 비ASCII 면 ASCII 임시 폴더로 복사해 그 경로를 넘긴다.
    ASCII 환경에서는 아무것도 하지 않는다.
    """
    global _ASCII_TMP
    if path is None:
        return None
    try:
        path.encode("ascii")
        return path
    except UnicodeEncodeError:
        pass
    import tempfile
    if _ASCII_TMP is None:
        for cand in (tempfile.gettempdir(), os.environ.get("PUBLIC"),
                     "/tmp", os.getcwd()):
            if not cand:
                continue
            try:
                cand.encode("ascii")
            except (UnicodeEncodeError, AttributeError):
                continue
            if os.path.isdir(cand) and os.access(cand, os.W_OK):
                _ASCII_TMP = tempfile.mkdtemp(prefix="itda_w_", dir=cand)
                break
    if _ASCII_TMP is None:
        return None
    dst = os.path.join(_ASCII_TMP, os.path.basename(path.rstrip("\\/")))
    if not os.path.exists(dst):
        shutil.copytree(path, dst, dirs_exist_ok=True) if os.path.isdir(path) \
            else shutil.copy2(path, dst)
    return dst


def line(ch="─", n=66):
    print(ch * n)


def step(no, title):
    print()
    line()
    print(" %d. %s" % (no, title))
    line()


# ────────────────────────────────────────────────── 1. 가중치 점검
def check_weights():
    step(1, "가중치 점검")
    need = [
        ("주력 det small", "rapidocr/PP-OCRv6_det_small.onnx", "필수"),
        ("주력 rec small", "rapidocr/PP-OCRv6_rec_small.onnx", "필수"),
        ("방향 분류기", "rapidocr/ch_ppocr_mobile_v2.0_cls_mobile.onnx", "필수"),
        ("3·4칸 det medium", "rapidocr/PP-OCRv6_det_medium.onnx", "권장"),
        ("5칸 rec 한국어", "rapidocr/korean_PP-OCRv5_rec_mobile.onnx", "권장"),
        ("6칸 rec 영문", "rapidocr/en_PP-OCRv5_rec_mobile.onnx", "권장"),
        ("7칸 도트 판독기", "dot/dotcrnn_v1.onnx", "권장"),
        ("날짜 검출기(YOLO)", "yolo/date_v1.onnx", "권장"),
    ]
    missing_req = []
    for name, rel, grade in need:
        p = os.path.join(WEIGHTS, rel)
        ok = os.path.exists(p)
        mb = (" %6.1f MB" % (os.path.getsize(p) / 1e6)) if ok else "        —"
        print("  %-18s %-4s %s  %s" % (name, grade, "O" if ok else "X", mb))
        if not ok and grade == "필수":
            missing_req.append(rel)

    # 보조 엔진은 폴더 단위로 본다
    print()
    lack = []
    for parts in PADDLE_PARTS:
        dst = os.path.join(WEIGHTS, *parts)
        ok = os.path.isdir(dst)
        print("  %-18s %-4s %s  %s" % ("보조 " + parts[0], "권장",
                                       "O" if ok else "X", "" if ok else "없음"))
        if not ok:
            lack.append(parts)

    if missing_req:
        print("\n  !! 필수 가중치가 없다. `bash download_weights.sh` 를 먼저 돌려야 한다.")
        sys.exit(1)
    return lack


# ────────────────────────────────────────────────── 2. 보조 가중치 복구
def restore_paddle(lack):
    step(2, "보조 엔진 가중치 복구")
    if not lack:
        print("  이미 weights/ 안에 있다. 복사할 것이 없다.")
        return True
    if not os.path.isdir(PADDLE_CACHE):
        print("  %s 가 없다." % PADDLE_CACHE)
        print("  → `bash download_weights.sh` 로 받아야 한다.")
        print("  → 지금은 보조 경로 없이 진행한다 (정확도가 조금 내려간다).")
        return False
    for parts in lack:
        src = os.path.join(PADDLE_CACHE, *parts)
        dst = os.path.join(WEIGHTS, *parts)
        if not os.path.isdir(src):
            print("  X %-40s 원본이 없다" % ("/".join(parts)))
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copytree(src, dst, dirs_exist_ok=True)
        mb = sum(os.path.getsize(os.path.join(r, f))
                 for r, _, fs in os.walk(dst) for f in fs) / 1e6
        print("  O %-40s %6.1f MB 복사" % ("/".join(parts), mb))
    return True


# ────────────────────────────────────────────────── 3. 파이프라인 조립
def build():
    step(3, "파이프라인 조립 (노트북과 같은 구성)")
    import combined
    import team2_rules
    import pipeline as P

    t0 = time.time()
    rapid, s = combined.build_rapid_ocr(RAPID_DIR)
    print("  1·2칸  주력 RapidOCR          %s" % s)
    retry, s = combined.build_rapid_retry(RAPID_DIR)
    print("  3·4칸  det medium             %s" % s)
    ko, s = combined.build_rapid_rec(RAPID_DIR, combined.REC_KOREAN)
    print("  5칸    rec 한국어             %s" % s)
    en, s = combined.build_rapid_rec(RAPID_DIR, combined.REC_EN)
    print("  6칸    rec 영문               %s" % s)

    dot = None
    try:
        import dotcrnn
        dot, s = dotcrnn.build_crnn(WEIGHTS)
        print("  7칸    도트 판독기           %s" % s)
    except Exception as e:
        print("  7칸    도트 판독기           건너뜀 (%s)" % e)

    onnx = os.path.join(WEIGHTS, "yolo", "date_v1.onnx")
    det = P.OnnxDateDetector(onnx, threads=1) if os.path.exists(onnx) else None
    print("  검출기  YOLO date_v1           %s" % ("bundled" if det else "없음"))

    # 보조 경로 — weights/ 에서만 읽는다. 노트북과 같은 규칙이다.
    fb = None
    parts_ok = all(os.path.isdir(os.path.join(WEIGHTS, *p)) for p in PADDLE_PARTS)
    if parts_ok:
        try:
            from paddleocr import PaddleOCR
            # 한글 경로면 ASCII 임시 폴더로 옮겨 넘긴다 (ascii_safe 주석 참고)
            _d = [ascii_safe(os.path.join(WEIGHTS, *p)) for p in PADDLE_PARTS]
            if any(x is None for x in _d):
                raise RuntimeError("ASCII 경로를 확보하지 못했다")
            if _d[0] != os.path.join(WEIGHTS, *PADDLE_PARTS[0]):
                print("  (한글 경로 우회: 모델을 %s 로 복사)"
                      % os.path.dirname(_d[0]))
            ocr = PaddleOCR(
                use_angle_cls=False, lang="korean", show_log=False, use_gpu=False,
                det_limit_side_len=640, det_limit_type="max", cpu_threads=1,
                det_model_dir=_d[0], rec_model_dir=_d[1], cls_model_dir=_d[2])
            fb = P.DatePipeline(ocr, long_side=640, rotations=(0, 90),
                                adaptive=True, keep=None)
            if det is not None:
                fb.yolo = det
                fb.yolo_conf = 0.25
                fb.yolo_mode = "always"
            print("  보조    PP-OCRv3 + YOLO       bundled")
        except Exception as e:
            print("  보조    PP-OCRv3 + YOLO       실패 (%s)" % e)
    else:
        print("  보조    PP-OCRv3 + YOLO       없음 — 이 경로가 꺼진다")

    pipe = combined.CombinedPipeline(rapid, team2_rules, fb, rapid_retry=retry,
                                     rec_korean=ko, rec_en=en,
                                     dot=dot, dot_boxes=det)
    print("\n  조립 %.1f초" % (time.time() - t0))
    return pipe, combined


# ────────────────────────────────────────────────── 4. 실행
def run(pipe, combined, files, labels):
    step(4, "실행 — %d장" % len(files))
    rows, ok_f, tot_f, exact, miss = [], 0, 0, 0, 0
    t0 = time.time()
    for i, f in enumerate(files, 1):
        img = combined.imread_any(os.path.join(IMG_DIR, f))
        r = pipe.predict(os.path.join(IMG_DIR, f), img)
        image_id = os.path.splitext(f)[0]
        rows.append(combined.to_row(image_id, r))
        pred = "%s-%s-%s" % (r["year"], r["month"], r["day"])

        mark = ""
        if f in labels:
            g = labels[f]
            gg = g.split("-") if g and g != "NONE" else ["NONE"] * 3
            pp = [r["year"], r["month"], r["day"]]
            if len(gg) != 3:
                gg = ["NONE"] * 3
            n = sum(gg[j] == pp[j] for j in range(3))
            ok_f += n
            tot_f += 3
            if n == 3:
                exact += 1
            if all(v == "NONE" for v in pp) and g not in (None, "NONE"):
                miss += 1
            mark = "  %d/3  정답 %s" % (n, g)

        if i <= 40 or i % 50 == 0:
            print("  [%4d/%d] %-14s %-14s%s" % (i, len(files), f, pred, mark))

    el = time.time() - t0
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, ["image_id", "year", "month", "day", "final_date"])
        w.writeheader()
        w.writerows(rows)

    step(5, "결과")
    print("  제출 파일     %s (%d행)" % (OUT_CSV, len(rows)))
    print("  소요          %.1f초  ·  장당 %.2f초" % (el, el / max(len(files), 1)))
    print("  500장 환산    %.0f초  (제한 2,500초의 %.0f%%)"
          % (el / len(files) * 500, el / len(files) * 500 / 2500 * 100))
    if tot_f:
        print()
        print("  라벨이 있는 장 %d개로 대조:" % (tot_f // 3))
        print("    필드 정확도  %.1f%%  (%d/%d)" % (ok_f / tot_f * 100, ok_f, tot_f))
        print("    완전일치     %.1f%%" % (exact / (tot_f // 3) * 100))
        print("    미검출       %d장" % miss)
    print()
    print("  칸별 분담:")
    KO = {"rapid": "1칸 주력", "adaptive": "2칸 adaptive", "retry": "3칸 det medium",
          "clahe": "4칸 CLAHE", "rec_ko": "5칸 rec 한국어", "rec_en": "6칸 rec 영문",
          "dot": "7칸 도트 판독기", "partial": "부분 결과",
          "fallback": "보조 경로", "none": "실패"}
    tot = sum(pipe.stats.values()) or 1
    for k, v in pipe.stats.items():
        if v:
            print("    %-16s %4d장  %3.0f%%" % (KO.get(k, k), v, v / tot * 100))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="처리할 장수")
    ap.add_argument("--all", action="store_true", help="전량 처리")
    a = ap.parse_args()

    print()
    print("  소비기한 추출 — 로컬 실행 점검")
    print("  [DScover]_카피바라")

    lack = check_weights()
    restore_paddle(lack)

    labels = {}
    if os.path.exists(LABELS):
        with open(LABELS, encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                labels[r["source_file"]] = r["final_date"]

    files = sorted(f for f in os.listdir(IMG_DIR)
                   if re.search(r"\.(jpe?g|png)$", f, re.I))
    if not a.all:
        # 라벨이 있는 것부터 고른다 — 정답 대조가 되어야 확인이 된다
        have = [f for f in files if f in labels]
        files = (have or files)[:a.n]

    pipe, combined = build()
    run(pipe, combined, files, labels)
    print()


if __name__ == "__main__":
    main()
