"""
EDA 113단계 — 도트 인쇄 날짜를 합성해 YOLO 학습 데이터를 만든다

왜 필요한가 (EDA 112)
  도트 판독기(CTC)는 됐다. `20.08.05` 를 정확히 읽어 미검출 2장을 살렸다.
  그런데 **YOLO 가 12장 중 7장에서 날짜 영역 자체를 못 잡는다.** 판독기가
  아무리 좋아도 그 7장은 손도 못 댄다.

  못 잡는 것들의 공통점을 눈으로 확인했다.
      투명 필름 아래 인쇄 · 연한 회색 도트 · 심하게 기울어진 포장
  기존 YOLO 는 일반 인쇄 날짜로 학습해서(약지도로 PP-OCR 성공 박스를 모음)
  이런 조건을 본 적이 없다.

어떻게 만드나
  실제 상품 사진을 배경으로 쓰고 그 위에 도트 날짜를 합성한다. **붙인 자리가
  곧 정답 박스**이므로 라벨링이 필요 없다. 못 잡던 조건을 일부러 만든다.
      대비를 낮추고 · 기울이고 · 흐리게 하고 · 잡음 위에 얹는다

  배경은 **라벨이 없는 이미지에서만** 고른다. 홀드아웃에 쓰는 장이 학습에
  섞이면 그 표본으로 성능을 잴 수 없게 된다.

    python eda/113_yolo_dot_synth.py --n 1600
"""
import os
import csv
import sys
import json
import random
import argparse
import warnings

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(ROOT, "images", "상품사진입니다")
OUT = os.path.join(ROOT, "eda", "out")
DS = os.path.join(ROOT, "yolo_ds_dot")
SIDE = 640

FONTS = [r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\consolab.ttf",
         r"C:\Windows\Fonts\cour.ttf", r"C:\Windows\Fonts\courbd.ttf",
         r"C:\Windows\Fonts\verdanab.ttf", r"C:\Windows\Fonts\tahomabd.ttf"]
NOISE = ["까지", "부터", "제조일자", "유통기한", "소비기한", "EXP", "EXP DATE",
         "BB", "BEST BEFORE", "L20238", "A1", "12:47"]


def labeled_files():
    """라벨이 있는 것 = 평가에 쓰는 것. 배경에서 뺀다."""
    used = set()
    p = os.path.join(ROOT, "custom_data", "labels.csv")
    if os.path.exists(p):
        with open(p, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                used.add(r["source_file"])
    return used


def date_text(rng):
    y4 = rng.random() < 0.45
    yy = rng.randint(2018, 2030)
    y = str(yy) if y4 else f"{yy % 100:02d}"
    m, d = f"{rng.randint(1,12):02d}", f"{rng.randint(1,28):02d}"
    sep = rng.choice([".", ".", "/", "-", " "])
    core = f"{y}{sep}{m}{sep}{d}"
    if rng.random() < 0.45:
        core = (rng.choice(NOISE) + rng.choice([" ", ":", ""]) + core
                if rng.random() < 0.5 else core + " " + rng.choice(NOISE))
    if rng.random() < 0.3:                       # 두 줄 (제조 + 소비)
        y2 = str(yy + 1) if y4 else f"{(yy+1) % 100:02d}"
        core += "\n" + f"{y2}{sep}{m}{sep}{d}" + rng.choice([" 까지", "", " A1"])
    return core


def make_patch(np, cv2, Image, ImageDraw, ImageFont, rng):
    """도트 날짜 패치(알파 포함)를 만든다. -> (patch BGR, alpha 0~1)"""
    txt = date_text(rng)
    size = rng.randint(16, 40)
    fp = rng.choice([p for p in FONTS if os.path.exists(p)] or [None])
    try:
        font = ImageFont.truetype(fp, size) if fp else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    lines = txt.split("\n")
    w = max(len(l) for l in lines) * size + 40
    h = size * (len(lines) + 1) + 30
    im = Image.new("L", (w, h), 0)
    ImageDraw.Draw(im).multiline_text((15, 10), txt, fill=255, font=font, spacing=4)
    a = np.array(im)

    # 도트화: 점 격자로 샘플링
    pitch = rng.randint(3, 7)
    rad = max(1, int(pitch * rng.uniform(0.3, 0.5)))
    ox, oy = rng.randrange(pitch), rng.randrange(pitch)
    mask = np.zeros_like(a)
    for yy_ in range(oy, h, pitch):
        for xx_ in range(ox, w, pitch):
            cv2.circle(mask, (xx_, yy_), rad, 255, -1)
    a = cv2.bitwise_and(a, mask)

    if rng.random() < 0.5:
        k = rng.choice([2, 3])
        a = cv2.dilate(a, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    if rng.random() < 0.7:                        # 기울기 — 못 잡던 조건
        ang = rng.uniform(-14, 14)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        a = cv2.warpAffine(a, M, (w, h))
    if rng.random() < 0.6:                        # 흐림
        a = cv2.GaussianBlur(a, (0, 0), rng.uniform(0.5, 2.0))

    ys, xs = np.where(a > 25)
    if len(ys) < 30:
        return None, None
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    alpha = (a.astype(np.float32) / 255.0) * rng.uniform(0.45, 1.0)  # 낮은 대비
    ink = rng.randint(0, 90)                       # 잉크 밝기 (검정~짙은 회색)
    patch = np.full((a.shape[0], a.shape[1], 3), ink, np.uint8)
    return patch, alpha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1600)
    ap.add_argument("--val", type=float, default=0.12)
    a = ap.parse_args()

    import numpy as np
    import cv2
    from PIL import Image, ImageDraw, ImageFont

    # 기존 검출기로 배경의 진짜 날짜 위치를 찾는다(덮어쓰기 위해)
    sys.path.insert(0, os.path.join(ROOT, "src"))
    det = None
    onnx = os.path.join(ROOT, "weights", "yolo", "date_v1.onnx")
    if os.path.exists(onnx):
        try:
            import pipeline as P
            det = P.OnnxDateDetector(onnx, threads=1)
        except Exception as e:
            print("기존 검출기 로딩 실패(무작위 위치로 진행):", e)

    rng = random.Random(7)
    used = labeled_files()
    pool = [f for f in sorted(os.listdir(IMG_DIR))
            if os.path.splitext(f)[1].lower() in (".jpg", ".jpeg", ".png")
            and f not in used]
    print(f"배경 후보 {len(pool)}장 (라벨 있는 {len(used)}장은 제외)")

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        os.makedirs(os.path.join(DS, sub), exist_ok=True)

    made = 0
    tries = 0
    while made < a.n and tries < a.n * 4:
        tries += 1
        bgf = rng.choice(pool)
        img = cv2.imdecode(np.fromfile(os.path.join(IMG_DIR, bgf), np.uint8),
                           cv2.IMREAD_COLOR)
        if img is None:
            continue
        # 640 정사각 레터박스 — YOLO 입력과 같은 규격
        h, w = img.shape[:2]
        s = SIDE / max(h, w)
        rz = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))))
        canvas = np.zeros((SIDE, SIDE, 3), np.uint8)
        canvas[:rz.shape[0], :rz.shape[1]] = rz

        patch, alpha = make_patch(np, cv2, Image, ImageDraw, ImageFont, rng)
        if patch is None:
            continue
        ph, pw = patch.shape[:2]
        scale = min(1.0, (SIDE * rng.uniform(0.18, 0.55)) / pw)
        pw2, ph2 = max(8, int(pw * scale)), max(6, int(ph * scale))
        if pw2 >= SIDE - 8 or ph2 >= SIDE - 8:
            continue
        patch = cv2.resize(patch, (pw2, ph2))
        alpha = cv2.resize(alpha, (pw2, ph2))[..., None]

        # ⚠️ 배경 사진에는 **이미 진짜 날짜가 찍혀 있다**. 합성 자리만 라벨로
        #    주면 YOLO 는 "저기 날짜가 있는데 정답이 아니다" 를 배워 검출력이
        #    오히려 떨어진다.
        #
        #    덮어쓰기도 해 봤지만 원래 글자가 비어져 나와 깔끔하지 않았다.
        #    그래서 **기존 검출기가 찾은 날짜도 함께 라벨에 넣는다.** 그러면
        #    "날짜처럼 생긴 것은 전부 날짜" 로 일관되고, 기존 지식도 지켜진다.
        keep = []
        if det is not None:
            try:
                for bx1, by1, bx2, by2 in det.detect(canvas, conf=0.3)[:4]:
                    if bx2 - bx1 >= 8 and by2 - by1 >= 6:
                        keep.append((float(bx1), float(by1),
                                     float(bx2), float(by2)))
            except Exception:
                pass

        # 합성 패치는 기존 날짜와 겹치지 않는 자리에 놓는다
        x0 = y0 = None
        for _ in range(24):
            cx0 = rng.randrange(4, SIDE - pw2 - 4)
            cy0 = rng.randrange(4, SIDE - ph2 - 4)
            if all(cx0 + pw2 < bx1 or cx0 > bx2 or cy0 + ph2 < by1 or cy0 > by2
                   for bx1, by1, bx2, by2 in keep):
                x0, y0 = cx0, cy0
                break
        if x0 is None:
            continue
        roi = canvas[y0:y0 + ph2, x0:x0 + pw2].astype(np.float32)
        canvas[y0:y0 + ph2, x0:x0 + pw2] = (
            roi * (1 - alpha) + patch.astype(np.float32) * alpha).astype(np.uint8)

        split = "val" if rng.random() < a.val else "train"
        stem = f"dot{made:05d}"
        cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tofile(
            os.path.join(DS, "images", split, stem + ".jpg"))
        with open(os.path.join(DS, "labels", split, stem + ".txt"), "w") as f:
            # 합성한 도트 날짜
            cx, cy = (x0 + pw2 / 2) / SIDE, (y0 + ph2 / 2) / SIDE
            f.write(f"0 {cx:.6f} {cy:.6f} {pw2/SIDE:.6f} {ph2/SIDE:.6f}\n")
            # 배경에 원래 있던 날짜도 함께 (라벨 없이 두면 억제 학습이 된다)
            for bx1, by1, bx2, by2 in keep:
                kcx, kcy = (bx1 + bx2) / 2 / SIDE, (by1 + by2) / 2 / SIDE
                kw, kh = (bx2 - bx1) / SIDE, (by2 - by1) / SIDE
                if 0 < kw < 1 and 0 < kh < 1 and 0 < kcx < 1 and 0 < kcy < 1:
                    f.write(f"0 {kcx:.6f} {kcy:.6f} {kw:.6f} {kh:.6f}\n")
        made += 1
        if made % 400 == 0:
            print(f"  {made}장", flush=True)

    # 기존 데이터셋도 함께 쓰도록 yaml 을 쓴다
    yaml = os.path.join(DS, "data.yaml")
    old = os.path.join(ROOT, "yolo_ds")
    with open(yaml, "w", encoding="utf-8") as f:
        f.write("# 도트 인쇄 합성 + 기존 약지도 데이터\n")
        f.write("# 배경은 라벨이 없는 이미지에서만 골랐다(평가 표본 오염 방지).\n")
        f.write(f"train:\n  - {os.path.join(DS, 'images', 'train')}\n")
        f.write(f"  - {os.path.join(old, 'images', 'train')}\n")
        f.write(f"val:\n  - {os.path.join(DS, 'images', 'val')}\n")
        f.write(f"  - {os.path.join(old, 'images', 'val')}\n")
        f.write("names:\n  0: date\n")

    # 표본 몇 장을 붙여서 눈으로 확인할 수 있게 남긴다
    samp = sorted(os.listdir(os.path.join(DS, "images", "train")))[:4]
    if samp:
        tiles = []
        for s in samp:
            im = cv2.imdecode(np.fromfile(
                os.path.join(DS, "images", "train", s), np.uint8), cv2.IMREAD_COLOR)
            lp = os.path.join(DS, "labels", "train", s.replace(".jpg", ".txt"))
            for li, ln in enumerate(open(lp)):
                v = ln.split()
                if len(v) < 5:
                    continue
                cx, cy, bw, bh = [float(x) for x in v[1:5]]
                p1 = (int((cx - bw / 2) * SIDE), int((cy - bh / 2) * SIDE))
                p2 = (int((cx + bw / 2) * SIDE), int((cy + bh / 2) * SIDE))
                col = (0, 0, 255) if li == 0 else (0, 200, 0)
                cv2.rectangle(im, p1, p2, col, 2)
            tiles.append(cv2.resize(im, (320, 320)))
        cv2.imencode(".jpg", np.hstack(tiles))[1].tofile(
            os.path.join(OUT, "113_yolo_synth_sample.jpg"))

    print(f"\n합성 {made}장 → {DS}")
    print(f"표본: eda/out/113_yolo_synth_sample.jpg")
    print(f"data.yaml: {yaml}")


if __name__ == "__main__":
    main()
