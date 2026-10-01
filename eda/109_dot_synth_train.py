"""
EDA 109단계 — 도트 숫자 분류기를 합성 데이터로 학습한다

왜 만드나 (EDA 107)
  남은 미검출의 주된 원인이 도트 매트릭스 인쇄다. 인식기가 그 글꼴을 모른다.
  `26.08.13` 을 `2:88:13` 으로 읽는다. 이진화·모폴로지·색분리·확대를 다 해도
  같다. 분리의 문제가 아니라 **학습한 적 없는 글꼴**이라서다.

학습 데이터를 어떻게 구하나
  도트 인쇄 크롭은 손에 몇십 장뿐이라 그것만으로는 학습이 안 된다.
  그런데 도트 인쇄는 **일반 글자를 점 격자로 샘플링한 것**과 같다.
  그래서 일반 폰트로 숫자를 그린 뒤 점 마스크를 씌우면 얼마든지 만들 수 있다.
  실제 사진에서 오는 열화(블러·기울기·굵기 변화·잘림)도 같이 흉내 낸다.

클래스
  0~9 와 **기타**. 기타가 있어야 한글·구분자·얼룩을 걸러낸다.

추론에는 torch 를 싣지 않는다. 학습만 torch 로 하고 ONNX 로 내보낸다
(torch 를 임포트만 해도 파이프라인이 2.4배 느려진다 — EDA 52~53).

    python eda/109_dot_synth_train.py --n 24000 --epochs 8
"""
import os
import sys
import random
import argparse
import warnings

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "eda", "out")
WDIR = os.path.join(ROOT, "weights", "dot")
SIDE = 48
CLASSES = [str(i) for i in range(10)] + ["기타"]

FONTS = [r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\arial.ttf",
         r"C:\Windows\Fonts\consolab.ttf", r"C:\Windows\Fonts\consola.ttf",
         r"C:\Windows\Fonts\cour.ttf", r"C:\Windows\Fonts\courbd.ttf",
         r"C:\Windows\Fonts\verdanab.ttf", r"C:\Windows\Fonts\tahomabd.ttf"]
# 기타 클래스에 넣을 것들. 실제로 크롭에 섞여 나오는 것들이다.
OTHER = list(".:/-년월일까지제조유통기한ABCDEFGHLMNPRSTVXYZ")


def _fonts():
    from PIL import ImageFont
    out = []
    for p in FONTS:
        if os.path.exists(p):
            out.append(p)
    if not out:                       # 폰트를 못 찾으면 기본으로
        out = [None]
    return out


def dotify(np, cv2, img, rng):
    """
    글자 이미지에 점 격자를 씌워 도트 인쇄처럼 만든다.

    실제 도트 프린터는 일정 간격 격자 위에만 잉크를 찍는다. 그래서 획이
    점선으로 끊긴다. 격자 간격·점 크기·격자 위상을 매번 흔들어 다양하게 만든다.
    """
    h, w = img.shape
    pitch = rng.randint(4, 9)                  # 격자 간격
    rad = max(1, int(pitch * rng.uniform(0.28, 0.5)))
    ox, oy = rng.randrange(pitch), rng.randrange(pitch)
    mask = np.zeros_like(img)
    for y in range(oy, h, pitch):
        for x in range(ox, w, pitch):
            cv2.circle(mask, (x, y), rad, 255, -1)
    out = cv2.bitwise_and(img, mask)
    # 점이 번지거나 뭉개지는 것도 흉내 낸다
    if rng.random() < 0.5:
        k = rng.choice([2, 3])
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        out = cv2.dilate(out, ker)
    return out


def render(np, cv2, Image, ImageDraw, ImageFont, ch, font_path, rng):
    """한 글자를 도트화해 48x48 로 만든다."""
    size = rng.randint(40, 72)
    try:
        font = (ImageFont.truetype(font_path, size) if font_path
                else ImageFont.load_default())
    except Exception:
        font = ImageFont.load_default()
    pad = 24
    im = Image.new("L", (size * 2 + pad, size * 2 + pad), 0)
    d = ImageDraw.Draw(im)
    d.text((pad // 2, pad // 2), ch, fill=255, font=font)
    a = np.array(im)

    if rng.random() < 0.35:                     # 획 굵기 변화
        k = rng.choice([2, 3])
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        a = cv2.dilate(a, ker) if rng.random() < 0.5 else cv2.erode(a, ker)

    a = dotify(np, cv2, a, rng)

    if rng.random() < 0.6:                      # 기울기 (인쇄가 비스듬한 경우)
        ang = rng.uniform(-8, 8)
        M = cv2.getRotationMatrix2D((a.shape[1] / 2, a.shape[0] / 2), ang, 1.0)
        a = cv2.warpAffine(a, M, (a.shape[1], a.shape[0]))
    if rng.random() < 0.5:                      # 초점 흐림
        a = cv2.GaussianBlur(a, (0, 0), rng.uniform(0.6, 2.0))
    if rng.random() < 0.3:                      # 잡티
        n = (np.random.rand(*a.shape) * 40).astype(np.uint8)
        a = cv2.add(a, n)

    ys, xs = np.where(a > 40)
    if len(ys) < 6:
        return None
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    _, a = cv2.threshold(a, 40, 255, cv2.THRESH_BINARY)

    # dotseg._fit48 과 같은 규격으로 맞춘다
    h, w = a.shape
    s = (SIDE - 8) / max(h, w, 1)
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    r = cv2.resize(a, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((SIDE, SIDE), np.uint8)
    canvas[(SIDE - nh) // 2:(SIDE - nh) // 2 + nh,
           (SIDE - nw) // 2:(SIDE - nw) // 2 + nw] = r
    return canvas


def build(np, cv2, n, seed=0):
    from PIL import Image, ImageDraw, ImageFont
    rng = random.Random(seed)
    fonts = _fonts()
    X, Y = [], []
    # 숫자 9 : 기타 1. 실제 크롭은 숫자가 대부분이고, 기타를 키우면 숫자가
    # 기타로 흘러간다(EDA 110에서 0 이 기타로 빠지는 것을 봤다).
    while len(X) < n:
        if rng.random() < 0.9:
            # 0·6·8·9 는 가운데가 뚫려 있어 도트화하면 특히 깨진다.
            # 실제로 0 이 기타로 새는 것을 봤으므로 그쪽을 더 뽑는다.
            lab = rng.choice([0, 0, 0, 1, 2, 3, 4, 5, 6, 6, 7, 8, 8, 9, 9])
            ch = str(lab)
        else:
            lab = 10
            ch = rng.choice(OTHER)
        a = render(np, cv2, Image, ImageDraw, ImageFont, ch,
                   rng.choice(fonts), rng)
        if a is None:
            continue
        X.append(a)
        Y.append(lab)
    return np.stack(X), np.array(Y, np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24000)
    ap.add_argument("--epochs", type=int, default=8)
    a = ap.parse_args()

    import numpy as np
    import cv2
    import torch
    import torch.nn as nn

    os.makedirs(WDIR, exist_ok=True)
    print(f"합성 {a.n}장 생성 중...", flush=True)
    X, Y = build(np, cv2, a.n, seed=1)
    Xv, Yv = build(np, cv2, max(2000, a.n // 8), seed=99)
    print(f"  학습 {len(X)} / 검증 {len(Xv)}", flush=True)

    # 눈으로 볼 수 있게 표본을 남긴다
    grid = np.hstack([X[i] for i in range(24)])
    cv2.imencode(".png", grid)[1].tofile(os.path.join(OUT, "109_synth_sample.png"))

    def tt(x, y):
        return (torch.from_numpy(x).float().unsqueeze(1) / 255.0,
                torch.from_numpy(y))
    Xt, Yt = tt(X, Y)
    Xvt, Yvt = tt(Xv, Yv)

    torch.manual_seed(0)
    net = nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.MaxPool2d(2),                                   # 24
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.MaxPool2d(2),                                   # 12
        nn.Conv2d(64, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Dropout(0.2), nn.Linear(96, len(CLASSES)))
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=3e-3, total_steps=a.epochs * (len(Xt) // 256 + 1))

    bs = 256
    for ep in range(a.epochs):
        net.train()
        perm = torch.randperm(len(Xt))
        tot = 0.0
        for i in range(0, len(Xt), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            out = net(Xt[idx])
            loss = lossf(out, Yt[idx])
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss) * len(idx)
        net.eval()
        with torch.no_grad():
            pv = net(Xvt).argmax(1)
            acc = float((pv == Yvt).float().mean())
            dig = Yvt < 10
            dacc = float((pv[dig] == Yvt[dig]).float().mean())
        print(f"  epoch {ep+1}/{a.epochs}  loss {tot/len(Xt):.4f}  "
              f"검증 {acc*100:.1f}%  (숫자만 {dacc*100:.1f}%)", flush=True)

    onnx_path = os.path.join(WDIR, "dotdigit_v1.onnx")
    net.eval()
    torch.onnx.export(net, torch.zeros(1, 1, SIDE, SIDE), onnx_path,
                      input_names=["x"], output_names=["logits"],
                      dynamic_axes={"x": {0: "n"}, "logits": {0: "n"}},
                      opset_version=12)
    mb = os.path.getsize(onnx_path) / 1024 / 1024
    print(f"\n→ {onnx_path}  ({mb:.1f}MB)")
    print(f"→ 합성 표본: eda/out/109_synth_sample.png")

    with open(os.path.join(OUT, "109_dot_synth_train.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join([
            "# EDA 109 — 도트 숫자 분류기 (합성 학습)\n",
            "도트 인쇄는 일반 글자를 점 격자로 샘플링한 것과 같다. 그래서 일반",
            "폰트로 숫자를 그린 뒤 점 마스크를 씌워 학습 데이터를 만들었다.",
            "",
            f"- 합성 학습 {len(X)}장 / 검증 {len(Xv)}장",
            f"- 클래스: 0~9 + 기타 (한글·구분자·얼룩을 걸러내려고 둔다)",
            f"- 검증 정확도 **{acc*100:.1f}%** (숫자만 {dacc*100:.1f}%)",
            f"- 모델 {mb:.1f}MB, ONNX 로 내보내 추론에 torch 를 싣지 않는다",
            "",
            "⚠️ 이 수치는 **합성 데이터에서의 정확도**다. 실제 사진에서도",
            "그만큼 나오는지는 별개이며, 110단계에서 실제 크롭으로 확인한다.",
        ]))


if __name__ == "__main__":
    main()
