"""
EDA 111단계 — 도트 인쇄를 CTC 시퀀스 모델로 읽는다

왜 분할을 버리나 (EDA 110)
  글자 분류기 자체는 됐다. `2022.07` 에서 뒤 네 자리 `2207` 을 확신 1.00 으로
  맞혔다. 그런데 최종 정답은 0장이었다. 날짜를 맞히려면 여덟 글자가 **전부**
  정확히 잘려야 하는데, 하나만 누락되거나 붙으면 날짜 전체가 틀린다.
  `20.08.05` 를 `1.....2.03.612` 로 읽는 식이다.

  분할이 병목이면 분할을 없애야 한다. CTC 는 입력 폭을 따라 문자열을 통째로
  내보내므로 글자를 자를 필요가 없고, 정렬도 학습이 알아서 맞춘다.

합성
  109단계의 도트화(dotify)를 그대로 쓰되, 렌더링 단위를 **글자에서 날짜
  문자열로** 바꾼다. 실제 인쇄에서 보이는 모양을 그대로 흉내 낸다.
      26.08.13 / 2022.07 / 20.08.05 까지 / EXP 2021.12.18
  라벨은 숫자와 구분자만 남긴다. 한글·영문은 배경 잡음으로 넣되 라벨에서 뺀다.

추론에는 torch 를 싣지 않는다. ONNX 로 내보낸다(EDA 52~53).

    python eda/111_dot_crnn_train.py --n 40000 --epochs 14
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

H, W = 32, 192                      # 입력 규격 (높이 32, 폭 192)
CHARS = "0123456789./- "            # 라벨에 쓰는 글자
BLANK = 0                           # CTC blank 는 0번
VOCAB = [""] + list(CHARS)          # 0 = blank

FONTS = [r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\arial.ttf",
         r"C:\Windows\Fonts\consolab.ttf", r"C:\Windows\Fonts\consola.ttf",
         r"C:\Windows\Fonts\cour.ttf", r"C:\Windows\Fonts\courbd.ttf",
         r"C:\Windows\Fonts\verdanab.ttf", r"C:\Windows\Fonts\tahomabd.ttf"]
NOISE = ["까지", "부터", "제조일자", "유통기한", "소비기한", "EXP", "BB",
         "A1", "F2", "L3", "P", "B", "12:47", "09:26"]


def _fonts():
    return [p for p in FONTS if os.path.exists(p)] or [None]


def make_text(rng):
    """날짜 문자열 하나. 라벨은 숫자·구분자만, 화면에는 잡음도 함께 그린다."""
    y4 = rng.random() < 0.45
    yy = rng.randint(2018, 2030)
    y = str(yy) if y4 else f"{yy % 100:02d}"
    m = f"{rng.randint(1, 12):02d}"
    d = f"{rng.randint(1, 28):02d}"
    sep = rng.choice([".", ".", ".", "/", "-", " "])
    if rng.random() < 0.12:                    # 월·연만 (`2022.07`)
        core = f"{y}{sep}{m}"
    else:
        core = f"{y}{sep}{m}{sep}{d}"

    left = right = ""
    if rng.random() < 0.4:
        left = rng.choice(NOISE) + rng.choice([":", " ", ""])
    if rng.random() < 0.4:
        right = rng.choice([" ", ""]) + rng.choice(NOISE)
    return left + core + right, core


def dotify(np, cv2, img, rng):
    """109단계와 같은 방식. 점 격자로 샘플링해 도트 인쇄처럼 만든다."""
    h, w = img.shape
    pitch = rng.randint(3, 8)
    rad = max(1, int(pitch * rng.uniform(0.30, 0.52)))
    ox, oy = rng.randrange(pitch), rng.randrange(pitch)
    mask = np.zeros_like(img)
    for y in range(oy, h, pitch):
        for x in range(ox, w, pitch):
            cv2.circle(mask, (x, y), rad, 255, -1)
    out = cv2.bitwise_and(img, mask)
    if rng.random() < 0.5:
        k = rng.choice([2, 3])
        out = cv2.dilate(out, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return out


def render(np, cv2, Image, ImageDraw, ImageFont, rng):
    """한 표본. -> (32x192 uint8, 라벨문자열)"""
    shown, core = make_text(rng)
    size = rng.randint(28, 48)
    fp = rng.choice(_fonts())
    try:
        font = ImageFont.truetype(fp, size) if fp else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    pad = 20
    im = Image.new("L", (size * len(shown) + pad * 2, size * 2 + pad * 2), 0)
    ImageDraw.Draw(im).text((pad, pad), shown, fill=255, font=font)
    a = np.array(im)

    if rng.random() < 0.35:
        k = rng.choice([2, 3])
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        a = cv2.dilate(a, ker) if rng.random() < 0.5 else cv2.erode(a, ker)

    a = dotify(np, cv2, a, rng)

    if rng.random() < 0.6:
        ang = rng.uniform(-6, 6)
        M = cv2.getRotationMatrix2D((a.shape[1] / 2, a.shape[0] / 2), ang, 1.0)
        a = cv2.warpAffine(a, M, (a.shape[1], a.shape[0]))
    if rng.random() < 0.55:
        a = cv2.GaussianBlur(a, (0, 0), rng.uniform(0.5, 2.2))
    if rng.random() < 0.3:
        a = cv2.add(a, (np.random.rand(*a.shape) * 45).astype(np.uint8))

    ys, xs = np.where(a > 35)
    if len(ys) < 20:
        return None, None
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    _, a = cv2.threshold(a, 35, 255, cv2.THRESH_BINARY)

    # 높이 32 로 맞추고 폭 192 안에 넣는다
    h0, w0 = a.shape
    s = (H - 6) / max(h0, 1)
    nw = max(8, min(W - 6, int(w0 * s)))
    r = cv2.resize(a, (nw, H - 6), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((H, W), np.uint8)
    x0 = rng.randrange(0, max(1, W - nw))
    canvas[3:3 + (H - 6), x0:x0 + nw] = r
    return canvas, core


def build(np, cv2, n, seed):
    from PIL import Image, ImageDraw, ImageFont
    rng = random.Random(seed)
    X, T = [], []
    while len(X) < n:
        a, lab = render(np, cv2, Image, ImageDraw, ImageFont, rng)
        if a is None or not lab:
            continue
        X.append(a)
        T.append(lab)
    return np.stack(X), T


def encode(t):
    return [CHARS.index(c) + 1 for c in t if c in CHARS]


def decode(seq):
    """CTC 그리디 디코드: 반복 제거 후 blank 제거."""
    out, prev = [], -1
    for i in seq:
        if i != prev and i != BLANK:
            out.append(VOCAB[i])
        prev = i
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--epochs", type=int, default=14)
    a = ap.parse_args()

    import numpy as np
    import cv2
    import torch
    import torch.nn as nn

    os.makedirs(WDIR, exist_ok=True)
    print(f"합성 {a.n}장 생성 중...", flush=True)
    X, T = build(np, cv2, a.n, 1)
    Xv, Tv = build(np, cv2, 2500, 77)
    print(f"  학습 {len(X)} / 검증 {len(Xv)}", flush=True)
    grid = np.vstack([X[i] for i in range(10)])
    cv2.imencode(".png", grid)[1].tofile(os.path.join(OUT, "111_synth_sample.png"))

    class CRNN(nn.Module):
        """CNN 으로 폭 방향 특징을 뽑고 GRU 로 이어 CTC 로 낸다."""
        def __init__(self, nc):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.MaxPool2d(2),                       # 16 x 96
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.MaxPool2d(2),                       # 8 x 48
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.MaxPool2d((2, 1)),                  # 4 x 48
                nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.MaxPool2d((4, 1)),                  # 1 x 48
            )
            self.rnn = nn.GRU(128, 128, bidirectional=True, batch_first=True,
                              num_layers=2, dropout=0.1)
            self.fc = nn.Linear(256, nc)

        def forward(self, x):
            f = self.cnn(x).squeeze(2).permute(0, 2, 1)   # (N, 48, 128)
            o, _ = self.rnn(f)
            return self.fc(o)                            # (N, 48, nc)

    torch.manual_seed(0)
    net = CRNN(len(VOCAB))
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    ctc = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    steps = a.epochs * (len(X) // 128 + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps)

    Xt = torch.from_numpy(X).float().unsqueeze(1) / 255.0
    Xvt = torch.from_numpy(Xv).float().unsqueeze(1) / 255.0
    enc = [encode(t) for t in T]

    bs = 128
    for ep in range(a.epochs):
        net.train()
        perm = torch.randperm(len(Xt))
        tot = 0.0
        for i in range(0, len(Xt), bs):
            idx = perm[i:i + bs].tolist()
            xb = Xt[idx]
            tg = [enc[j] for j in idx]
            tl = torch.tensor([len(t) for t in tg], dtype=torch.long)
            flat = torch.tensor([c for t in tg for c in t], dtype=torch.long)
            opt.zero_grad()
            out = net(xb)                                  # (N, Tt, C)
            lp = out.log_softmax(2).permute(1, 0, 2)       # (Tt, N, C)
            il = torch.full((len(idx),), out.shape[1], dtype=torch.long)
            loss = ctc(lp, flat, il, tl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            sched.step()
            tot += float(loss) * len(idx)

        net.eval()
        with torch.no_grad():
            pv = net(Xvt[:1500]).argmax(2).cpu().numpy()
        hit = sum(1 for k in range(len(pv)) if decode(pv[k]) == Tv[k])
        print(f"  epoch {ep+1}/{a.epochs}  loss {tot/len(Xt):.4f}  "
              f"완전일치 {hit/len(pv)*100:.1f}%", flush=True)

    # 몇 개 예시를 보여 준다
    print("\n  예시 (정답 → 예측)")
    for k in range(6):
        print(f"    {Tv[k]:14s} → {decode(pv[k])}")

    onnx_path = os.path.join(WDIR, "dotcrnn_v1.onnx")
    net.eval()
    torch.onnx.export(net, torch.zeros(1, 1, H, W), onnx_path,
                      input_names=["x"], output_names=["logits"],
                      dynamic_axes={"x": {0: "n"}, "logits": {0: "n"}},
                      opset_version=17)
    mb = sum(os.path.getsize(os.path.join(WDIR, f))
             for f in os.listdir(WDIR) if f.startswith("dotcrnn")) / 1024 / 1024
    acc = hit / len(pv)
    print(f"\n→ {onnx_path}  ({mb:.1f}MB)")

    with open(os.path.join(OUT, "111_dot_crnn_train.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join([
            "# EDA 111 — 도트 인쇄를 CTC 시퀀스 모델로\n",
            "110단계에서 글자 분할이 병목임이 드러났다. 분류기는 숫자를 맞히는데",
            "(`2022.07` 의 `2207` 을 확신 1.00), 여덟 글자가 전부 정확히 잘리지",
            "않으면 날짜가 통째로 틀린다. 그래서 분할을 버리고 CTC 로 간다.",
            "",
            f"- 합성 학습 {len(X)}장 / 검증 {len(Xv)}장",
            f"- 입력 {H}x{W}, 어휘 {len(VOCAB)}자 (숫자·구분자 + blank)",
            f"- 검증 **완전일치 {acc*100:.1f}%**",
            f"- 모델 {mb:.1f}MB, ONNX",
            "",
            "⚠️ 합성 데이터에서의 수치다. 실제 사진은 112단계에서 확인한다.",
        ]))


if __name__ == "__main__":
    main()
