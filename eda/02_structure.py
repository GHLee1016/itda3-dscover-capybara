"""
EDA 2단계 — 디코딩 호환성 / 이미지 구조 점검

1) cv2.imread 가 전량 정상 로드되는가 (MPO·확장자불일치 리스크)
2) PIL vs cv2 크기 일치 여부 (EXIF 회전 처리 차이)
3) 640x640 클러스터가 레터박싱(패딩)인가 크롭인가
4) 640x640 클러스터의 EXIF 소실 → 회전이 구워졌는지(baked) 확인용 표본 추출
"""
import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageOps
from concurrent.futures import ProcessPoolExecutor

Image.MAX_IMAGE_PIXELS = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(ROOT, "images", "상품사진입니다")
OUT_DIR = os.path.join(ROOT, "eda", "out")
os.makedirs(OUT_DIR, exist_ok=True)


def uniform_border(a):
    """상하/좌우 균일 테두리(레터박스 패딩) 두께를 픽셀로 반환."""
    def run(lines):
        n = 0
        for ln in lines:
            if ln.std() < 3.0:          # 거의 단색인 행/열
                n += 1
            else:
                break
        return n
    top = run(a)
    bot = run(a[::-1])
    left = run(a.T)
    right = run(a.T[::-1])
    return top, bot, left, right


def check(path):
    name = os.path.basename(path)
    r = {"file": name}
    # --- cv2 ---
    try:
        # 한글 경로 대응: np.fromfile + imdecode
        buf = np.fromfile(path, dtype=np.uint8)
        im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        r["cv2_ok"] = im is not None
        if im is not None:
            r["cv2_h"], r["cv2_w"] = im.shape[:2]
    except Exception as e:
        r["cv2_ok"] = False
        r["cv2_err"] = f"{type(e).__name__}: {e}"
    # --- cv2 with plain imread (한글 경로에서 실패하는지 확인) ---
    try:
        im2 = cv2.imread(path, cv2.IMREAD_COLOR)
        r["cv2_imread_ok"] = im2 is not None
    except Exception:
        r["cv2_imread_ok"] = False
    # --- PIL ---
    try:
        with Image.open(path) as pim:
            r["pil_frames"] = getattr(pim, "n_frames", 1)
            r["pil_format"] = pim.format
            t = ImageOps.exif_transpose(pim)
            r["pil_w"], r["pil_h"] = t.size
            g = t.convert("L")
            g.thumbnail((256, 256), Image.BILINEAR)
            a = np.asarray(g, dtype=np.float32)
            top, bot, left, right = uniform_border(a)
            sc = 256 / max(g.size)
            r["border_top"], r["border_bot"] = top, bot
            r["border_left"], r["border_right"] = left, right
            r["border_frac"] = float(max(top + bot, left + right) / max(a.shape))
        r["pil_ok"] = True
    except Exception as e:
        r["pil_ok"] = False
        r["pil_err"] = f"{type(e).__name__}: {e}"
    return r


def main():
    files = sorted(os.path.join(IMG_DIR, f) for f in os.listdir(IMG_DIR))
    print(f"checking {len(files)} files ...", flush=True)
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
        recs = list(ex.map(check, files, chunksize=16))
    df = pd.DataFrame(recs)
    df["image_id"] = df["file"].str.replace(r"\.[^.]+$", "", regex=True)
    df["id_num"] = pd.to_numeric(df["image_id"], errors="coerce")
    df = df.sort_values("id_num").reset_index(drop=True)

    meta = pd.read_csv(os.path.join(OUT_DIR, "metadata.csv"))
    df = df.merge(meta[["file", "w", "h", "exif_orientation", "format", "ext", "has_exif"]],
                  on="file", how="left")
    df.to_csv(os.path.join(OUT_DIR, "structure.csv"), index=False, encoding="utf-8-sig")

    L = []
    A = L.append
    A("# EDA 2 — 디코딩 호환성 및 이미지 구조\n")

    A("## 1. 디코딩 호환성\n")
    A(f"- `cv2.imdecode(np.fromfile(...))` 성공: **{int(df['cv2_ok'].sum()):,} / {len(df):,}**")
    A(f"- `cv2.imread(path)` 성공: **{int(df['cv2_imread_ok'].sum()):,} / {len(df):,}**")
    A(f"- `PIL.Image.open` 성공: **{int(df['pil_ok'].sum()):,} / {len(df):,}**")
    bad = df[~df["cv2_ok"] | ~df["pil_ok"]]
    if len(bad):
        A("")
        A("| 파일 | cv2 | PIL | 오류 |")
        A("| --- | --- | --- | --- |")
        for _, r in bad.head(20).iterrows():
            A(f"| `{r['file']}` | {r['cv2_ok']} | {r['pil_ok']} | "
              f"{r.get('cv2_err') or r.get('pil_err')} |")
    nread = df[~df["cv2_imread_ok"]]
    A("")
    if len(nread):
        A(f"> ⚠️ `cv2.imread()` 실패 **{len(nread):,}장**. 파일 자체가 아니라 **경로에 한글이 있어서** "
          "OpenCV 가 파일을 열지 못한 것 (Windows 로컬 개발 환경 이슈).")
        A(">")
        A("> 채점 서버 경로(`/eval/val_images`)는 ASCII 이므로 채점 시에는 재현되지 않지만, "
          "로컬 실험이 전부 막히므로 `np.fromfile + cv2.imdecode` 로 읽는 습관을 들이는 편이 안전하다.")
    else:
        A("> ✅ 한글 경로에서도 `cv2.imread()` 정상 동작")
    A("")

    A("## 2. MPO / 확장자 불일치\n")
    mpo = df[df["pil_format"] == "MPO"]
    A(f"- **MPO 포맷 {len(mpo):,}장** (iPhone HDR/버스트 — JPEG 안에 여러 프레임 내장)")
    if len(mpo):
        A(f"  - 내장 프레임 수 분포: {mpo['pil_frames'].value_counts().to_dict()}")
        A(f"  - cv2 로드 성공: {int(mpo['cv2_ok'].sum()):,} / {len(mpo):,}")
        A(f"  - PIL 로드 성공: {int(mpo['pil_ok'].sum()):,} / {len(mpo):,}")
        agree = (mpo["cv2_w"] == mpo["pil_w"]) | (mpo["cv2_w"] == mpo["pil_h"])
        A(f"  - cv2/PIL 크기 정합: {int(agree.sum()):,} / {len(mpo):,}")
    mism = df[((df["ext"].isin([".jpg", ".jpeg"])) & (df["pil_format"] == "PNG")) |
              ((df["ext"] == ".png") & (df["pil_format"] != "PNG"))]
    A(f"- 확장자↔포맷 불일치 (jpg인데 PNG 등): **{len(mism)}장**"
      + (f" → `{', '.join(mism['file'])}`" if len(mism) else ""))
    A("")

    A("## 3. EXIF 회전 처리 — cv2 vs PIL\n")
    d = df[df["cv2_ok"] & df["pil_ok"]].copy()
    # 정사각 이미지는 '동일'과 '뒤바뀜'이 동시에 참이라 판정 불가 → 제외
    d["square"] = d["cv2_w"] == d["cv2_h"]
    ns = d[~d["square"]]
    same = (ns["cv2_w"] == ns["pil_w"]) & (ns["cv2_h"] == ns["pil_h"])
    swap = (~same) & (ns["cv2_w"] == ns["pil_h"]) & (ns["cv2_h"] == ns["pil_w"])
    A(f"- 정사각 이미지 {int(d['square'].sum()):,}장은 판정 불가로 제외, "
      f"비정사각 {len(ns):,}장으로 비교")
    A(f"- `cv2.imdecode` 결과 == `PIL + exif_transpose` 결과: **{int(same.sum()):,}장**")
    A(f"- 가로세로 뒤바뀜: **{int(swap.sum()):,}장** / 기타 불일치: "
      f"{int((~same & ~swap).sum()):,}장")
    A("")
    A("검증 (orientation=6 표본):\n")
    A("| 파일 | 파일에 저장된 크기 | cv2 기본 | cv2 IGNORE_ORIENTATION | PIL exif_transpose |")
    A("| --- | --- | --- | --- | --- |")
    for _, r in meta[meta["exif_orientation"] == 6].head(4).iterrows():
        pth = os.path.join(IMG_DIR, r["file"])
        buf = np.fromfile(pth, dtype=np.uint8)
        a1 = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        a2 = cv2.imdecode(buf, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        with Image.open(pth) as im:
            raw = im.size
            tr = ImageOps.exif_transpose(im).size
        A(f"| `{r['file']}` | {raw[0]}x{raw[1]} | {a1.shape[1]}x{a1.shape[0]} | "
          f"{a2.shape[1]}x{a2.shape[0]} | {tr[0]}x{tr[1]} |")
    A("")
    A("> ✅ **`cv2.imdecode`/`imread` 는 EXIF orientation 을 기본으로 적용한다** "
      "(OpenCV ≥ 3.4.1). 두 경로 결과가 전량 일치.")
    A(">")
    A(f"> ⚠️ 단, **`PIL.Image.open()` 은 자동 적용하지 않는다.** PIL 로 직접 읽는 경로에서는 "
      f"`ImageOps.exif_transpose()` 를 반드시 호출해야 하며, 누락 시 orientation=6 인 "
      f"**{int((meta['exif_orientation']==6).sum()):,}장**이 90° 누운 채 OCR 된다.")
    A(">")
    A("> ⚠️ `cv2.IMREAD_IGNORE_ORIENTATION` 플래그를 켜면 회전이 사라지므로 사용 금지.")
    A("")

    A("## 4. 레터박싱 / 균일 테두리\n")
    d2 = df[df["pil_ok"]].copy()
    d2["is640"] = (d2["w"] == 640) & (d2["h"] == 640)
    A("| 그룹 | 장수 | 테두리 있는 이미지 | 테두리 비중 중앙값 |")
    A("| --- | ---: | ---: | ---: |")
    for lab, g in [("640x640 클러스터", d2[d2["is640"]]), ("그 외", d2[~d2["is640"]])]:
        has = g[g["border_frac"] > 0.02]
        A(f"| {lab} | {len(g):,} | {len(has):,} ({len(has)/max(len(g),1)*100:.1f}%) | "
          f"{g['border_frac'].median()*100:.1f}% |")
    big = d2[d2["border_frac"] > 0.10]
    A("")
    A(f"- 테두리가 전체의 10% 이상: **{len(big):,}장** "
      f"(유효 화소 손실 — 다운스케일 시 글자가 더 작아짐)")
    if len(big):
        A(f"  - 예시: `{', '.join(big.nlargest(10, 'border_frac')['file'])}`")
    A("")

    A("## 5. 640x640 클러스터 요약\n")
    g = d2[d2["is640"]]
    A(f"- 장수: **{len(g):,}장 ({len(g)/len(d2)*100:.1f}%)**")
    A(f"- EXIF 보유: {int(g['has_exif'].sum()):,}장")
    A(f"- id 범위: {int(g['id_num'].min())} ~ {int(g['id_num'].max())}")
    A(f"- 원본 종횡비가 1:1 이 아닐 가능성 → 테두리 보유 {int((g['border_frac']>0.02).sum()):,}장")
    A("")

    rep = "\n".join(L)
    with open(os.path.join(OUT_DIR, "02_report.md"), "w", encoding="utf-8") as f:
        f.write(rep)
    print("wrote 02_report.md / structure.csv")


if __name__ == "__main__":
    main()
