"""
EDA 1단계 — 이미지 메타데이터 전수 조사

3,352장 전체에 대해 파일/해상도/EXIF/품질 지표를 수집하고
metadata.csv + 01_report.md 를 생성한다.
"""
import os
import sys
import io
import json
from concurrent.futures import ProcessPoolExecutor
from collections import Counter

import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageOps, ExifTags
import imagehash

Image.MAX_IMAGE_PIXELS = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(ROOT, "images", "상품사진입니다")
OUT_DIR = os.path.join(ROOT, "eda", "out")
os.makedirs(OUT_DIR, exist_ok=True)

EXIF_TAGS = {v: k for k, v in ExifTags.TAGS.items()}


def probe(path):
    """이미지 1장의 메타데이터를 뽑는다."""
    name = os.path.basename(path)
    stem, ext = os.path.splitext(name)
    rec = {
        "file": name,
        "image_id": stem,          # 확장자 제외 파일명 = 제출 규격의 image_id
        "ext": ext.lower(),
        "bytes": os.path.getsize(path),
    }
    try:
        with Image.open(path) as im:
            rec["format"] = im.format
            rec["mode"] = im.mode
            w, h = im.size
            rec["w_raw"], rec["h_raw"] = w, h

            # --- EXIF ---
            ori, dt, make, model, software = 1, None, None, None, None
            try:
                ex = im.getexif()
                if ex:
                    ori = int(ex.get(274, 1) or 1)
                    make = ex.get(271)
                    model = ex.get(272)
                    software = ex.get(305)
                    dt = ex.get(306)  # DateTime
                    sub = ex.get_ifd(0x8769)
                    if sub:
                        dt = sub.get(36867) or dt   # DateTimeOriginal 우선
            except Exception:
                pass
            rec["exif_orientation"] = ori
            rec["exif_datetime"] = str(dt) if dt else None
            rec["exif_make"] = str(make).strip() if make else None
            rec["exif_model"] = str(model).strip() if model else None
            rec["exif_software"] = str(software).strip() if software else None
            rec["has_exif"] = bool(make or model or dt)

            # EXIF 회전 적용 후 실제 표시 크기
            if ori in (5, 6, 7, 8):
                rec["w"], rec["h"] = h, w
            else:
                rec["w"], rec["h"] = w, h

            # --- 축소 로드 후 품질 지표 ---
            im.draft("L", (768, 768))          # JPEG 고속 축소 디코딩
            g = ImageOps.exif_transpose(im).convert("L")
            g.thumbnail((512, 512), Image.BILINEAR)
            a = np.asarray(g, dtype=np.uint8)

            rec["sharpness"] = float(cv2.Laplacian(a, cv2.CV_64F).var())
            rec["brightness"] = float(a.mean())
            rec["contrast"] = float(a.std())
            # 어두운/밝은 극단 픽셀 비율 (노출 문제 탐지)
            rec["dark_frac"] = float((a < 25).mean())
            rec["blown_frac"] = float((a > 245).mean())

            rec["phash"] = str(imagehash.phash(Image.fromarray(a)))
            rec["ok"] = True
    except Exception as e:
        rec["ok"] = False
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


_CTRL = {c: None for c in range(32)}
_CTRL.pop(9, None)


def clean(v):
    """EXIF 문자열의 제어문자/NUL 제거 (CSV 저장 시 깨짐 방지)."""
    if not isinstance(v, str):
        return v
    return v.translate(_CTRL).replace('"', "'").replace("\\", "/").strip() or None


def main():
    files = sorted(
        os.path.join(IMG_DIR, f) for f in os.listdir(IMG_DIR)
        if not f.startswith(".")
    )
    cache = os.path.join(OUT_DIR, "_probe_cache.pkl")
    if os.path.exists(cache):
        print("loading probe cache ...", flush=True)
        recs = pd.read_pickle(cache)
    else:
        print(f"scanning {len(files)} files ...", flush=True)
        with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
            recs = list(ex.map(probe, files, chunksize=16))
        pd.to_pickle(recs, cache)

    df = pd.DataFrame(recs)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(clean)
    df["mp"] = df["w"] * df["h"] / 1e6
    df["aspect"] = df["w"] / df["h"]
    df["id_num"] = pd.to_numeric(df["image_id"], errors="coerce")
    df["id_padded"] = df["image_id"].str.fullmatch(r"\d{6}")
    df = df.sort_values("id_num").reset_index(drop=True)
    df.to_csv(os.path.join(OUT_DIR, "metadata.csv"), index=False, encoding="utf-8-sig")

    # ---------------- 리포트 ----------------
    L = []
    A = L.append
    n = len(df)
    ok = df[df["ok"]]

    A("# EDA 1 — 이미지 메타데이터 전수 조사\n")
    A(f"- 대상: `images/상품사진입니다/` **{n:,}장**")
    A(f"- 디코딩 성공: {len(ok):,}장 / 실패: {n - len(ok):,}장")
    if (~df["ok"]).any():
        for _, r in df[~df["ok"]].iterrows():
            A(f"  - ❌ `{r['file']}` — {r.get('error')}")
    A("")

    A("## 1. 파일명 / 포맷\n")
    A(f"- `image_id` 6자리 zero-padding: {int(df['id_padded'].sum()):,}장 / "
      f"비패딩: {int((~df['id_padded']).sum()):,}장")
    npad = df.loc[~df["id_padded"], "image_id"].tolist()
    if npad:
        A(f"  - 비패딩 목록: `{', '.join(npad)}`")
    A(f"- id 범위: {int(df['id_num'].min())} ~ {int(df['id_num'].max())}, "
      f"고유 id {df['image_id'].nunique():,}개")
    missing = sorted(set(range(1, int(df["id_num"].max()) + 1)) - set(df["id_num"].dropna().astype(int)))
    A(f"- 결번: {len(missing)}개" + (f" → `{missing[:20]}`{' ...' if len(missing) > 20 else ''}" if missing else ""))
    A("")
    A("| 확장자 | 장수 | 실제 포맷 |")
    A("| --- | ---: | --- |")
    for e, cnt in df["ext"].value_counts().items():
        fmts = df.loc[df["ext"] == e, "format"].value_counts().to_dict()
        A(f"| `{e}` | {cnt:,} | {fmts} |")
    A("")
    mism = ok[((ok["ext"].isin([".jpg", ".jpeg"])) & (ok["format"] != "JPEG")) |
              ((ok["ext"] == ".png") & (ok["format"] != "PNG"))]
    A(f"- 확장자↔실제포맷 불일치: **{len(mism)}장**"
      + (f" → `{', '.join(mism['file'].head(10))}`" if len(mism) else ""))
    A(f"- 컬러 모드: {ok['mode'].value_counts().to_dict()}")
    A("")

    A("## 2. 해상도\n")
    A("| 지표 | 값 |")
    A("| --- | --- |")
    A(f"| 최소 / 최대 megapixel | {ok['mp'].min():.2f} MP / {ok['mp'].max():.2f} MP "
      f"({ok['mp'].max()/max(ok['mp'].min(),1e-9):.0f}배 차이) |")
    A(f"| megapixel 중앙값 | {ok['mp'].median():.2f} MP |")
    A(f"| 긴 변 min / median / max | {int(ok[['w','h']].max(axis=1).min())} / "
      f"{int(ok[['w','h']].max(axis=1).median())} / {int(ok[['w','h']].max(axis=1).max())} px |")
    A(f"| 파일 크기 median / max | {ok['bytes'].median()/1e6:.2f} MB / {ok['bytes'].max()/1e6:.2f} MB |")
    A(f"| 총 용량 | {ok['bytes'].sum()/1e9:.2f} GB |")
    A("")
    A("### 해상도 상위 15종 (EXIF 회전 적용 후)\n")
    A("| WxH | 장수 | 비율 |")
    A("| --- | ---: | ---: |")
    dims = (ok["w"].astype(str) + "x" + ok["h"].astype(str)).value_counts()
    for d, c in dims.head(15).items():
        A(f"| {d} | {c:,} | {c/len(ok)*100:.1f}% |")
    A(f"\n- 고유 해상도 종류: **{dims.nunique() if hasattr(dims,'nunique') else len(dims)}종**")
    A("")
    bins = [0, 0.3, 0.5, 1, 2, 5, 9, 13, 1e9]
    labels = ["<0.3MP", "0.3-0.5MP", "0.5-1MP", "1-2MP", "2-5MP", "5-9MP", "9-13MP", ">13MP"]
    A("### megapixel 분포\n")
    A("| 구간 | 장수 | 비율 |")
    A("| --- | ---: | ---: |")
    for lab, c in pd.cut(ok["mp"], bins, labels=labels).value_counts().sort_index().items():
        A(f"| {lab} | {c:,} | {c/len(ok)*100:.1f}% |")
    A("")
    shape = pd.cut(ok["aspect"], [0, 0.95, 1.05, 1e9], labels=["세로(portrait)", "정사각", "가로(landscape)"])
    A(f"- 방향: {shape.value_counts().to_dict()}")
    A("")

    A("## 3. EXIF\n")
    A(f"- EXIF 보유: **{int(ok['has_exif'].sum()):,}장 ({ok['has_exif'].mean()*100:.1f}%)** "
      f"/ 없음: {int((~ok['has_exif']).sum()):,}장")
    A("")
    A("### Orientation 태그\n")
    A("| orientation | 의미 | 장수 | 비율 |")
    A("| ---: | --- | ---: | ---: |")
    omean = {1: "정상", 2: "좌우반전", 3: "180° 회전", 4: "상하반전",
             5: "전치", 6: "시계 90° 회전 필요", 7: "역전치", 8: "반시계 90° 회전 필요"}
    for o, c in ok["exif_orientation"].value_counts().sort_index().items():
        A(f"| {o} | {omean.get(int(o), '?')} | {c:,} | {c/len(ok)*100:.1f}% |")
    rot = int(ok["exif_orientation"].isin([3, 5, 6, 7, 8]).sum())
    A(f"\n> ⚠️ **회전 보정이 필요한 이미지 {rot:,}장 ({rot/len(ok)*100:.1f}%)** — "
      f"`ImageOps.exif_transpose()` 미적용 시 그대로 오독됨")
    A("")
    A("### 촬영 기기 (EXIF Model) 상위 12\n")
    A("| 기기 | 장수 |")
    A("| --- | ---: |")
    for m, c in ok["exif_model"].value_counts().head(12).items():
        A(f"| {m} | {c:,} |")
    sw = ok["exif_software"].value_counts()
    if len(sw):
        A("")
        A(f"- Software 태그 상위: {sw.head(6).to_dict()}")
    A("")

    A("### 촬영 일시 (EXIF DateTimeOriginal)\n")
    dt = pd.to_datetime(ok["exif_datetime"], format="%Y:%m:%d %H:%M:%S", errors="coerce")
    ok = ok.assign(shot=dt)
    have = ok["shot"].notna()
    A(f"- 촬영일시 보유: **{int(have.sum()):,}장 ({have.mean()*100:.1f}%)**")
    if have.any():
        A(f"- 촬영 기간: **{ok.loc[have,'shot'].min():%Y-%m-%d} ~ {ok.loc[have,'shot'].max():%Y-%m-%d}**")
        A("")
        A("| 촬영 연월 | 장수 |")
        A("| --- | ---: |")
        for ym, c in ok.loc[have, "shot"].dt.to_period("M").value_counts().sort_index().items():
            A(f"| {ym} | {c:,} |")
        A("")
        A("| 촬영 시간대 | 장수 |")
        A("| --- | ---: |")
        for hh, c in ok.loc[have, "shot"].dt.hour.value_counts().sort_index().items():
            A(f"| {hh:02d}시 | {c:,} |")
    A("")

    A("## 4. 이미지 품질\n")
    A("| 지표 | min | p10 | median | p90 | max |")
    A("| --- | ---: | ---: | ---: | ---: | ---: |")
    for col, lab in [("sharpness", "선명도(Laplacian var)"), ("brightness", "밝기(0-255)"),
                     ("contrast", "대비(std)"), ("dark_frac", "암부 비율"), ("blown_frac", "포화 비율")]:
        s = ok[col]
        A(f"| {lab} | {s.min():.2f} | {s.quantile(.10):.2f} | {s.median():.2f} | "
          f"{s.quantile(.90):.2f} | {s.max():.2f} |")
    A("")
    blur = ok[ok["sharpness"] < 100]
    dark = ok[ok["brightness"] < 60]
    bright = ok[ok["brightness"] > 200]
    lowc = ok[ok["contrast"] < 30]
    A(f"- 흐림 의심 (sharpness < 100): **{len(blur):,}장 ({len(blur)/len(ok)*100:.1f}%)**")
    A(f"- 어두움 의심 (brightness < 60): **{len(dark):,}장 ({len(dark)/len(ok)*100:.1f}%)**")
    A(f"- 과노출 의심 (brightness > 200): **{len(bright):,}장 ({len(bright)/len(ok)*100:.1f}%)**")
    A(f"- 저대비 의심 (contrast < 30): **{len(lowc):,}장 ({len(lowc)/len(ok)*100:.1f}%)**")
    A("")
    A(f"- 가장 흐린 10장: `{', '.join(ok.nsmallest(10,'sharpness')['file'])}`")
    A(f"- 가장 어두운 10장: `{', '.join(ok.nsmallest(10,'brightness')['file'])}`")
    A("")

    A("## 5. 중복 (perceptual hash)\n")
    dup = ok.groupby("phash").filter(lambda g: len(g) > 1)
    groups = dup.groupby("phash")
    A(f"- 완전 동일 phash 그룹: **{groups.ngroups}개**, 관련 이미지 **{len(dup):,}장** "
      f"(중복 제거 시 {len(dup) - groups.ngroups:,}장 감소)")
    if groups.ngroups:
        A("")
        A("| phash | 장수 | 파일 |")
        A("| --- | ---: | --- |")
        for h, g in sorted(groups, key=lambda kv: -len(kv[1]))[:15]:
            A(f"| `{h}` | {len(g)} | {', '.join(g['file'].tolist()[:8])} |")
    A("")

    A("## 6. id 구간별 성격 (출처 클러스터 추정)\n")
    ok2 = ok.copy()
    ok2["bucket"] = (ok2["id_num"] // 250 * 250).astype("Int64")
    A("| id 구간 | 장수 | 중앙 MP | EXIF 보유율 | 회전필요 | 최빈 해상도 |")
    A("| --- | ---: | ---: | ---: | ---: | --- |")
    for b, g in ok2.groupby("bucket"):
        top = (g["w"].astype(str) + "x" + g["h"].astype(str)).value_counts()
        A(f"| {int(b)}–{int(b)+249} | {len(g):,} | {g['mp'].median():.2f} | "
          f"{g['has_exif'].mean()*100:.0f}% | "
          f"{g['exif_orientation'].isin([3,5,6,7,8]).mean()*100:.0f}% | "
          f"{top.index[0]} ({top.iloc[0]}) |")
    A("")

    rep = "\n".join(L)
    with open(os.path.join(OUT_DIR, "01_report.md"), "w", encoding="utf-8") as f:
        f.write(rep)
    print(f"\nwrote {OUT_DIR}\\metadata.csv  ({len(df)} rows)")
    print(f"wrote {OUT_DIR}\\01_report.md")


if __name__ == "__main__":
    main()
