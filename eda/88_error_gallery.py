"""
EDA 88단계 — 라벨이 있는 580장 전체를 돌려 오답만 사진과 함께 모은다

배포된 3,352장에는 정답이 없다. 정답이 있는 것은 우리가 만든 580장뿐이다.
    팀 라벨 383장   custom_data/labels.csv  (tuning 137 / holdout1 48 / holdout2 198)
    내 라벨 197장   eda/out/truth_holdout3.json
둘은 겹치지 않는다.

여기서 하는 일
  · 580장을 현재 최종 구성으로 돌린다
  · 틀린 것만 뽑아 **사진 · 정답 · 예측 · 틀린 필드**를 한 페이지에 모은다
  · 사람이 눈으로 훑으면서 원인을 찾을 수 있게 유형별로 묶는다

⚠️ 이 페이지를 보고 규칙을 고치면 580장 전부가 오염된다. 남는 깨끗한 표본이
   없어지므로, 여기서 얻은 것은 **가설**로만 쓰고 검증은 따로 설계해야 한다.

    python eda/88_error_gallery.py --workers 4
"""
import os
import csv
import sys
import json
import time
import shutil
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
DEST = os.path.join(ROOT, "오답모음")

_P = None


def load_labels():
    """팀 383 + 내 197. 확장자가 어긋난 것은 실제 파일로 맞춘다."""
    lab, src = {}, {}
    p = os.path.join(ROOT, "custom_data", "labels.csv")
    with open(p, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            lab[r["source_file"]] = r["final_date"]
            src[r["source_file"]] = r["split"]

    h3 = json.load(open(os.path.join(OUT, "truth_holdout3.json"),
                        encoding="utf-8"))["labels"]
    for k, v in h3.items():
        lab[k] = v
        src[k] = "holdout3"

    # ⚠️ 홀드아웃4 는 **최종 판정 전용**이다. 97단계에서 labels.csv 에 합치면서
    #    이 갤러리에도 딸려 들어왔는데, 여기 오답을 들여다보는 순간 마지막 남은
    #    깨끗한 표본이 사라진다. 읽어 들이되 바로 걷어낸다.
    drop = [k for k, v in src.items() if v == "holdout4"]
    for k in drop:
        lab.pop(k, None)
        src.pop(k, None)
    if drop:
        print(f"홀드아웃4 {len(drop)}장 제외 (최종 판정 전용)")

    # ⚠️ 77단계에서 truth_holdout2.json 의 확장자만 고치고 labels.csv 는
    #    두었다. 그대로 두면 imread 가 None 을 내고 전부 미검출로 세어져
    #    수치가 실제보다 나빠진다. 여기서 실제 파일명으로 맞춘다.
    fixed = {}
    have = {}
    for n in os.listdir(IMG_DIR):
        have[os.path.splitext(n)[0]] = n
    for k in list(lab):
        if os.path.exists(os.path.join(IMG_DIR, k)):
            continue
        stem = os.path.splitext(k)[0]
        if stem in have:
            fixed[k] = have[stem]
            lab[have[stem]] = lab.pop(k)
            src[have[stem]] = src.pop(k)
    if fixed:
        print(f"확장자 보정 {len(fixed)}장: "
              f"{list(fixed.items())[:3]}{' ...' if len(fixed) > 3 else ''}")
    return lab, src


def build():
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
    retry, _ = combined.build_rapid_retry(RAPID_DIR)
    ko, _ = combined.build_rapid_rec(RAPID_DIR, combined.REC_KOREAN)
    en, _ = combined.build_rapid_rec(RAPID_DIR, combined.REC_EN)
    return combined.CombinedPipeline(rapid, team2_rules, fb, rapid_retry=retry,
                                     rec_korean=ko, rec_en=en)


def work(f):
    global _P
    if _P is None:
        _P = build()
    t0 = time.time()
    fields = _P.predict(os.path.join(IMG_DIR, f))
    return {"file": f, "sec": round(time.time() - t0, 2),
            "pred": "-".join(fields[k] for k in ("year", "month", "day"))}


def s3(v):
    if not v or v == "NONE":
        return ("NONE",) * 3
    q = str(v).split("-")
    return tuple(q) if len(q) == 3 else ("NONE",) * 3


def kind(g, p):
    if p == ("NONE",) * 3:
        return "미검출"
    if g == ("NONE",) * 3:
        return "오탐"
    hit = [g[i] == p[i] for i in range(3)]
    if hit == [False, True, True]:
        return "연도만 틀림"
    if hit == [True, False, False] and g[1] == p[2] and g[2] == p[1]:
        return "월일 뒤바뀜"
    if g[0] == p[2] and g[2] == p[0] and g[1] == p[1] and not hit[0]:
        return "연일 뒤바뀜"
    if sum(hit) == 2:
        return "한 필드 오독"
    if "NONE" in p:
        return "일부만 인식"
    return "전부 다름"


def thumb(src, dst, long_side=1400):
    """원본은 2MB 가 넘는다. 페이지가 무거워지지 않게 줄여서 복사한다."""
    try:
        import cv2
        import numpy as np
        a = cv2.imdecode(np.fromfile(src, dtype=np.uint8), cv2.IMREAD_COLOR)
        if a is None:
            shutil.copy2(src, dst)
            return
        h, w = a.shape[:2]
        m = max(h, w)
        if m > long_side:
            s = long_side / m
            a = cv2.resize(a, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", a, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if ok:
            buf.tofile(dst)
        else:
            shutil.copy2(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    lab, split = load_labels()
    files = [f for f in sorted(lab) if os.path.exists(os.path.join(IMG_DIR, f))]
    print(f"라벨 {len(lab)}장 중 이미지가 있는 {len(files)}장 실행", flush=True)

    t0 = time.time()
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(work, files, chunksize=2))
    pred = {r["file"]: r["pred"] for r in rows}
    sec = sum(r["sec"] for r in rows) / max(len(rows), 1)
    print(f"  {time.time()-t0:.0f}초 (장당 {sec:.2f}초)", flush=True)

    json.dump({"pred": pred, "label": {f: lab[f] for f in files},
               "split": {f: split[f] for f in files}},
              open(os.path.join(OUT, "88_all_pred.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    wrong = [f for f in files if s3(lab[f]) != s3(pred[f])]
    ok = len(files) - len(wrong)
    fld = sum(sum(s3(lab[f])[i] == s3(pred[f])[i] for i in range(3)) for f in files)
    print(f"완전일치 {ok}/{len(files)} ({ok/len(files)*100:.1f}%) · "
          f"필드 정확도 {fld/(len(files)*3)*100:.1f}% · 오답 {len(wrong)}장", flush=True)

    os.makedirs(os.path.join(DEST, "imgs"), exist_ok=True)
    data = []
    for f in wrong:
        thumb(os.path.join(IMG_DIR, f), os.path.join(DEST, "imgs", f))
        g, p = s3(lab[f]), s3(pred[f])
        data.append({"f": f, "truth": lab[f], "pred": pred[f],
                     "kind": kind(g, p), "split": split[f],
                     "hit": [g[i] == p[i] for i in range(3)]})

    from collections import Counter
    by_kind = Counter(d["kind"] for d in data)
    by_split = Counter(d["split"] for d in data)
    tot_split = Counter(split[f] for f in files)

    html = HTML
    html = html.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    html = html.replace("__SUMMARY__", json.dumps({
        "total": len(files), "wrong": len(wrong), "exact": ok,
        "field": round(fld / (len(files) * 3) * 100, 1),
        "by_kind": dict(by_kind.most_common()),
        "by_split": {k: {"wrong": by_split.get(k, 0), "total": tot_split[k]}
                     for k in tot_split},
    }, ensure_ascii=False))
    with open(os.path.join(DEST, "오답모음.html"), "w", encoding="utf-8") as fh:
        fh.write(html)

    with open(os.path.join(DEST, "오답목록.csv"), "w", encoding="utf-8-sig",
              newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["파일", "정답", "모델예측", "유형", "집합"])
        for d in data:
            w.writerow([d["f"], d["truth"], d["pred"], d["kind"], d["split"]])

    size = sum(os.path.getsize(os.path.join(DEST, "imgs", d["f"])) for d in data)
    print(f"\n오답 {len(data)}장 ({size/1024/1024:.0f}MB)")
    for k, v in by_kind.most_common():
        print(f"  {k:12s} {v:3d}")
    print(f"\n→ {DEST}\\오답모음.html")


HTML = r"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>오답 모음</title>
<style>
:root {
  --bg:#fbfaf8; --fg:#1a1a18; --muted:#6b6b64; --line:#e2e0da;
  --card:#fff; --ok:#2f6f4f; --bad:#a33a2a; --chip:#f1efe9;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg:#171714; --fg:#ece9e3; --muted:#9a968c; --line:#33322c;
    --card:#1f1e1a; --ok:#6fbf8f; --bad:#e0705c; --chip:#26251f;
  }
}
:root[data-theme="dark"] {
  --bg:#171714; --fg:#ece9e3; --muted:#9a968c; --line:#33322c;
  --card:#1f1e1a; --ok:#6fbf8f; --bad:#e0705c; --chip:#26251f;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
 font-family:-apple-system,"Segoe UI","Malgun Gothic",sans-serif;line-height:1.55}
.wrap{max-width:1180px;margin:0 auto;padding:22px 16px 60px}
h1{font-size:1.3rem;margin:0 0 4px}
.sub{color:var(--muted);font-size:.875rem;margin:0 0 18px}
.stats{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:16px}
.stat{background:var(--card);border:1px solid var(--line);border-radius:9px;
 padding:10px 14px;min-width:104px}
.stat b{display:block;font-size:1.5rem;line-height:1.2}
.stat span{font-size:.75rem;color:var(--muted)}
.filters{display:flex;gap:7px;flex-wrap:wrap;margin-bottom:18px}
.chip{background:var(--chip);border:1px solid var(--line);border-radius:999px;
 padding:5px 13px;font-size:.8125rem;cursor:pointer;color:var(--fg);font-family:inherit}
.chip[aria-pressed="true"]{background:var(--fg);color:var(--bg);border-color:var(--fg)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
 overflow:hidden;display:flex;flex-direction:column}
.card img{width:100%;height:230px;object-fit:cover;background:#000;cursor:zoom-in;display:block}
.body{padding:11px 13px 13px}
.fn{font-family:ui-monospace,Consolas,monospace;font-size:.75rem;color:var(--muted);
 display:flex;justify-content:space-between;gap:8px;margin-bottom:7px}
.dates{font-family:ui-monospace,Consolas,monospace;font-size:.9375rem}
.dates div{display:flex;gap:8px;align-items:baseline}
.dates .k{color:var(--muted);font-size:.75rem;width:34px;flex:none;font-family:inherit}
.f{padding:0 1px}
.f.bad{color:var(--bad);font-weight:700;text-decoration:underline;
 text-decoration-style:wavy;text-underline-offset:3px}
.tag{display:inline-block;margin-top:8px;font-size:.75rem;padding:2px 8px;
 border-radius:999px;background:var(--chip);border:1px solid var(--line)}
dialog{border:none;background:transparent;padding:0;max-width:96vw;max-height:96vh}
dialog::backdrop{background:rgba(0,0,0,.85)}
dialog img{max-width:96vw;max-height:92vh;display:block}
dialog .cap{color:#fff;font-family:ui-monospace,Consolas,monospace;
 font-size:.8125rem;text-align:center;padding:8px}
.note{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--bad);
 border-radius:8px;padding:11px 14px;font-size:.875rem;margin-bottom:18px}
@media (max-width:520px){ .card img{height:180px} }
</style>
</head>
<body>
<div class="wrap">
  <h1>오답 모음</h1>
  <p class="sub">라벨이 있는 전체를 현재 모델로 돌려, 정답과 다른 것만 모았습니다.</p>

  <div class="note">
    이 페이지를 보고 규칙을 고치면 여기 쓰인 표본 전부가 오염됩니다.
    여기서 얻은 것은 <b>가설</b>로만 쓰고, 검증은 쓰지 않은 표본으로 따로 해야 합니다.
  </div>

  <div class="stats" id="stats"></div>
  <div class="filters" id="filters"></div>
  <div class="grid" id="grid"></div>
</div>

<dialog id="dlg"><img id="dimg" alt=""><div class="cap" id="dcap"></div></dialog>

<script>
const DATA = __DATA__;
const S = __SUMMARY__;
let kindF = null, splitF = null;

document.getElementById("stats").innerHTML = `
  <div class="stat"><b>${S.total}</b><span>라벨 있는 장수</span></div>
  <div class="stat"><b>${S.wrong}</b><span>오답</span></div>
  <div class="stat"><b>${(S.exact / S.total * 100).toFixed(1)}%</b><span>완전일치</span></div>
  <div class="stat"><b>${S.field}%</b><span>필드 정확도</span></div>`;

function chips() {
  const ks = Object.entries(S.by_kind);
  const sp = Object.entries(S.by_split);
  document.getElementById("filters").innerHTML =
    `<button class="chip" data-t="k" data-v="" aria-pressed="${kindF === null}">유형 전체 (${S.wrong})</button>` +
    ks.map(([k, v]) => `<button class="chip" data-t="k" data-v="${k}" aria-pressed="${kindF === k}">${k} ${v}</button>`).join("") +
    `<span style="width:14px"></span>` +
    `<button class="chip" data-t="s" data-v="" aria-pressed="${splitF === null}">집합 전체</button>` +
    sp.map(([k, v]) => `<button class="chip" data-t="s" data-v="${k}" aria-pressed="${splitF === k}">${k} ${v.wrong}/${v.total}</button>`).join("");
  document.querySelectorAll(".chip").forEach(b => b.onclick = () => {
    const v = b.dataset.v || null;
    if (b.dataset.t === "k") kindF = v; else splitF = v;
    chips(); render();
  });
}

function render() {
  const rows = DATA.filter(d => (!kindF || d.kind === kindF) && (!splitF || d.split === splitF));
  document.getElementById("grid").innerHTML = rows.map(d => {
    const t = d.truth.split("-"), p = d.pred.split("-");
    const cells = p.map((x, i) => `<span class="f ${d.hit[i] ? "" : "bad"}">${x}</span>`).join("-");
    return `<div class="card">
      <img src="imgs/${d.f}" alt="" loading="lazy" data-f="${d.f}" data-c="정답 ${d.truth} · 예측 ${d.pred}">
      <div class="body">
        <div class="fn"><span>${d.f}</span><span>${d.split}</span></div>
        <div class="dates">
          <div><span class="k">정답</span><span>${t.join("-")}</span></div>
          <div><span class="k">예측</span><span>${cells}</span></div>
        </div>
        <span class="tag">${d.kind}</span>
      </div></div>`;
  }).join("") || `<p style="color:var(--muted)">해당 없음</p>`;

  document.querySelectorAll(".grid img").forEach(im => im.onclick = () => {
    document.getElementById("dimg").src = im.src;
    document.getElementById("dcap").textContent = im.dataset.f + "  —  " + im.dataset.c;
    document.getElementById("dlg").showModal();
  });
}
document.getElementById("dlg").onclick = e => { if (e.target.id === "dlg") e.target.close(); };
chips(); render();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
