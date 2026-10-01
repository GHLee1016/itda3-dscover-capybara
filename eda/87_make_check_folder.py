"""
EDA 87단계 — 라벨 대조용 폴더 만들기

83단계의 91.6% 는 전부 내가 단 라벨 위에 서 있다. 내 라벨 오류율을 모르면
그 수치에 오차범위를 붙일 수 없다. 사람이 일부를 다시 읽어 대조한다.

설계에서 지킨 것
  · **표본을 내가 고르지 않는다.** 고정 시드로 무작위 추출한다.
    내가 고르면 자신 있는 것만 넣게 되고, 오류율이 낮게 나온다.
  · **내 라벨을 먼저 보여주지 않는다.** 사람이 직접 읽어 입력한 뒤에야
    내 라벨이 공개되고 자동 대조된다. 먼저 보여주면 "그럴듯한데" 쪽으로
    판단이 끌려간다(확증 편향).
  · **원본 이미지를 쓴다.** 나는 긴 변 1024 로 줄인 것을 봤다. 줄이면서
    날아간 정보 때문에 내가 틀렸다면, 원본으로 봐야 그게 잡힌다.

    python eda/87_make_check_folder.py [--n 30]

2차 — 쟁점 라벨만
  1차(무작위 30장)는 30/30 일치였다. 다만 무작위라서 판단이 갈리는 것이
  거의 안 들어갔다(197장 중 10장뿐). 그 10장은 따로 확인해야 한다.
  **이건 내가 골랐으므로 오류율 추정에는 쓸 수 없다.** 쟁점을 닫는 용도다.

    python eda/87_make_check_folder.py --only 쟁점 --dest 라벨대조2
"""
import os
import sys
import json
import random
import shutil
import argparse

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "eda", "out")
IMG_DIR = os.path.join(ROOT, "images", "상품사진입니다")

SEED = 20260926          # 80단계(20260925)와 다른 값. 같은 시드면 같은 표본이 뽑힌다

# 라벨링하면서 근거를 따로 적어 둔, 판단이 갈릴 수 있는 것들.
# 무작위 표본에 거의 안 들어가므로(197장 중 10장) 따로 묶어 확인한다.
MFG_ONLY = ["002914.jpg", "001686.jpg", "002600.jpg", "002943.jpg"]
FOREIGN_2D = ["002726.jpg", "001290.jpg", "001875.jpg"]


HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>라벨 대조</title>
<style>
:root {
  --bg: #fbfaf8; --fg: #1a1a18; --muted: #6b6b64; --line: #e2e0da;
  --card: #ffffff; --accent: #2f6f4f; --bad: #a33a2a; --warn: #8a6d1f;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #171714; --fg: #ece9e3; --muted: #9a968c; --line: #33322c;
    --card: #1f1e1a; --accent: #6fbf8f; --bad: #e0705c; --warn: #d4ab45;
  }
}
:root[data-theme="dark"] {
  --bg: #171714; --fg: #ece9e3; --muted: #9a968c; --line: #33322c;
  --card: #1f1e1a; --accent: #6fbf8f; --bad: #e0705c; --warn: #d4ab45;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--fg);
  font-family: -apple-system, "Segoe UI", "Malgun Gothic", sans-serif;
  line-height: 1.6;
}
.wrap { max-width: 860px; margin: 0 auto; padding: 20px 16px 60px; }
h1 { font-size: 1.25rem; margin: 0 0 4px; }
.sub { color: var(--muted); font-size: .875rem; margin: 0 0 20px; }
.bar { height: 5px; background: var(--line); border-radius: 3px; overflow: hidden; margin-bottom: 18px; }
.bar > i { display: block; height: 100%; background: var(--accent); width: 0; transition: width .2s; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 16px; }
.imgbox { text-align: center; background: #000; border-radius: 8px; overflow: hidden; margin-bottom: 14px; }
.imgbox img { max-width: 100%; max-height: 62vh; display: block; margin: 0 auto; cursor: zoom-in; }
.meta { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; margin-bottom: 10px; flex-wrap: wrap; }
.fname { font-family: ui-monospace, Consolas, monospace; font-size: .8125rem; color: var(--muted); }
label { display: block; font-size: .8125rem; color: var(--muted); margin-bottom: 4px; }
.fields { display: flex; gap: 10px; align-items: flex-end; flex-wrap: wrap; }
.fields > div { flex: 1 1 90px; min-width: 80px; }
input[type=text] {
  width: 100%; padding: 9px 10px; font-size: 1.0625rem; text-align: center;
  font-family: ui-monospace, Consolas, monospace;
  border: 1px solid var(--line); border-radius: 7px;
  background: var(--bg); color: var(--fg);
}
input:focus { outline: 2px solid var(--accent); outline-offset: -1px; }
button {
  padding: 9px 16px; font-size: .9375rem; border-radius: 7px; cursor: pointer;
  border: 1px solid var(--line); background: var(--card); color: var(--fg);
  font-family: inherit;
}
button.primary { background: var(--accent); border-color: var(--accent); color: #fff; font-weight: 600; }
button:hover { filter: brightness(1.08); }
.row { display: flex; gap: 8px; margin-top: 14px; flex-wrap: wrap; align-items: center; }
.spacer { flex: 1; }
.verdict { margin-top: 14px; padding: 12px 14px; border-radius: 8px; border: 1px solid var(--line); }
.verdict.same { border-color: var(--accent); }
.verdict.diff { border-color: var(--bad); }
.verdict b { font-family: ui-monospace, Consolas, monospace; font-size: 1.0625rem; }
.tag { font-weight: 700; }
.tag.same { color: var(--accent); }
.tag.diff { color: var(--bad); }
.hint { font-size: .8125rem; color: var(--muted); margin-top: 10px; }
table { width: 100%; border-collapse: collapse; font-size: .875rem; margin-top: 12px; }
th, td { text-align: left; padding: 7px 8px; border-bottom: 1px solid var(--line); }
th { color: var(--muted); font-weight: 600; font-size: .8125rem; }
td.mono { font-family: ui-monospace, Consolas, monospace; }
.big { font-size: 2rem; font-weight: 700; }
textarea {
  width: 100%; height: 160px; margin-top: 10px; padding: 10px;
  font-family: ui-monospace, Consolas, monospace; font-size: .75rem;
  border: 1px solid var(--line); border-radius: 7px;
  background: var(--bg); color: var(--fg);
}
.note { background: var(--card); border: 1px solid var(--line); border-left: 3px solid var(--warn);
        border-radius: 7px; padding: 12px 14px; font-size: .875rem; margin-bottom: 18px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>라벨 대조</h1>
  <p class="sub">__SUB__</p>

  <div class="note">
    <b>읽는 기준</b> — 소비기한과 제조일자가 함께 있으면 <b>늦은 쪽</b>.
    일부만 판독되면 그 칸만 채우고 나머지는 비워 둡니다(비우면 NONE으로 처리).
    전혀 못 읽겠으면 <b>판독 불가</b>를 누르세요.
  </div>

  <div class="bar"><i id="bar"></i></div>
  <div id="main"></div>
</div>

<script>
const DATA = __DATA__;
const KEY = "__KEY__";   // file:// 은 여러 문서가 같은 저장소를 공유할 수 있어 회차마다 다르게 둔다
let state = {};
try { state = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { state = {}; }
let i = 0;
while (i < DATA.length && state[DATA[i].f]) i++;

function save() {
  try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {}
}
function norm(v) { return (v || "").trim() === "" ? "NONE" : (v || "").trim(); }
function pad(v, n) {
  v = norm(v);
  if (v === "NONE") return v;
  return v.padStart(n, "0");
}
function joined(y, m, d) { return pad(y,4) + "-" + pad(m,2) + "-" + pad(d,2); }

function render() {
  const bar = document.getElementById("bar");
  const done = Object.keys(state).length;
  bar.style.width = (done / DATA.length * 100) + "%";

  if (i >= DATA.length) return summary();

  const it = DATA[i];
  document.getElementById("main").innerHTML = `
    <div class="card">
      <div class="meta">
        <span>${i + 1} / ${DATA.length}</span>
        <span class="fname">${it.f}</span>
      </div>
      <div class="imgbox"><img src="imgs/${it.img}" alt="" id="pic"></div>
      <div class="fields">
        <div><label>연도 (YYYY)</label><input type="text" id="y" inputmode="numeric" maxlength="4" autocomplete="off"></div>
        <div><label>월</label><input type="text" id="m" inputmode="numeric" maxlength="2" autocomplete="off"></div>
        <div><label>일</label><input type="text" id="d" inputmode="numeric" maxlength="2" autocomplete="off"></div>
      </div>
      <div class="row">
        <button class="primary" id="ok">확인 (Enter)</button>
        <button id="skip">판독 불가</button>
        <div class="spacer"></div>
        <button id="back" ${i === 0 ? "disabled" : ""}>이전</button>
      </div>
      <p class="hint">사진을 클릭하면 새 창에서 원본 크기로 열립니다. 못 읽는 칸은 비워 두세요.</p>
      <div id="out"></div>
    </div>`;

  document.getElementById("pic").onclick = () => window.open("imgs/" + it.img, "_blank");
  document.getElementById("ok").onclick = () => check(false);
  document.getElementById("skip").onclick = () => check(true);
  document.getElementById("back").onclick = () => { i = Math.max(0, i - 1); render(); };
  ["y","m","d"].forEach(id => {
    document.getElementById(id).addEventListener("keydown", e => {
      if (e.key === "Enter") check(false);
    });
  });
  document.getElementById("y").focus();
}

function check(unreadable) {
  const it = DATA[i];
  const mine = unreadable ? "판독불가"
    : joined(document.getElementById("y").value,
             document.getElementById("m").value,
             document.getElementById("d").value);
  const same = mine === it.label;
  state[it.f] = { user: mine, ai: it.label, same: same, unreadable: unreadable };
  save();

  document.getElementById("out").innerHTML = `
    <div class="verdict ${same ? "same" : "diff"}">
      <div><span class="tag ${same ? "same" : "diff"}">${same ? "일치" : "불일치"}</span></div>
      <div style="margin-top:6px">내가 읽은 값 <b>${mine}</b></div>
      <div>AI 라벨 &nbsp;&nbsp;&nbsp;&nbsp;<b>${it.label}</b></div>
    </div>
    <div class="row"><button class="primary" id="next">다음 (Enter)</button></div>`;
  const nx = document.getElementById("next");
  nx.onclick = () => { i++; render(); };
  nx.focus();
  document.onkeydown = e => {
    if (e.key === "Enter" && document.getElementById("next")) {
      document.onkeydown = null; i++; render();
    }
  };
}

function summary() {
  const rows = DATA.map(it => state[it.f]).filter(Boolean);
  const n = rows.length;
  const same = rows.filter(r => r.same).length;
  const diffs = DATA.filter(it => state[it.f] && !state[it.f].same);
  const rate = n ? (same / n * 100).toFixed(1) : "0.0";

  document.getElementById("main").innerHTML = `
    <div class="card">
      <h2 style="margin:0 0 4px;font-size:1.05rem">대조 결과</h2>
      <p class="sub" style="margin:0 0 12px">${n}장 중 ${same}장 일치</p>
      <div class="big">${rate}%</div>
      <p class="sub" style="margin:4px 0 0">
        불일치 ${n - same}장. 아래 표에서 어느 쪽이 맞는지 확인해 주세요 —
        제가 틀렸다면 라벨을 고치고, 검수자가 잘못 읽었다면 그대로 둡니다.
      </p>
      ${diffs.length ? `<table>
        <tr><th>파일</th><th>직접 읽은 값</th><th>AI 라벨</th></tr>
        ${diffs.map(it => `<tr>
          <td class="mono">${it.f}</td>
          <td class="mono">${state[it.f].user}</td>
          <td class="mono">${it.label}</td></tr>`).join("")}
      </table>` : `<p style="margin-top:12px">불일치 없음.</p>`}
      <div class="row" style="margin-top:18px">
        <button class="primary" id="copy">결과 복사</button>
        <button id="again">처음부터 다시</button>
      </div>
      <p class="hint">복사한 내용을 대화창에 붙여 넣어 주시면 제가 라벨을 고치고 수치에 오차범위를 붙이겠습니다.</p>
      <textarea id="json" readonly></textarea>
    </div>`;

  const payload = { 대조장수: n, 일치: same, 일치율: rate + "%",
                    불일치: diffs.map(it => ({ 파일: it.f, 직접: state[it.f].user, AI: it.label })) };
  const txt = JSON.stringify(payload, null, 1);
  document.getElementById("json").value = txt;
  document.getElementById("copy").onclick = () => {
    const ta = document.getElementById("json");
    ta.select();
    try { document.execCommand("copy"); } catch (e) {}
    if (navigator.clipboard) { navigator.clipboard.writeText(txt).catch(() => {}); }
    document.getElementById("copy").textContent = "복사됨";
  };
  document.getElementById("again").onclick = () => {
    state = {}; save(); i = 0; render();
  };
  document.getElementById("bar").style.width = "100%";
}

render();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--only", choices=["쟁점"],
                    help="무작위 대신 판단이 갈리는 것만 모은다")
    ap.add_argument("--dest", default="라벨대조")
    a = ap.parse_args()

    dest = os.path.join(ROOT, a.dest)

    lab = json.load(open(os.path.join(OUT, "truth_holdout3.json"),
                         encoding="utf-8"))["labels"]

    if a.only == "쟁점":
        partial = sorted(f for f, v in lab.items() if "NONE" in v)
        pick = [f for f in dict.fromkeys(FOREIGN_2D + MFG_ONLY + partial)
                if f in lab]
        random.Random(SEED).shuffle(pick)
    else:
        files = sorted(lab)                  # 정렬해야 시드가 같은 표본을 준다
        rng = random.Random(SEED)
        pick = sorted(rng.sample(files, min(a.n, len(files))))
        rng.shuffle(pick)                    # 보는 순서는 섞는다(순서 효과 방지)

    DEST = dest
    os.makedirs(os.path.join(DEST, "imgs"), exist_ok=True)
    쟁점 = a.only == "쟁점"
    data = []
    for f in pick:
        src = os.path.join(IMG_DIR, f)
        if not os.path.exists(src):
            print(f"  원본 없음, 건너뜀: {f}")
            continue
        shutil.copy2(src, os.path.join(DEST, "imgs", f))
        data.append({"f": f, "img": f, "label": lab[f]})

    SUB = ("판단이 갈릴 수 있는 __N__장입니다. 무작위가 아니라 제가 고른 것이라"
           " 오류율 추정에는 쓰지 않고, 쟁점을 닫는 데만 씁니다."
           if 쟁점 else
           "무작위 __N__장. 사진을 보고 직접 입력하면 그때 제 라벨이 공개되어 대조됩니다.")

    html = (HTML.replace("__SUB__", SUB)
                .replace("__KEY__", "h3check-" + ("쟁점" if 쟁점 else f"rnd{SEED}"))
                .replace("__DATA__", json.dumps(data, ensure_ascii=False))
                .replace("__N__", str(len(data))))
    with open(os.path.join(DEST, "대조표.html"), "w", encoding="utf-8") as fh:
        fh.write(html)

    # 표 형태를 선호할 경우를 위한 대조표. 라벨은 마지막 열에 둬서
    # 먼저 읽고 나중에 보도록 한다.
    with open(os.path.join(DEST, "대조표.csv"), "w", encoding="utf-8-sig") as fh:
        fh.write("번호,파일,직접읽은값,AI라벨\n")
        for k, d in enumerate(data, 1):
            fh.write(f"{k},{d['f']},,{d['label']}\n")

    with open(os.path.join(DEST, "읽어주세요.txt"), "w", encoding="utf-8") as fh:
        fh.write(
            "라벨 대조\n"
            "=========\n\n"
            "대조표.html 을 더블클릭해서 브라우저로 여세요.\n\n"
            "사진을 보고 날짜를 직접 입력하면, 그때 AI 라벨이 공개되어 자동으로\n"
            "대조됩니다. 먼저 보여주지 않는 이유는, 라벨을 먼저 보면 판단이\n"
            "그쪽으로 끌려가 대조가 의미를 잃기 때문입니다.\n\n"
            "읽는 기준\n"
            "  · 소비기한과 제조일자가 함께 있으면 늦은 쪽\n"
            "  · 일부만 읽히면 그 칸만 채우고 나머지는 비워 둔다\n"
            "  · 전혀 못 읽겠으면 '판독 불가'\n\n"
            "다 하시면 '결과 복사'를 눌러 대화창에 붙여 넣어 주세요.\n"
            "중간에 닫아도 진행 상황은 브라우저에 남습니다.\n\n"
            + (f"표본: 판단이 갈릴 수 있는 {len(data)}장.\n"
               "무작위 30장 대조에서는 이런 것이 거의 안 뽑혔습니다.\n"
               "제가 고른 것이므로 오류율 추정에는 쓰지 않고,\n"
               "라벨을 확정하는 데만 씁니다.\n\n"
               if 쟁점 else
               f"표본: 라벨 {len(lab)}장에서 무작위 {len(data)}장 (시드 {SEED}).\n"
               "무작위인 이유는, 제가 고르면 자신 있는 것만 넣게 되어\n"
               "오류율이 낮게 나오기 때문입니다.\n\n")
            + "사진은 원본입니다. AI 는 긴 변 1024 로 줄인 것을 봤으므로,\n"
            "줄이면서 날아간 정보 때문에 AI 가 틀렸다면 여기서 잡힙니다.\n")

    size = sum(os.path.getsize(os.path.join(DEST, "imgs", d["img"])) for d in data)
    print(f"대조용 {len(data)}장 준비 ({size/1024/1024:.0f}MB)")
    print(f"→ {DEST}")
    print("   대조표.html   브라우저로 열기")
    print("   대조표.csv    표로 하고 싶을 때")


if __name__ == "__main__":
    main()
