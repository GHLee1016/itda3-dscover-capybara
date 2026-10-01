"""
EDA 117단계 — YOLO 가 날짜 영역을 못 잡는 이미지는 어떤가

116단계에서 뜻밖의 결과가 나왔다.
    YOLO 가 잡은 635장 안에서는  도트 97.4% / 일반 96.1%
    **도트가 오히려 더 잘 읽힌다.**

지표가 맞는지 크롭을 눈으로 확인했고 맞았다(`EXP 2022.05.31` 같은 것들이
상위에 있었다). 그렇다면 결론은 하나다. **문제는 도트가 아니라 YOLO 가
영역을 못 잡는 쪽에 있다.** 828장 중 193장이 그렇다.

여기서는 두 무리의 성능을 갈라 본다. 남은 손잡이가 어디인지 정하려는 것이다.

    python eda/117_yolo_blind.py
"""
import os
import csv
import sys
import json

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "eda", "out")
IMG_DIR = os.path.join(ROOT, "images", "상품사진입니다")


def s3(v):
    if not v or v == "NONE":
        return ("NONE",) * 3
    q = str(v).split("-")
    return tuple(q) if len(q) == 3 else ("NONE",) * 3


def main():
    lab = {}
    with open(os.path.join(ROOT, "custom_data", "labels.csv"),
              encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            lab[r["source_file"]] = r["final_date"]

    # 예측은 두 곳에 나뉘어 있다. 580장(88)과 248장(106).
    pred = {}
    d88 = json.load(open(os.path.join(OUT, "88_all_pred.json"), encoding="utf-8"))
    pred.update(d88["pred"])
    p106 = os.path.join(OUT, "106_final.json")
    if os.path.exists(p106):
        pred.update(json.load(open(p106, encoding="utf-8"))["pred"])

    dot = json.load(open(os.path.join(OUT, "116_dotness.json"), encoding="utf-8"))
    seen = set(dot)                       # YOLO 가 날짜 영역을 잡은 것

    files = [f for f in sorted(lab)
             if f in pred and os.path.exists(os.path.join(IMG_DIR, f))]
    got = [f for f in files if f in seen]
    blind = [f for f in files if f not in seen]

    def score(group):
        ok = tot = miss = exact = 0
        for f in group:
            g, p = s3(lab[f]), s3(pred[f])
            ok += sum(g[i] == p[i] for i in range(3))
            tot += 3
            exact += (g == p)
            if p == ("NONE",) * 3 and g != ("NONE",) * 3:
                miss += 1
        n = max(len(group), 1)
        return {"n": len(group), "fa": ok / max(tot, 1) * 100,
                "ex": exact / n * 100, "miss": miss,
                "missr": miss / n * 100}

    A, B = score(got), score(blind)
    print(f"예측이 있는 {len(files)}장  ·  YOLO 잡음 {A['n']} / 못 잡음 {B['n']}\n")

    # 도트 무리도 다시 (같은 예측 집합으로)
    hi = [f for f in got if dot.get(f, 0) >= 3.0]
    lo = [f for f in got if dot.get(f, 0) < 3.0]
    H, Lw = score(hi), score(lo)

    L = ["# EDA 117 — YOLO 가 날짜 영역을 못 잡는 이미지\n",
         "116단계에서 **도트가 오히려 더 잘 읽힌다**는 결과가 나왔다(97.4% 대 96.1%).",
         "지표는 크롭을 눈으로 확인해 맞았다. 그렇다면 문제는 도트가 아니라",
         "**YOLO 가 영역을 못 잡는 쪽**이다. 두 무리를 갈라 본다.",
         "",
         "## YOLO 가 잡았는가로 가른 성능\n",
         "| 무리 | 장수 | 필드 정확도 | 완전일치 | 미검출 | 미검출률 |",
         "| --- | ---: | ---: | ---: | ---: | ---: |",
         f"| YOLO 가 **잡은** 것 | {A['n']} | **{A['fa']:.1f}%** | {A['ex']:.1f}% | {A['miss']} | {A['missr']:.1f}% |",
         f"| YOLO 가 **못 잡은** 것 | {B['n']} | **{B['fa']:.1f}%** | {B['ex']:.1f}% | {B['miss']} | {B['missr']:.1f}% |",
         "",
         f"차이 **{B['fa'] - A['fa']:+.1f}%p**",
         "",
         "## 도트 여부로 가른 성능 (YOLO 가 잡은 것 안에서)\n",
         "| 무리 | 장수 | 필드 정확도 | 미검출 |",
         "| --- | ---: | ---: | ---: |",
         f"| 도트 인쇄 | {H['n']} | **{H['fa']:.1f}%** | {H['miss']} |",
         f"| 일반 인쇄 | {Lw['n']} | **{Lw['fa']:.1f}%** | {Lw['miss']} |",
         ""]

    if B["fa"] < A["fa"] - 2:
        L.append("**YOLO 가 못 잡는 쪽이 뚜렷하게 나쁘다.** 도트 판독기를 더 다듬는 것보다")
        L.append("검출 범위를 넓히는 쪽이 남은 손잡이다. 다만 도트 합성으로 파인튜닝한")
        L.append("v2 는 잡던 것을 놓쳐 기각했으므로(EDA 115), 다시 한다면 실제 도트")
        L.append("이미지를 모아 라벨링하는 쪽이 낫다.")
    else:
        L.append("두 무리 차이가 크지 않다. YOLO 검출 여부가 성능을 가르지 않는다.")
        L.append("보조 경로와 주력 계단이 YOLO 없이도 대부분을 처리한다는 뜻이다.")
    L.append("")
    L.append("> ⚠️ YOLO 는 **보조 경로와 도트 칸에서만** 쓴다. 주력 계단(1~6칸)은")
    L.append("> YOLO 를 거치지 않으므로, 못 잡는다고 곧바로 실패하는 것은 아니다.")

    with open(os.path.join(OUT, "117_yolo_blind.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L[5:22]))
    print("\n→ eda/out/117_yolo_blind.md")


if __name__ == "__main__":
    main()
