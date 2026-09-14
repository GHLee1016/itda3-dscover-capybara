"""
EDA 59단계 — 두 파이프라인을 합치면 이득이 있는가

우리(PP-OCR + YOLO/ONNX)와 팀원(classical CV + EasyOCR)은 검출·인식 엔진이
완전히 다르다. 서로 다른 이미지를 맞힌다면 합쳐서 이득이 난다.

홀드아웃2 198장에서 두 예측을 이미 받아 놓았다.
  우리   : eda/out/57_env_snapshot.json   (66.8%)
  팀원   : eda/out/58_team_pred.json      (30.1%)

합치는 방식을 네 가지로 모의한다. 실제로 붙이기 전에 **이득이 있는지부터** 본다.
팀원 엔진을 붙이는 비용이 크기 때문이다 — easyocr 는 torch 를 끌고 오고,
torch 는 임포트만으로 우리 파이프라인을 2.4배 느리게 만든다(EDA 52~53).
별도 프로세스로 격리해야 하는데, 그 복잡도를 감수할 만한지가 이 단계의 질문이다.
"""
import os
import sys
import json
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "eda", "out")


def split3(v):
    if not v or v == "NONE":
        return ("NONE",) * 3
    p = str(v).split("-")
    return tuple(p) if len(p) == 3 else ("NONE",) * 3


def is_none(t):
    return t == ("NONE",) * 3


def score(pred, lab, files):
    ok = tot = filled = exact = 0
    for f in files:
        g, p = split3(lab[f]), pred[f]
        ok += sum(g[i] == p[i] for i in range(3))
        tot += 3
        filled += not is_none(p)
        exact += g == p
    return {"score": ok / tot, "fill": filled / len(files),
            "exact": exact / len(files), "hits": ok, "tot": tot}


def main():
    ap = argparse.ArgumentParser()
    a = ap.parse_args()

    lab = json.load(open(os.path.join(OUT_DIR, "truth_holdout2.json"),
                         encoding="utf-8"))["labels"]
    ours_raw = json.load(open(os.path.join(OUT_DIR, "57_env_snapshot.json"),
                              encoding="utf-8"))["pred"]
    team_raw = json.load(open(os.path.join(OUT_DIR, "58_team_pred.json"),
                              encoding="utf-8"))["pred"]
    files = [f for f in lab if f in ours_raw and f in team_raw]

    A = {f: split3(ours_raw[f]) for f in files}      # 우리만
    B = {f: split3(team_raw[f]) for f in files}      # 팀원만

    # 합치기 1 — 이미지 단위: 우리가 전부 NONE 일 때만 팀원 것을 쓴다
    M1 = {f: (B[f] if is_none(A[f]) else A[f]) for f in files}

    # 합치기 2 — 필드 단위: 우리가 NONE 인 칸만 팀원 값으로 채운다
    M2 = {}
    for f in files:
        M2[f] = tuple(B[f][i] if A[f][i] == "NONE" else A[f][i] for i in range(3))

    # 합치기 3 — 둘이 일치할 때만 신뢰 (정밀도 우선). 참고용 하한
    M3 = {f: (A[f] if A[f] == B[f] else ("NONE",) * 3) for f in files}

    # 오라클 — 이미지마다 더 나은 쪽을 고르는 상한 (실제로는 불가능)
    OR = {}
    for f in files:
        g = split3(lab[f])
        ha = sum(g[i] == A[f][i] for i in range(3))
        hb = sum(g[i] == B[f][i] for i in range(3))
        OR[f] = A[f] if ha >= hb else B[f]

    rows = [("우리 (PP-OCR + YOLO)", A),
            ("팀원 (classical CV + EasyOCR)", B),
            ("합치기 1 — 우리 실패 시에만 팀원", M1),
            ("합치기 2 — NONE 칸만 팀원으로 채움", M2),
            ("합치기 3 — 둘이 일치할 때만 채택", M3),
            ("오라클 상한 (도달 불가)", OR)]

    L = ["# EDA 59 — 두 파이프라인을 합치면 이득이 있는가\n",
         "우리와 팀원은 검출·인식 엔진이 완전히 다르다. 서로 다른 이미지를 맞힌다면",
         "합쳐서 이득이 난다. 홀드아웃2 198장에서 실제로 그런지 본다.",
         "",
         "| 구성 | **부분점수** | 값이 채워진 행 | 완전일치 |",
         "| --- | ---: | ---: | ---: |"]
    S = {}
    for name, pred in rows:
        s = score(pred, lab, files)
        S[name] = s
        L.append(f"| {name} | **{s['score']*100:.1f}%** | {s['fill']*100:.1f}% | "
                 f"{s['exact']*100:.1f}% |")
    L.append("")

    base = S["우리 (PP-OCR + YOLO)"]["score"]
    for name in ("합치기 1 — 우리 실패 시에만 팀원",
                 "합치기 2 — NONE 칸만 팀원으로 채움"):
        d = (S[name]["score"] - base) * 100
        L.append(f"- **{name}**: {d:+.1f}%p")
    orc = (S["오라클 상한 (도달 불가)"]["score"] - base) * 100
    L.append(f"- 오라클 상한: {orc:+.1f}%p — **어떤 합치기 방식도 이 값을 넘을 수 없다**")
    L.append("")

    # 상호보완성 상세
    a_none = [f for f in files if is_none(A[f])]
    both = [f for f in a_none if not is_none(B[f])]
    gained = []
    for f in both:
        g = split3(lab[f])
        hb = sum(g[i] == B[f][i] for i in range(3))
        gained.append((f, lab[f], "-".join(B[f]), hb))
    L.append("## 우리가 놓친 이미지를 팀원이 건지는가\n")
    L.append(f"- 우리가 전부 NONE 으로 낸 이미지: **{len(a_none)}장**")
    L.append(f"- 그중 팀원이 값을 낸 것: **{len(both)}장**")
    L.append(f"- 그 값이 실제로 맞은 필드 수: **{sum(h for _, _, _, h in gained)}개** "
             f"(최대 {len(both)*3}개)")
    L.append("")
    if gained:
        L.append("| 파일 | 정답 | 팀원 출력 | 맞은 필드 |")
        L.append("| --- | --- | --- | ---: |")
        for f, g, p, h in sorted(gained, key=lambda t: -t[3]):
            L.append(f"| `{f}` | {g} | {p} | {h}/3 |")
        L.append("")

    # 반대 방향 — 팀원이 맞고 우리가 틀린 경우 전체
    better = []
    for f in files:
        g = split3(lab[f])
        ha = sum(g[i] == A[f][i] for i in range(3))
        hb = sum(g[i] == B[f][i] for i in range(3))
        if hb > ha:
            better.append((f, lab[f], "-".join(A[f]), "-".join(B[f]), ha, hb))
    L.append(f"## 팀원 쪽이 더 나은 이미지 — 전체 {len(better)}장\n")
    if better:
        L.append("| 파일 | 정답 | 우리 | 팀원 | 우리 | 팀원 |")
        L.append("| --- | --- | --- | --- | ---: | ---: |")
        for f, g, pa, pb, ha, hb in better:
            L.append(f"| `{f}` | {g} | {pa} | {pb} | {ha}/3 | {hb}/3 |")
    L.append("")

    L.append("## 붙이는 비용\n")
    L.append("팀원 엔진(EasyOCR)은 torch 를 요구한다. torch 는 **임포트만으로**")
    L.append("우리 파이프라인을 장당 2.63 → 5.88초로 만든다(EDA 52~53, 500장이 제한 초과).")
    L.append("또 easyocr 가 `opencv-python-headless` 를 끌고 와 고정해 둔 cv2 버전을 바꾼다")
    L.append("(이번에 실제로 발생 — cv2 4.10.0 → 4.11.0, 되돌리며 numpy 까지 흔들렸다).")
    L.append("")
    L.append("따라서 합치려면 **별도 프로세스로 격리**해야 한다. 우리 루프가 끝난 뒤")
    L.append("실패한 이미지 목록만 자식 프로세스에 넘겨 EasyOCR 로 한 번 더 돌리는 구조다.")
    L.append("가중치 99MB 가 추가되고, 채점 환경에서 프로세스를 하나 더 띄우는 위험이 붙는다.")

    with open(os.path.join(OUT_DIR, "59_merge_eval.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L[:18]))
    print(f"\n→ eda/out/59_merge_eval.md")


if __name__ == "__main__":
    main()
