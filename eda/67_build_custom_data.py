"""
EDA 67단계 — 가산점 심사용 custom_data/ 구성

운영진 안내(2026-09-14): 자체 수집·라벨링 데이터는 저장소 루트의 custom_data/ 에
두고, README 에 폴더 구조를 1~2줄 남긴다. 내부 구조는 자율.

우리 기여는 '수집'이 아니라 **라벨링**이다. 배포된 3,352장에는 정답이 전혀 없어
직접 판독·검수한 383장을 만들었고, 그 라벨이 없었다면 세 파이프라인 중 어느 것이
나은지 판단할 방법 자체가 없었다.

산출물
  custom_data/labels.csv            383장 통합 (제출 CSV 와 같은 스키마 + split)
  custom_data/labels_*.json         집합별 원본
  custom_data/README.md             구조·기준·통계
  custom_data/tool/                 직접 만든 라벨링 도구
"""
import os
import sys
import json
import shutil
import csv
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "eda", "out")
DST = os.path.join(ROOT, "custom_data")

SETS = [
    ("tuning",   "truth_v4.json",         "튜닝셋 — 규칙 설계·파라미터 결정에 사용"),
    ("holdout1", "truth_holdout.json",    "1차 홀드아웃 — 연도 후행형 버그 발견에 사용"),
    ("holdout2", "truth_holdout2.json",   "2차 홀드아웃 — 최종 성능 판정에만 사용"),
]


def split3(v):
    if not v or v == "NONE":
        return ("NONE", "NONE", "NONE")
    p = str(v).split("-")
    return tuple(p) if len(p) == 3 else ("NONE", "NONE", "NONE")


def main():
    os.makedirs(DST, exist_ok=True)

    rows, per_set = [], {}
    for name, fn, _desc in SETS:
        src = os.path.join(OUT_DIR, fn)
        if not os.path.exists(src):
            print(f"!! 없음: {src}")
            continue
        data = json.load(open(src, encoding="utf-8"))
        labels = data["labels"]
        per_set[name] = len(labels)
        shutil.copyfile(src, os.path.join(DST, f"labels_{name}.json"))
        for f, v in labels.items():
            y, m, d = split3(v)
            rows.append({
                "image_id": os.path.splitext(f)[0],
                "year": y, "month": m, "day": d,
                "final_date": v if v else "NONE",
                "split": name,
                "source_file": f,
            })

    rows.sort(key=lambda r: r["image_id"])
    with open(os.path.join(DST, "labels.csv"), "w", encoding="utf-8-sig",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "year", "month", "day",
                                          "final_date", "split", "source_file"])
        w.writeheader()
        w.writerows(rows)

    # 라벨링 도구 동봉 (이미지 축소본은 제외 — 용량)
    tool = os.path.join(DST, "tool")
    os.makedirs(tool, exist_ok=True)
    for fn in ("index.html", "montage.py", "montage_all.py", "make_tasks.py",
               "prep_active.py"):
        p = os.path.join(ROOT, "label", fn)
        if os.path.exists(p):
            shutil.copyfile(p, os.path.join(tool, fn))

    years = Counter(r["year"] for r in rows)
    n = len(rows)

    md = [
        "# custom_data — 직접 라벨링한 소비기한 정답 383장\n",
        "배포된 3,352장에는 **정답 라벨이 전혀 없다.** 자체 라벨링 도구를 만들어",
        "직접 판독·검수한 383장이며, 파이프라인 설계와 최종 성능 판정에 이 라벨만 썼다.",
        "",
        "> 이미지 자체는 운영진이 배포한 원본이므로 재배포하지 않는다.",
        "> `source_file` 컬럼이 배포 데이터셋의 파일명을 그대로 가리킨다.",
        "",
        "## 폴더 구조\n",
        "```",
        "custom_data/",
        "├── labels.csv            383장 통합 라벨 (제출 CSV 스키마 + split 컬럼)",
        "├── labels_tuning.json    튜닝셋 %d장" % per_set.get("tuning", 0),
        "├── labels_holdout1.json  1차 홀드아웃 %d장" % per_set.get("holdout1", 0),
        "├── labels_holdout2.json  2차 홀드아웃 %d장" % per_set.get("holdout2", 0),
        "└── tool/                 직접 만든 라벨링 도구 (index.html 외)",
        "```",
        "",
        "## labels.csv 스키마\n",
        "| 컬럼 | 설명 |",
        "| --- | --- |",
        "| `image_id` | 배포 데이터셋 파일명(확장자 제외) |",
        "| `year` / `month` / `day` | 사람이 판독한 값. 판독 불가 필드는 `NONE` |",
        "| `final_date` | `YYYY-MM-DD`. 전부 미판독이면 `NONE` |",
        "| `split` | `tuning` / `holdout1` / `holdout2` |",
        "| `source_file` | 원본 파일명(확장자 포함) |",
        "",
        "## 분할을 셋으로 나눈 이유\n",
        "| 집합 | 장수 | 용도 |",
        "| --- | ---: | --- |",
    ]
    for name, _fn, desc in SETS:
        md.append(f"| `{name}` | {per_set.get(name, 0)} | {desc} |")
    md += [
        "",
        "`holdout2` 198장은 **최종 판정에만** 썼다. 설계에 쓴 표본으로 성능을 재면",
        "그 표본에 맞춰진 정도를 성능으로 착각하게 된다. 실제로 합성·회귀 테스트를",
        "통과하고도 홀드아웃에서 떨어진 변경이 두 건 있었고, 그때부터 모든 채택 판정을",
        "홀드아웃 부분점수로만 내렸다.",
        "",
        "## 라벨링 기준 (작업 전에 문서로 고정)\n",
        "1. **소비기한과 제조일자가 함께 인쇄되면 늦은 쪽**을 정답으로 한다.",
        "   (`2025.09.30 부터 / 2026.06.29 까지` → `2026-06-29`)",
        "2. **2자리 연도의 순서**는 한국식 `YY.MM.DD` 를 기본으로 하되,",
        "   포장지가 수입품이면 유럽식 `DD.MM.YY` 로 읽는다.",
        "3. **일부 필드만 판독되면 그 필드만 채운다.** 연·월·일이 독립 채점이므로",
        "   `2025-05-NONE` 처럼 확인된 만큼만 남기는 편이 유리하다.",
        "4. **판독 불가·중의적인 이미지는 제외**한다. 억지로 채운 라벨은 평가를 망친다.",
        "   실제로 2장을 이 기준으로 뺐다(385장 작업 → 383장 확정).",
        "5. 달력에 없는 날짜(`2026-02-30` 등)는 오독으로 보고 다시 읽는다.",
        "",
        "## 통계\n",
        "| 항목 | 값 |",
        "| --- | ---: |",
        "| 총 라벨 | %d장 |" % n,
        "| 완전한 날짜(연·월·일 모두) | %d장 |" % sum(
            1 for r in rows if "NONE" not in (r["year"], r["month"], r["day"])),
        "| 연도 범위 | %s ~ %s |" % (
            min(y for y in years if y != "NONE"),
            max(y for y in years if y != "NONE")),
        "",
        "연도 분포: " + ", ".join(
            f"{y} {c}장" for y, c in sorted(years.items()) if y != "NONE"),
        "",
        "## 라벨링 도구\n",
        "`tool/index.html` 을 로컬 서버로 열면 후보 날짜와 앵커가 함께 표시되고",
        "휠 확대·드래그 이동이 된다. 숫자키로 후보를 고르고 `0` 은 NONE,",
        "`R` 로 90° 회전한다. 여러 명이 나눠 작업할 수 있게 `?shard=1..3` 을 지원한다.",
        "",
        "```bash",
        "cd custom_data/tool && python -m http.server 8000",
        "# http://localhost:8000/index.html",
        "```",
        "",
        "> 도구는 배포 이미지의 축소본을 `imgs/` 에서 읽도록 되어 있다.",
        "> 용량 때문에 축소본은 저장소에 넣지 않았다 — 원본에서 다시 만들 수 있다.",
    ]

    with open(os.path.join(DST, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    total = sum(os.path.getsize(os.path.join(dp, fn))
                for dp, _, fns in os.walk(DST) for fn in fns)
    print(f"custom_data/ 생성 — 라벨 {n}장, 파일 "
          f"{sum(len(fns) for _, _, fns in os.walk(DST))}개, {total/1024:.0f} KB")
    for name in per_set:
        print(f"  {name}: {per_set[name]}장")


if __name__ == "__main__":
    main()
