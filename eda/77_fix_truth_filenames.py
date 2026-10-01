"""
EDA 77단계 — 정답 파일명 확장자 불일치 수정

발견: 정답 JSON 의 키가 `000834.jpg` 인데 실제 파일은 `000834.jpeg` 다.
      평가 스크립트는 정답 키로 이미지를 열기 때문에, 이런 항목은 imread 가
      None 을 돌려주고 **자동으로 전부 NONE 처리**됐다.

      truth_holdout2  10/198
      truth_v4         3/137
      truth_holdout    3/48

영향 범위
  - 제출 노트북은 INPUT_DIR 를 직접 훑으므로 **영향 없다.** image_id 는 실제
    파일명에서 나온다.
  - 우리 평가 수치만 과소평가됐다. 홀드아웃2 198장 기준 수치는 최대 10장이
    자동 실패로 깔린 값이다.

수정 방식: 확장자만 실제 파일에 맞춰 키를 바꾼다. 라벨 값은 건드리지 않는다.
          원본은 `.bak` 으로 남긴다.

    python eda/77_fix_truth_filenames.py
"""
import os
import sys
import json
import shutil

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(ROOT, "images", "상품사진입니다")
OUT_DIR = os.path.join(ROOT, "eda", "out")
FILES = ("truth_v4.json", "truth_holdout.json", "truth_holdout2.json")


def main():
    stems = {}
    for x in os.listdir(IMG_DIR):
        stems.setdefault(os.path.splitext(x)[0], x)

    total_fixed = 0
    for fn in FILES:
        p = os.path.join(OUT_DIR, fn)
        if not os.path.exists(p):
            continue
        data = json.load(open(p, encoding="utf-8"))
        labels = data["labels"]

        fixed, unresolved = {}, []
        n_fix = 0
        for k, v in labels.items():
            if os.path.exists(os.path.join(IMG_DIR, k)):
                fixed[k] = v
                continue
            real = stems.get(os.path.splitext(k)[0])
            if real:
                fixed[real] = v
                n_fix += 1
            else:
                fixed[k] = v
                unresolved.append(k)

        if n_fix:
            bak = p + ".bak"
            if not os.path.exists(bak):
                shutil.copyfile(p, bak)
            data["labels"] = fixed
            # membership 이 있으면 키도 같이 고친다
            if isinstance(data.get("membership"), dict):
                mm = {}
                for grp, lst in data["membership"].items():
                    mm[grp] = [stems.get(os.path.splitext(x)[0], x) for x in lst]
                data["membership"] = mm
            json.dump(data, open(p, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=1)
        total_fixed += n_fix
        print("%-22s %4d장  수정 %2d  미해결 %d%s"
              % (fn, len(labels), n_fix, len(unresolved),
                 "  (백업 .bak 생성)" if n_fix else ""))
        for u in unresolved[:5]:
            print("     미해결:", u)

    print()
    print("총 %d장 수정." % total_fixed)
    if total_fixed:
        print("이전 측정치는 이만큼이 자동 실패로 깔린 값이므로 과소평가였다.")
        print("영향받은 단계를 다시 재야 한다 (56·65·76 등).")


if __name__ == "__main__":
    main()
