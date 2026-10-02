# 소비기한 추출 — 제3회 ITDA 연합학술제 본선

상품 뒷면 이미지에서 소비기한 날짜를 추출해 `submission.csv` 를 만든다.
팀: [DScover]_카피바라

**성능** (직접 라벨링한 홀드아웃4·5 합산 248장, 설계에 쓰지 않은 표본):

| 항목 | 값 |
| --- | --- |
| 필드 정확도 | **92.3%** (95% CI 90.2~94.0%) |
| 완전일치 | 89.5% |
| 미검출 / 오탐 | 10장 / **0건** |

**속도** (단일 프로세스 순차 실행, 모델 로딩 제외, 제한 500장 2,500초):

| 실행 환경 | 측정 | 장당 | 500장 | 제한 대비 |
| --- | --- | ---: | ---: | ---: |
| 로컬 · AMD Ryzen 5 7500F (6코어/12스레드), Windows 11 | 채점과 같은 500장 | 1.01초 | 503초 (실측) | 20% |
| GitHub Codespaces · 4 vCPU x86, RAM 16GB, Ubuntu 24.04 | `sample/` 50장 | 3.16초 | 약 1,580초 (환산) | 약 63% |

Codespaces 는 채점 환경(4코어 CPU, GPU 없음)과 비슷한 조건이라 이쪽이 채점 속도에 더
가깝다고 본다. 50장 환산이라 오차가 있고, 이 표본은 1칸에서 끝난 비율이 70%로 평소(80%)보다
낮아 다소 보수적인 값이다. 실행 로그는 [logs/](logs/) 에 있다.

설계 근거는 전부 실측이며 [docs/METHOD.md](docs/METHOD.md), [docs/JOURNAL.md](docs/JOURNAL.md),
[eda/out/](eda/out/) 에 기록되어 있다. 본선 보고서는 [docs/](docs/) 에 있다.

---

## 실행 방법 (채점 순서 그대로)

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash download_weights.sh        # 인터넷 연결 상태에서 1회만

ITDA_INPUT_DIR=./sample ITDA_OUTPUT_PATH=/tmp/out.csv \
  jupyter nbconvert --to notebook --execute predict.ipynb --output /tmp/executed.ipynb
```

`predict.ipynb` 는 **Run All 만으로 단독 실행**된다. 다른 파일을 import 하지 않는다.

### 가중치 — 노트북은 내려받지 않고 읽기만 한다

> **채점 서버가 오프라인이므로 `predict.ipynb` 안에는 다운로드 코드가 없다.**
> 모델을 받는 코드는 `download_weights.sh` 에만 있다. 노트북은 `weights/` 에 이미
> 있는 파일만 읽고, 없으면 그 단계를 건너뛴다.
> (`python audit_notebook.py` 가 이를 코드로 검증한다)

| 가중치 | 출처 | 크기 | 쓰는 단계 |
| --- | --- | ---: | --- |
| `weights/rapidocr/PP-OCRv6_det_small·rec_small`, `cls` | RapidOCR 사전학습 | 30MB | 1·2칸 (필수) |
| `weights/rapidocr/PP-OCRv6_det_medium` | RapidOCR 사전학습 | 59MB | 3·4칸 |
| `weights/rapidocr/korean·en_PP-OCRv5_rec_mobile` | RapidOCR 사전학습 | 20MB | 5·6칸 |
| `weights/dot/dotcrnn_v1.onnx` (+ `.onnx.data`) | **직접 학습** (CRNN+CTC) | 3MB | 7칸 |
| `weights/yolo/date_v1.onnx` | **직접 학습** (YOLOv8n) | 11MB | 보조 경로 |
| `weights/det·rec·cls/` (PP-OCRv3 det + 한국어 PP-OCRv4 rec) | PaddleOCR 사전학습 | 30MB | 보조 경로 |

가중치가 하나도 없어도 **노트북은 죽지 않는다.** 형식이 올바른 `submission.csv` 를
만들되 값이 전부 `NONE` 이 된다. 규정상 가중치는 Git 에 커밋하지 않고, 직접 학습한
두 모델은 Release Assets 에서 `download_weights.sh` 가 받는다.

> ⚠️ `requirements.txt` 에 **torch / ultralytics 를 넣으면 안 된다.** 추론에는 필요 없고,
> 설치돼 있는 것만으로 OpenMP 스레드 충돌로 2.4배 느려진다(500장이 제한 초과).
> 학습·ONNX 내보내기용 의존성은 [requirements-train.txt](requirements-train.txt) 로 분리했다.

### 제출 전 자가 검증

```bash
python audit_notebook.py     # 0점 유발 요인 감사 (다운로드·누락 패키지·하드코딩 경로·GPU)
python check_submission.py   # 형식·의존성·가중치·문서·구문 점검
```

---

## 환경

채점 환경(Ubuntu 22.04 / Python 3.10 / CPU 4코어 · RAM 8GB / GPU 없음 / 오프라인)을
전제로 구성했다. onnxruntime 은 `CPUExecutionProvider` 고정이고, 실행 중 네트워크를 쓰지 않는다.
`paddlepaddle` 2.6.x 가 numpy 1.x 를 요구하므로 `requirements.txt` 에 버전을 고정해 두었다.

> ⚠️ **시스템 라이브러리 `libGL.so.1` 이 필요하다.** `paddleocr` 2.7.3 이 의존성으로 화면 출력용
> `opencv-python` 을 설치하는데, 이 패키지는 리눅스에서 libGL 을 요구한다. 서버용 Ubuntu 에는
> 기본으로 없는 경우가 있어, 없으면 노트북 첫 셀의 `import cv2` 에서
> `ImportError: libGL.so.1: cannot open shared object file` 로 멈춘다. 이때는 한 번만 설치하면 된다.
>
> ```bash
> sudo apt-get install -y libgl1 libglib2.0-0
> ```
>
> 같은 원인으로 `download_weights.sh` 가 "rapidocr 가 없다" 고 잘못 안내할 수 있다(rapidocr 가
> 내부에서 cv2 를 불러오다 실패하는 것이다). Codespaces 재현 때 이 설치를 한 뒤 정상 완주했다.

---

## 파이프라인 — 계단(cascade) 구조

각 칸은 **앞 칸이 온전한 날짜를 못 낸 이미지에만** 실행된다. 맞힌 이미지는 이미
빠져나갔으므로 칸을 새로 붙여도 기존 정답이 바뀌지 않고, 비용은 실패분에만 든다.

| 단계 | 처리 | 종료 비율 |
| --- | --- | ---: |
| 1칸 | RapidOCR PP-OCRv6 det small + rec small, 긴 변 1024px | 80% |
| 2칸 | adaptive 이진화 후 재시도 | 2% |
| 3칸 | det medium (대형 검출기) | 5% |
| 4칸 | CLAHE 대비 보정 + det medium | 1% |
| 5·6칸 | 인식기 교체: 한국어, 영문 | 각 1% |
| 7칸 | 도트 인쇄 전용 판독기 (직접 학습, CRNN+CTC) | 1% |
| 보조 | YOLOv8n 날짜 영역 검출 → PP-OCRv3 det + PP-OCRv4 rec | 1% |

각 칸의 텍스트는 날짜 규칙 엔진(`src/team2_rules.py`)을 거친다.

- 주변 단어 점수: `까지` +15, `소비기한`·`유통기한` +10, `부터` −8, `제조일자`·`MFG` −12
- 동점이면 늦은 날짜, 최고점이 음수면 `NONE`
- 날짜 뒤 `:숫자`(시각) 차단, 9자리 이상 숫자열 내부 매치 차단
- 2자리 연도는 한국식 `YY.MM.DD` 기본 (라벨 대조: 한국식 91%, 유럽식 52%)
- 일부만 읽힌 결과(`NONE-08-25`)는 보관했다가, 뒤 칸이 채운 필드 수를 늘릴 때만 교체

### 기각한 것 (새 표본에서 효과가 재현되지 않음)

| 시도 | 결과 |
| --- | --- |
| 1600px 재시도 칸 · det server 칸 | 오염 표본에서 이득, 새 표본 회수 0장 |
| 전 칸 실행 후 다수결 | 93.6% → 93.4% |
| YOLO 도트 파인튜닝 | mAP50 0.80이지만 기존 정답 6장 상실 |
| 검출 문턱 하향 | 248장에서 744필드 중 1필드 차이, 표본 간 부호 반대 |
| 전처리 상시 적용 · EasyOCR 결합 | 유의하지 않음 / 상호보완 0장 |

---

## 저장소 구조

```
├── predict.ipynb          # 채점 대상 (build_notebook.py 로 생성)
├── requirements.txt       # 버전 고정
├── requirements-train.txt # 학습 전용 — 채점 환경에 설치하지 말 것
├── download_weights.sh    # 가중치 다운로드 (여기에만 다운로드 코드가 있다)
├── README.md
├── audit_notebook.py      # 0점 유발 요인 감사
├── check_submission.py    # 제출 형식 자동 점검
├── build_notebook.py      # src/*.py → predict.ipynb 생성기
├── run_local.py           # 로컬 실행·정답 대조
├── publish_release.*      # 가중치 Release 업로드
├── src/                   # 로직 원본 — 노트북에 내장되어 배포된다
│   ├── combined.py        #   계단 구조 (CombinedPipeline)
│   ├── team2_rules.py     #   날짜 판별 규칙
│   ├── dotcrnn.py · dotseg.py  # 도트 인쇄 판독기
│   ├── pipeline.py        #   보조 경로 (PP-OCR + YOLO)
│   └── rejected_rungs.py  #   기각한 칸 (근거 보존용)
├── custom_data/           # 직접 라벨링한 정답 828장과 라벨링 도구
├── docs/                  # 본선 보고서·요약서 PDF, METHOD · JOURNAL · 체크리스트, roi_retail.py
├── weights/               # 가중치 경로 (gitignore, download_weights.sh 로 채운다)
└── eda/                   # 실험 스크립트와 결과 리포트 (채점 대상 아님)
```

**`custom_data/`** — 배포 데이터 3,352장에는 정답이 없어 828장을 직접 판독했다.
`labels.csv` 의 `split` 컬럼으로 튜닝 137장, 홀드아웃1 ~ 5(48·198·197·148·100장)를 구분한다.
오답을 검토한 표본(튜닝, 홀드아웃1 ~ 3)은 성능 근거에서 제외했다. 기준과 근거는
[custom_data/README.md](custom_data/README.md) 에 있다.

**`docs/roi_retail.py`** — 보고서 6장(편의점·물류센터 도입 비용과 ROI)의 계산 스크립트.
상단 상수만 바꿔 다시 계산할 수 있다. 추론과는 무관하다.

`predict.ipynb` 를 직접 편집하지 말 것. `src/` 를 고친 뒤 다시 생성한다.

```bash
python build_notebook.py
```

---

## 출력 형식

| image_id | year | month | day | final_date |
| --- | --- | --- | --- | --- |
| 000001 | 2027 | 06 | 26 | 2027-06-26 |
| 000002 | NONE | 08 | 25 | NONE-08-25 |
| 000003 | NONE | NONE | NONE | NONE |

- `image_id` 는 확장자를 뺀 파일명 **문자열 그대로** (`000001` ≠ `1`).
- `month` / `day` 는 2자리 문자열, 미인식은 문자열 `NONE`. 읽힌 필드만 채운다.
- `to_csv(..., index=False)` — 인덱스를 쓰면 컬럼이 밀린다.

노트북 마지막 셀에서 위 조건을 `assert` 로 검증한다.
