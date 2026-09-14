# 소비기한 추출 — 제3회 ITDA 연합학술제

상품 뒷면 이미지에서 소비기한 날짜를 추출해 `submission.csv` 를 생성한다.

**성능** (직접 라벨링한 홀드아웃 198장, 설계에 쓰지 않은 표본):

| 구성 | 부분점수 | 완전일치 | 장당 | 500장 환산 |
| --- | ---: | ---: | ---: | ---: |
| 보조 경로 단독 (PP-OCRv3/v4 + YOLO) | 66.8% | 59.1% | 2.19초 | 1,096초 |
| 주력 경로 단독 (RapidOCR PP-OCRv6) | 79.8% | 75.8% | 0.37초 | 184초 |
| **둘을 합침 (제출 구성)** | **82.3%** | **77.8%** | **0.53초** | **267초** |

제한 2,400초의 **11%** 만 쓴다. 설계 근거는 전부 실측이며
[METHOD.md](METHOD.md) 와 [eda/out/](eda/out/) 에 기록되어 있다.

---

## 실행 방법 (채점 순서 그대로)

```bash
# 1) 환경 구축 — Python 3.10
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# 2) 가중치 준비 — 인터넷 연결 상태에서 1회만
bash download_weights.sh

# 3) 이후는 오프라인에서 실행 가능
export ITDA_INPUT_DIR=/eval/val_images
export ITDA_OUTPUT_PATH=/eval/submission.csv

jupyter nbconvert --to notebook --execute predict.ipynb \
    --ExecutePreprocessor.timeout=2400 \
    --output /tmp/executed.ipynb
```

`predict.ipynb` 는 **Run All 만으로 단독 실행**된다. 다른 파일을 import 하지 않는다.

### 가중치 — 노트북은 내려받지 않고 읽기만 한다

> **채점 서버가 오프라인이므로, `predict.ipynb` 안에는 다운로드 코드가 전혀 없다.**
> 모델을 받는 코드는 `download_weights.sh` 에만 있다. 노트북은 `weights/` 에 이미
> 있는 파일만 읽고, 없으면 그 엔진을 건너뛴다 — 네트워크에 나가지 않는다.
> (`python audit_notebook.py` 가 이 사실을 코드로 검증한다)

| 가중치 | 출처 | 크기 | 없을 때 |
| --- | --- | ---: | --- |
| `weights/rapidocr/*.onnx` (PP-OCRv6 det/rec/cls) | RapidOCR 사전학습 | 30MB | **주력 경로 비활성** → 보조 경로만 (66.8%) |
| `weights/det·rec·cls/…` (PP-OCRv3 det + 한국어 PP-OCRv4 rec) | PaddleOCR 사전학습 | 30MB | 보조 경로 비활성 → 주력만 (79.8%) |
| `weights/yolo/date_v1.onnx` | **직접 학습**(YOLOv8n 파인튜닝) | 11MB | 보조 경로가 det 단독으로 동작 |

셋 다 없어도 **노트북은 죽지 않는다.** 형식이 올바른 `submission.csv` 를 만들되
값이 전부 `NONE` 이 된다(검증 완료). 규정상 가중치는 Git 에 커밋하지 않고
Release Assets 로 배포하며, 직접 학습한 YOLO 는 `download_weights.sh` 의
`YOLO_URL`(또는 `ITDA_YOLO_URL` 환경변수)로 받는다.

> ⚠️ `requirements.txt` 에 **torch / ultralytics 를 넣으면 안 된다.** 추론에는 필요 없고,
> 임포트되는 것만으로 파이프라인이 2.4배 느려진다(장당 2.76 → 5.88초, 500장이 제한 초과).
> paddle 과 torch 가 각자 OpenMP 런타임을 싣고 스레드를 다투기 때문이다.
> 학습·ONNX 내보내기용 의존성은 [requirements-train.txt](requirements-train.txt) 로 분리했다.

### 제출 전 자가 검증

```bash
python audit_notebook.py     # 0점 유발 4대 원인 감사 (다운로드·누락패키지·하드코딩경로·GPU)
python check_submission.py   # 형식·의존성·가중치·PDF·구문 48개 항목
```

---

## 환경

채점 환경(Ubuntu 22.04 / Python 3.10 / CPU 4코어 · RAM 8GB / GPU 미제공 / 오프라인)을
전제로 구성했다. GPU 코드는 없고(`use_gpu=False`, onnxruntime 은 `CPUExecutionProvider` 고정),
실행 중 네트워크를 쓰지 않는다.

버전 조합에 두 가지 함정이 있다(실측으로 확인).

- `paddlepaddle` 2.6.x 는 **numpy 1.x** 를 요구한다. numpy 2.x 면
  `numpy.core.multiarray failed to import` 로 import 단계에서 죽는다.
- opencv 는 **한 계열만** 설치한다. `opencv-python` / `opencv-contrib-python` /
  `opencv-python-headless` 를 섞으면 서로 다른 numpy ABI 로 빌드된 `cv2` 가 충돌한다.

`requirements.txt` 에 해당 버전을 고정해 두었다.

---

## 파이프라인

```
━━ 주력 경로 (이미지의 약 90% 를 여기서 끝낸다) ━━━━━━━━━━━━━━━━━━━━━━
[1] 긴 변 1024px 축소
[2] RapidOCR      PP-OCRv6 det + rec (ONNX)
[3] 날짜 판별     인접 OCR 조각 2~3개 이어붙여 후보 생성
                  소비기한/유통기한/EXP 가점 · 제조일자/포장일/MFG 감점
                  완전 날짜 → 부분 날짜(연·월만 등) 순으로 내려감
      └ 실패 → adaptive threshold 로 1회 재시도

━━ 보조 경로 (주력이 아무것도 못 읽은 이미지에만) ━━━━━━━━━━━━━━━━━━━
[4] 긴 변 640px   ← 960px 보다 빠르면서 더 정확
[5] det (0°)      PP-OCRv3 다국어 검출
[5b] 날짜 검출    YOLOv8n(ONNX)이 날짜 영역을 따로 제안 ← det 가 놓친 곳 보충
[6] rec           한국어 PP-OCRv4 인식
[7] 박스 병합     같은 행·인접 박스 텍스트 결합       ← 비용 0, 정확도 48%→68%
[8] 날짜 파싱     구분자 소실 허용 정규식
      └ 못 찾으면 → 90° 회전 → 512·448·384px → 전처리 3종 (적응형)

[9] 출력          image_id / year / month / day / final_date
```

### 설계 판단의 근거

| 결정 | 근거 (실측) |
| --- | --- |
| det 입력 960px 고정 | 3200px 까지 올려도 recall 불변, 비용만 1.7→2.8초 |
| 회전은 합집합, 판별기 없음 | 판별기는 +0.2%p 뿐, 합집합은 +11.6%p |
| 인접 박스 병합 | det 가 `2022/11/04` 를 `20221`+`104` 로 쪼갠다. 정확도 **48%→68%** |
| 7자리 느슨 패턴 **미채택** | coverage 는 +12%p 지만 정확도는 48%→48% (후보만 늘리는 노이즈) |
| 단일 전처리 **미채택** | 13종 실험, n=400 에서 +3.0%p, McNemar p=0.201 |
| 4자리 연도 우선 | 정밀도 99.2% vs 2자리 연도 79.4% |
| 동점이면 늦은 날짜 | 제조/소비 쌍 사례에서 전부 성립 |
| 미래성 필터 금지 | 정답에 2020~2022년 과거 날짜가 다수 |

---

## 저장소 구조

```
├── predict.ipynb          # 채점 대상 (build_notebook.py 로 생성)
├── requirements.txt       # 버전 고정
├── requirements-train.txt # 학습 전용 — 채점 환경에 설치하지 말 것
├── download_weights.sh    # 가중치 다운로드 (여기에만 다운로드 코드가 있다)
├── .gitignore             # 가중치·데이터 커밋 방지
├── README.md
├── audit_notebook.py      # 0점 유발 요인 감사
├── check_submission.py    # 제출 형식 자동 점검
├── build_notebook.py      # src/*.py → predict.ipynb 생성기
├── custom_data/           # ★ 가산점 — 직접 라벨링한 정답 383장
│   ├── labels.csv         #   통합 라벨 (image_id, year, month, day, final_date, split)
│   ├── labels_*.json      #   집합별 원본 (튜닝 137 / 홀드아웃1 48 / 홀드아웃2 198)
│   ├── README.md          #   라벨링 기준과 분할 근거
│   └── tool/              #   직접 만든 라벨링 도구
├── weights/               # 가중치 저장 경로 (gitignore, download_weights.sh 로 채운다)
├── src/                   # 로직 원본 — 노트북에 내장되어 배포된다
│   ├── combined.py        #   주력/보조 전환 층
│   ├── team2_rules.py     #   날짜 판별 규칙
│   └── pipeline.py        #   보조 경로 (PP-OCR + YOLO)
├── METHOD.md  JOURNAL.md  WHERE.md  SUBMISSION_CHECKLIST.md  RULES_QNA.md
└── eda/                   # 실험 스크립트 01~67 + 리포트 (채점 대상 아님)
    └── out/
```

**`custom_data/`** — 배포 데이터 3,352장에는 정답이 없어 직접 판독·검수한 **383장**을 만들었다.
`labels.csv` 가 통합 라벨이고(`split` 컬럼으로 튜닝/홀드아웃 구분), 기준과 근거는
[custom_data/README.md](custom_data/README.md) 에 있다.

`predict.ipynb` 를 직접 편집하지 말 것. `src/` 를 고친 뒤 다시 생성한다.

```bash
python build_notebook.py
```

---

## 재현

```bash
# EDA (이미지 압축 해제 필요)
python eda/01_metadata.py
python eda/02_structure.py
python eda/03_ocr_sample.py --n 500 --workers 4
python eda/04_analyze.py

# 파이프라인 정확도·속도 검증
python eda/20_pipeline_eval.py --speed-n 150 --workers 4
```

라벨링 도구 (`custom_data/tool/`):

```bash
python custom_data/tool/make_tasks.py --n 250 --hard 50 --shards 3
cd custom_data/tool && python -m http.server 8000
# → http://localhost:8000/index.html?shard=1
```

---

## 출력 형식

| image_id | year | month | day | final_date |
| --- | --- | --- | --- | --- |
| 000001 | 2027 | 06 | 26 | 2027-06-26 |
| 000002 | NONE | NONE | NONE | NONE |

- `image_id` 는 확장자를 뺀 파일명 **문자열 그대로**. `000001` 을 `int` 로 바꾸면
  `1` 이 되어 전량 불일치한다.
- `month` / `day` 는 2자리 zero-padding 문자열, 미인식은 문자열 `NONE`.
- `to_csv(..., index=False)` — 인덱스를 쓰면 컬럼이 밀려 파싱 에러가 난다.

노트북 마지막 셀에서 위 조건을 `assert` 로 검증한다.
