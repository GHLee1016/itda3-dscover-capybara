# 뭐가 어디에 있나

프로젝트 루트: `C:\Users\조영우\Desktop\대학\DScover\12기\연합 학술제`

---

## 1. 데이터셋 — 직접 보려면

### 원본 3,352장

```
images\상품사진입니다\        ← 2.00 GB, 3,352장
```

탐색기에서 바로 열면 된다:

```powershell
explorer "$PWD\images\상품사진입니다"
```

- `000001.jpg` ~ `003343.jpg` (6자리 zero-padding)
- `3344.jpeg` ~ `3352.jpeg` (패딩 없음, 9장)
- 원본 압축파일은 루트의 `images.zip` (1.99 GB)

> 큰 파일이 많아(최대 11 MB) 탐색기 미리보기가 느릴 수 있다.
> 그럴 땐 아래 축소본을 보는 편이 빠르다.

### 축소본 399장 (보기 편함)

```
label\imgs\                   ← 긴 변 1500px, 장당 ~200 KB
```

라벨링용으로 만든 것이라 날짜가 읽힐 만큼은 선명하다.
JOURNAL.md 에 나오는 사례 이미지는 대부분 여기 있다.

### 라벨링 도구로 보기 (추천)

후보 날짜와 앵커가 함께 표시되고, 휠 확대·드래그 이동이 된다.

```powershell
cd label
python -m http.server 8000
# 브라우저에서 http://localhost:8000/index.html
```

- 숫자키 `1`~`8` 로 후보 선택, `0` 은 NONE
- `←` `→` 이동, `F` 전체보기, `R` 90° 회전
- 우측 상단 「JSON 내려받기」로 `truth_labeled.json` 저장
- 팀원 3명이 나눠 할 거면 `?shard=1` `?shard=2` `?shard=3`

---

## 2. 문서 — 읽는 순서

| 파일 | 내용 |
| --- | --- |
| **[JOURNAL.md](JOURNAL.md)** | **전 과정 상세 기록. 여기부터** |
| [METHOD.md](METHOD.md) | 설계 결론 요약 |
| [README.md](README.md) | 실행 방법·환경 구축 |
| [SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md) | 제출 전 점검 항목 |
| [eda/out/00_SUMMARY.md](eda/out/00_SUMMARY.md) | 데이터 EDA 요약 |

---

## 3. 제출물

### 메일 첨부

```
[DScover]_카피바라_아키텍처구조도.pdf   ← A4 2장. 요약서.html 이 원본
요약서.html                            ← 내용 수정은 여기서
build_pdf.ps1                          ← 요약서.html → PDF (헤드리스 Edge)
```

### GitHub 저장소

```
predict.ipynb                 ← 채점 대상 (Run All 단독 실행, 15셀)
requirements.txt              ← 버전 고정 (numpy/opencv 충돌 주의)
requirements-train.txt        ← 학습 전용. 채점 환경에 설치하지 말 것
download_weights.sh           ← 가중치 사전 다운로드
publish_release.ps1           ← 저장소 생성 + 가중치 Release 업로드 자동화
build_notebook.py             ← src\*.py → predict.ipynb 생성기

src\combined.py               ← 두 경로를 잇는 층 (주력/보조 전환 로직)
src\team2_rules.py            ← 날짜 판별 규칙 (팀원2 박안젤라 원문 그대로)
src\pipeline.py               ← 보조 경로 (PP-OCR + YOLO). 분량의 대부분

weights\rapidocr\*.onnx       ← 주력 엔진 PP-OCRv6 det/rec/cls (30MB)
weights\yolo\date_v1.onnx     ← 직접 학습한 날짜 검출기 (11MB)
```

> `predict.ipynb` 를 직접 편집하지 말 것.
> `src\` 를 수정 → `python build_notebook.py` 로 재생성.

### 구조 한눈에

```
RapidOCR(PP-OCRv6) → 날짜 판별 → 실패 시 adaptive threshold 재시도
                                  → 그래도 실패한 이미지에만 PP-OCRv3/v4 + YOLO
```

| | 부분점수 | 장당 | 500장 |
| --- | ---: | ---: | ---: |
| 주력 단독 | 79.8% | 0.37초 | 184초 |
| 보조 단독 | 66.8% | 2.19초 | 1,096초 |
| **합친 것 (제출)** | **82.3%** | **0.53초** | **267초 (제한의 11%)** |

---

## 4. 실험 결과

```
eda\*.py                      ← 실험 스크립트 01~66
eda\out\*.md                  ← 단계별 리포트 (JOURNAL.md §14 에 색인)
eda\out\*.csv, *.jsonl        ← 원자료
```

### 결론이 바뀐 지점만 보려면

| 파일 | 내용 |
| --- | --- |
| `eda\out\44_success_vs_fail.md` | 실패 원인은 화질이 아니라 **검출 누락**이었다 |
| `eda\out\52_omp_conflict.md` | torch 를 임포트하기만 해도 2.4배 느려진다 |
| `eda\out\59_merge_eval.md` | 팀원1(EasyOCR) 결합은 **+0.0%p** — 기각 |
| `eda\out\64_merge3_eval.md` | 세 파이프라인 비교 + 합치기 4방식 |
| `eda\out\65_combined_eval.md` | **최종 82.3% 실측** |
| `eda\out\66_combined_speed.md` | 최종 속도 0.53초/장 |

### 자주 볼 만한 원자료

| 파일 | 내용 |
| --- | --- |
| `eda\out\truth_holdout2.json` | **최종 판정에 쓴 정답 198장** |
| `eda\out\truth_v4.json` | 튜닝셋 정답 137장 |
| `eda\out\metadata.csv` | 3,352장 전체 메타데이터 (해상도·EXIF·품질) |
| `eda\out\ocr_sample.jsonl` | 표본 500장의 OCR 원문 (박스 좌표 포함) |
| `eda\out\ocr_analysis.csv` | 앵커·날짜후보·오탐원 분석 결과 |
| `eda\out\boxfeat.csv` | 박스 26,499개의 특징 + rec 결과 |

CSV 는 엑셀로 열면 된다 (UTF-8 BOM 으로 저장해 한글이 깨지지 않는다).

---

## 5. 채점 시뮬레이션

로컬 한글 경로에서는 PaddleOCR 이 모델을 못 여는 문제가 있어,
채점 환경(ASCII 경로)을 따로 만들어 검증한다. 매번 새로 만들고 끝나면 지운다.

```powershell
$T = "C:\Users\Public\itda_test"
mkdir $T, "$T\val_images", "$T\weights"
copy predict.ipynb $T
copy C:\Users\Public\ocr_models\* "$T\weights\" -Recurse   # det/rec/cls
copy weights\yolo "$T\weights\yolo" -Recurse
copy weights\rapidocr "$T\weights\rapidocr" -Recurse
# val_images 에 테스트 이미지를 넣고

$env:ITDA_INPUT_DIR  = "$T\val_images"
$env:ITDA_OUTPUT_PATH = "$T\submission.csv"
cd $T
& (Resolve-Path ..\..\..\.venv\Scripts\python.exe) -m nbconvert --to notebook `
    --execute --inplace --ExecutePreprocessor.timeout=2400 predict.ipynb
```

**최근 실측(2026-09-14, 홀드아웃2 이미지 188장)**

```
경로별 처리: {rapid: 167, adaptive: 3, fallback: 6, none: 12}
done: 188 rows in 163s (0.867s/img)
부분점수 86.5%
```

실행이 끝난 `predict.ipynb` 에는 셀 출력이 남으므로, 경로 분담과 시간 예산
사다리 동작을 로그로 확인할 수 있다.

---

## 6. 실행 환경

```
.venv\                        ← Python 3.10 (채점 환경과 동일 버전). 이것만 쓰면 된다
.venv-team\                   ← [삭제 가능] 팀원1(EasyOCR) 비교 측정용 격리 환경
.venv-team2\                  ← [삭제 가능] 팀원2(RapidOCR) 비교 측정용 격리 환경
```

> 팀원 환경을 따로 만든 이유: easyocr 는 torch 와 `opencv-python-headless` 를,
> rapidocr 는 `opencv-python-headless 5.x` 를 끌고 온다. 같은 환경에 섞으면
> 고정해 둔 cv2 4.10.0.84 가 깨진다 (실제로 한 번 깨뜨렸고, 복구 후
> 홀드아웃 66.8% 재현으로 무결성을 확인했다 — EDA 57).
> 비교가 끝났으므로 두 폴더는 지워도 된다 (합계 약 1.8 GB).

```powershell
# 아무 스크립트나 실행할 때
.venv\Scripts\python.exe eda\01_metadata.py
```

전체 재구축이 필요하면 [README.md](README.md) 의 "환경" 절을 따른다.

> ⚠️ 이 PC 한정 이슈: 경로에 한글이 있어 `cv2.imread` 와 PaddleOCR 모델 로딩이
> 실패한다. 코드는 `np.fromfile + imdecode` 로 우회하고, 모델은
> `C:\Users\Public\ocr_models` 에 복사해 쓴다. 채점 서버는 ASCII 경로라 무관하다.

---

## 7. Git 에 올릴 때

`.gitignore` 에 이미 등록되어 있다.

| 제외되는 것 | 이유 |
| --- | --- |
| `images\`, `images.zip` | 2 GB |
| `label\imgs\` | 축소본도 80 MB |
| `.venv\`, `.venv-team\`, `.venv-team2\` | 환경 (합계 3.4 GB) |
| `eda\out\_probe_cache.pkl`, `ocr_sample.jsonl` | 용량 |
| `weights\`, `*.pt`, `*.onnx` 등 | **규정상 Git 직접 커밋 금지** |
| `yolo_runs\`, `yolo_ds\` | 학습 산출물. 재현 절차는 `requirements-train.txt` 주석에 |
| `팀원 기록\` | 팀원 각자의 저장소가 따로 있다 |
| `오답사진\`, `yolo_시각화\` | 분석용 이미지 사본 |

가중치는 `download_weights.sh` 로 받게 되어 있고,
직접 학습한 YOLO 는 `publish_release.ps1` 로 Release Assets 에 올린다.
