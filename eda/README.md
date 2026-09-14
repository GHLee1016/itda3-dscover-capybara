# EDA — 실행 방법

## 환경

프로젝트 루트에 Python 3.10 venv (채점 환경과 동일 버전).

```powershell
uv venv --python 3.10 .venv
uv pip install --python .venv\Scripts\python.exe `
  "numpy==1.26.4" "opencv-python==4.10.0.84" "opencv-contrib-python==4.10.0.84" `
  pandas pillow matplotlib tqdm imagehash setuptools `
  "paddlepaddle==2.6.2" "paddleocr==2.7.3"
```

> ⚠️ 버전 고정 이유: `opencv-python-headless` 5.x 는 numpy 2.x 로 빌드되어 있고
> paddlepaddle 2.6.2 는 numpy 1.x 를 요구한다. 섞으면 `numpy.core.multiarray failed to import` 로 죽는다.
> opencv 는 **한 종류만** 설치할 것 (headless/contrib/일반 3종 동시 설치가 충돌 원인이었다).

### 로컬 한글 경로 우회

프로젝트 경로에 한글이 있으면 OpenCV 와 Paddle 의 C++ 파일 로더가 파일을 열지 못한다.
채점 서버(`/eval/val_images`)에서는 발생하지 않는 로컬 전용 이슈다.

- 이미지: `cv2.imread(path)` 대신 `cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)`
- OCR 모델: ASCII 경로로 복사 후 명시적으로 지정

```powershell
Copy-Item "$env:USERPROFILE\.paddleocr\whl\*" "C:\Users\Public\ocr_models" -Recurse -Force
```

## 데이터 준비

```powershell
Expand-Archive images.zip -DestinationPath images    # → images\상품사진입니다\ (3,352장)
```

## 실행 순서

```powershell
$py = ".venv\Scripts\python.exe"

& $py eda\01_metadata.py                        # 전수 메타데이터  → 01_report.md, metadata.csv
& $py eda\02_structure.py                       # 디코딩/구조 점검 → 02_report.md, structure.csv
& $py eda\03_ocr_sample.py --n 500 --workers 4  # 표본 OCR (약 14분) → ocr_sample.jsonl
& $py eda\04_analyze.py                         # 내용 분석        → 04_report.md, ocr_analysis.csv
```

`03` 은 이미지당 0°/90° 두 방향으로 OCR 하여 원문 라인을 그대로 저장한다.
`04` 는 OCR 을 다시 돌리지 않고 jsonl 만 재해석하므로, 파싱 규칙을 바꿔가며 빠르게 반복할 수 있다.

## 파일

| 파일 | 내용 |
| --- | --- |
| `01_metadata.py` | 해상도·EXIF·품질·중복(phash) 전수 조사 |
| `02_structure.py` | cv2/PIL 디코딩 호환성, MPO, EXIF 회전, 레터박싱 |
| `03_ocr_sample.py` | 층화 표본 OCR 실행 (양방향, 원문 보존) |
| `04_analyze.py` | 앵커·날짜형식·오탐원·난이도 등급 분석 |
| `datex.py` | 날짜 후보 추출 / 앵커 탐지 — **추론 파이프라인에서 재사용** |

## 결과

- **[out/00_SUMMARY.md](out/00_SUMMARY.md) ← 종합 요약. 여기부터 읽을 것**
- `out/01_report.md` `out/02_report.md` `out/04_report.md` — 단계별 상세
- `out/metadata.csv` `out/structure.csv` `out/ocr_analysis.csv` — 원자료

`out/_probe_cache.pkl` 과 `out/ocr_sample.jsonl` 은 용량이 커서 `.gitignore` 에 등록되어 있다.
`01` 은 캐시가 있으면 재스캔하지 않으므로, 스캔을 다시 하려면 캐시 파일을 지운다.
