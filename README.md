# ITDA 3rd 학술제 제출 저장소 - DScover_카피바라

OCR 기반 상품 소비기한 정보 추출 아키텍처. 자세한 설계 논리는 `[DScover_카피바라]_아키텍처구조도.pdf` (이메일 제출본)를 참고하세요.

## 1. 요약

- **입력**: 상품 뒷면 이미지 (`ITDA_INPUT_DIR` 환경변수로 경로 주입)
- **출력**: `submission.csv` (`ITDA_OUTPUT_PATH` 환경변수 경로에 저장)
- **엔진**: EasyOCR (`ko`+`en`, CPU, 오프라인) + 정규식/키워드 기반 날짜 후보 스코어링
- **핵심 설계**: 1) classical CV 기반 초저비용 텍스트 라인 후보 검출 → 실패 시 2) CLAHE 대비보정 + EasyOCR 전체 파이프라인 fallback, 3) 4-core 병렬처리, 4) 시간 예산 가드레일로 타임아웃 원천 차단

## 2. 실행 방법

### 1) 환경 구축

```bash
git clone <본인 팀 저장소 URL>
cd itda3-dscover-capybara
pip install -r requirements.txt
```

macOS 사용자는 `requirements.txt` 상단의 `+cpu` 접미사를 제거하고 설치하세요.

### 2) 가중치 다운로드 (인터넷 필요, 1회)

```bash
bash download_weights.sh
```

`weights/craft_mlt_25k.pth`, `weights/korean_g2.pth` 두 파일을 GitHub Release Assets(공식 EasyOCR 저장소)에서 내려받습니다. 이 스크립트는 predict.ipynb 실행 **전에** 1회만 실행하면 되며, predict.ipynb 자체는 `download_enabled=False` 로 동작하여 실행 중 어떤 네트워크 요청도 하지 않습니다.

### 3) 채점 재현성 검증 (운영진 표준 명령어)

```bash
export ITDA_INPUT_DIR=./val_images
export ITDA_OUTPUT_PATH=./submission.csv

jupyter nbconvert --to notebook --execute predict.ipynb \
    --ExecutePreprocessor.timeout=2400 \
    --output /tmp/executed.ipynb
```

오프라인(네트워크 차단) 상태에서도 위 명령이 에러 없이 완료되고 `submission.csv` 가 생성되는지 반드시 확인하세요.

## 3. 아키텍처 설계 논리 (요약)

상품 뒷면에는 소비기한 외에도 제조일자, 바코드, 품목보고번호, 전화번호, 영양성분, LOT 번호 등
날짜와 혼동될 수 있는 숫자가 다수 존재한다. 동시에 채점 환경은 CPU 전용·오프라인·2400초
타임아웃이라는 강한 제약을 가진다. 실험 결과, 이미지를 크게 축소하면(예: 500px대) 임베로/저대비
각인 숫자(병뚜껑 잉크젯 각인 등)의 인식률이 급격히 떨어지고, 반대로 고해상도(800px+)로 전체
이미지를 딥러닝 OCR(CRAFT 검출기)에 넣으면 연산량이 초선형으로 증가해 3,000장 이상 처리 시
시간 예산을 초과한다.

이에 다음과 같은 2단계 캐스케이드 구조를 채택했다.

1. **1단계 (classical CV, ms 단위)**: 형태학적 gradient + Otsu 이진화 + 가로 방향 전용 morphological
   closing 으로 텍스트 "줄" 후보 박스를 찾는다 (세로 방향 커널을 최소화하여 서로 다른 줄이 섞이는
   것을 방지). 문단형 긴 설명문(포장재질, 보관방법 등)은 폭 비율로 제외한다. 이후 EasyOCR의
   **검출기(CRAFT)는 생략**하고 인식기(recognizer)만 원본 해상도 그대로 호출하므로, 라벨이 깨끗한
   사진에서는 1초 내외로 소비기한을 정확히 찾아낸다.
2. **2단계 (fallback)**: 1단계에서 유효한 날짜 형식 후보를 찾지 못한 경우(배경이 복잡하거나 손에
   들고 찍은 사진 등)에만, CLAHE로 대비를 보정한 뒤 EasyOCR 전체 파이프라인(검출+인식)을
   축소 해상도(350px)로 1회 추가 실행한다. (원래 550px이었으나, 실제 4-core 환경에서 측정한 결과
   전체 데이터셋의 상당 부분이 시간 예산 안에 처리되지 못하는 문제가 확인되어, 장당 정확도를
   일부 희생하고 커버리지를 크게 늘리는 방향으로 재조정했다.)
3. **후보 스코어링**: 정규식(`YYYY.MM.DD`, `YYYY-MM-DD`, `YYYY년MM월DD일`, 콤마 오인식 보정,
   구분자 없는 `YYYYMMDD` 등)으로 날짜 형식 후보를 모두 추출한 뒤, 같은 줄/인접 줄에 `소비기한`,
   `유통기한`, `까지` 가 있으면 가점, `제조일자`, `제조`, `LOT` 가 있으면 감점하여 최고 점수 후보를
   최종 날짜로 채택한다. 서로 다른 박스로 쪼개진 날짜 조각(예: `2022` / `02.14` / `16:01`)은 좌표
   기반으로 이어붙여(stitching) 재시도한다.
4. **4-core 병렬 처리**: `concurrent.futures.ThreadPoolExecutor` 로 이미지 단위 작업을 워커
   스레드에 분배한다. 각 스레드는 최초 호출 시 자신만의 EasyOCR Reader 를 한 번만 로드해
   재사용하며, 프로세스 전체의 BLAS/torch 스레드 수를 1로 고정해(oversubscription 방지)
   워커 수만큼만 코어를 활용한다. EasyOCR/torch/OpenCV 연산 대부분은 C/C++ 구현이라 GIL을
   해제하므로 스레드 기반으로도 실질적인 병렬 처리량을 얻는다. `multiprocessing.Pool(fork)`
   는 개발 초기에 사용했으나, Jupyter 커널(ipykernel)이 이미 여러 백그라운드 스레드를 띄운
   상태에서 `fork()` 하면 자식 프로세스가 락 획득에 실패해 멈추는 사례가 있어(대표 증상: 여러
   번 실행해도 동일한 지점에서 이후 이미지가 전부 `NONE` 으로 채워짐) 스레드 기반으로 교체했다.
5. **시간 예산 가드레일**: 전체 경과 시간이 예산(기본 1800초, nbconvert 타임아웃 2400초 대비 여유
   600초)을 초과하면 이후 이미지는 즉시 `NONE` 으로 안전 처리하여, 어떤 경우에도 정량 0점으로
   이어지는 타임아웃이 발생하지 않도록 했다.
6. **정밀도 우선(NONE 허용) 전략**: 키워드/형식 근거가 불충분한 경우 억지로 추정하기보다 `NONE`
   을 반환하도록 설계했다. 잘못된 날짜를 자신 있게 제시하는 것보다, 근거가 있는 경우에만 값을
   내는 편이 더 신뢰할 수 있는 시스템이라고 판단했기 때문이다.

### 알려진 한계

- 병뚜껑/캡에 잉크젯으로 각인된 저대비·곡면 인쇄 날짜(예: `2026.07.02`)는 해상도를 높여도
  인식률이 낮다. CLAHE 대비 보정을 적용했지만 완전히 해결되지는 않았다.
- 키워드(`소비기한`/`까지`) 없이 날짜만 단독으로 인쇄된 경우, 다른 숫자(로트번호 등)와 형식이
  같으면 오탐 가능성이 있다. 정규식의 유효 연도 범위(2020~2035) 검증으로 일부를 방어한다.
- 팀에서 추가로 라벨링한 데이터로 미세조정하거나, 바코드 위치 기반 ROI 추정 등을 추가하면
  재현율을 더 높일 수 있다 (향후 개선 방향).

## 4. 저장소 구조

```
itda3-dscover-capybara/
├── predict.ipynb          # 메인 추론 노트북 (채점 대상)
├── requirements.txt
├── README.md
├── download_weights.sh
├── .gitignore
├── notebooks/              # 실험용 노트북 (채점 대상 아님)
└── weights/                 # download_weights.sh 로 받는 가중치 폴더
```

## 5. 제출 전 체크리스트

- [x] CONFIG 셀 미변경 (`ITDA_INPUT_DIR` / `ITDA_OUTPUT_PATH` 환경변수 그대로 사용)
- [x] `input()`, `getpass()` 등 대화형 코드 없음
- [x] `df.to_csv(OUTPUT_PATH, index=False)` — 인덱스 제외 저장
- [x] `image_id, year, month, day, final_date` 5개 컬럼 스키마 준수
- [ ] 오프라인(네트워크 차단) 상태에서 Run All 완주 확인 (제출 전 필수)
- [ ] 저장소 Public 설정 또는 `b9511242000-blip` Collaborator 초대
