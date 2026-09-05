# ITDA 3rd 학술제 - 소비기한 추출 제출 템플릿 📌

본 저장소는 **제3회 ITDA 연합학술제** 참가자를 위한 공식 제출 템플릿 및 환경 검증용 저장소입니다.

---

## 1. 대회 개요 및 과제 정의

- **주제**: OCR 기반 상품 소비기한 정보 추출 아키텍처 설계 및 도메인 활용 기획
- **주최**: 수도권 데이터사이언스 연합학회 ITDA (경희대 CODE, 서강대 INSIGHT, 성균관대 DScover, 인하대 IBAS, 한국외대 DAT)
- **입력 (Input)**: 상품 뒷면 이미지 (`ITDA_INPUT_DIR` 환경변수로 경로 주입)
- **출력 (Output)**: `submission.csv` (`ITDA_OUTPUT_PATH` 환경변수 경로에 저장)

### submission.csv 표준 스키마

| image_id | year | month | day | final_date |
| --- | --- | --- | --- | --- |
| 1 | 2026 | 05 | 29 | 2026-05-29 |
| 2 | NONE | NONE | NONE | NONE |

- `image_id` : 확장자를 제외한 이미지 파일명 (예: 1)
- `year` : 4자리 연도 문자열 (예: 2026, 미인식 시 NONE)
- `month` : 2자리 월 문자열 (예: 05, 미인식 시 NONE)
- `day` : 2자리 일 문자열 (예: 29, 미인식 시 NONE)
- `final_date` : 하이픈(-)으로 연결된 정규화 날짜 (예: 2026-05-29, 미인식 시 NONE)

---

## 2. 파일 및 저장소 구조

````
itda3-[학회영문]-[영문팀명]/
├── predict.ipynb            # 메인 추론 노트북 (운영진 채점용 필수)
├── requirements.txt         # 실행 환경 패키지 목록 (필수)
├── README.md                # 가중치 다운로드 및 실행 가이드 (필수)
├── .gitignore               # 가중치·데이터 커밋 방지 (수정 시 주의)
├── download_weights.sh      # [선택] 외부 가중치 다운로드 스크립트
├── notebooks/               # [선택] 실험·분석 노트북 (채점 대상 아님)
└── weights/                 # [선택] 모델 가중치 저장 폴더
````

---

## 3. 시작하기 및 실행 방법

### 1) 가상환경 구축 및 패키지 설치

````
git clone <본인 팀 저장소 URL>
cd <저장소 디렉토리>
pip install -r requirements.txt
````

### 2) 가중치 파일 설정

용량이 큰 모델 가중치 파일(`.pt`, `.pth`, `.safetensors` 등)은 Git에 직접 푸시하지 마시고, Google Drive, HuggingFace 링크 또는 Release Assets를 통해 `download_weights.sh` 스크립트 등으로 내려받도록 설정하세요.

### 3) 채점 재현성 검증 (운영진 채점 표준 명령어)

운영진은 Standard 4-Core vCPU 환경에서 아래 명령어를 실행하여 순차 실행(Run All) 및 채점을 진행합니다.

````
export ITDA_INPUT_DIR=./val_images
export ITDA_OUTPUT_PATH=./submission.csv

jupyter nbconvert --to notebook --execute predict.ipynb \
    --ExecutePreprocessor.timeout=2400 \
    --output /tmp/executed.ipynb
````

---

## 4. ⚠️ 채점 환경 필수 공지 (반드시 읽어주세요)

### 1) 팀 저장소 공개 범위

- 팀 저장소는 **Public** 으로 생성해 주세요.
- Private 으로 운영할 경우, 마감 전까지 운영진 계정 **`b9511242000-blip`** 을 Collaborator 로 초대해야 합니다. (Settings → Collaborators → Add people)
- 마감 시각 기준 운영진이 접근할 수 없는 저장소는 채점 대상에서 제외됩니다.

### 2) 채점 서버는 오프라인입니다

채점은 **인터넷이 차단된 Standard 4-Core vCPU 환경**에서 진행됩니다.

- EasyOCR, PaddleOCR 등 상당수 라이브러리는 최초 실행 시 가중치를 인터넷에서 **자동 다운로드** 합니다. 오프라인 환경에서는 이 단계가 실패해 실행 오류(정량 0점)가 발생합니다.
- 모든 가중치는 **노트북 실행 전에 로컬에 존재**해야 합니다.
  - `download_weights.sh` 는 채점 실행 **전에** 운영진이 1회 실행합니다.
  - `predict.ipynb` 의 Run All **도중에** 다운로드하는 코드는 동작하지 않습니다.

EasyOCR 사용 예시:

````python
reader = easyocr.Reader(
    ['en'], gpu=False,
    model_storage_directory='./weights',
    download_enabled=False,   # 오프라인 강제
)
````

네트워크를 끄고 Run All 이 끝까지 돌아가면 통과입니다. 제출 전 반드시 한 번 검증해 보세요.

### 3) 환경 설치 시간은 속도 점수에 포함되지 않습니다

- `pip install -r requirements.txt` 및 `download_weights.sh` 소요 시간은 속도 점수(10점) 산정에서 **제외** 됩니다.
- 속도 점수는 `predict.ipynb` 의 Run All 실행 시간(최대 2400초)만으로 산정합니다.

---

## 5. 제출 전 필수 체크리스트

1. **CONFIG 셀 수정 금지**: `predict.ipynb` 최상단의 환경변수 주입 코드는 절대 변경하거나 값을 직접 하드코딩 대입하지 마세요.
2. **대화형 코드 제거**: 실행 중 사용자 입력을 대기하는 코드(`input()`, `getpass()` 등)가 있으면 실행이 중단되어 정량 0점 처리됩니다.
3. **인덱스 제외 저장**: CSV 저장 시 반드시 인덱스를 제외해야 합니다. (`df.to_csv(OUTPUT_PATH, index=False)`)
4. **결과 스키마 준수**: 누락된 컬럼이 없도록 `image_id, year, month, day, final_date` 5개 컬럼 스키마를 엄격히 지켜주세요.
5. **오프라인 실행 검증**: 네트워크 차단 상태에서 Run All 이 완주하는지 확인하세요.
6. **저장소 접근 권한**: Public 설정 또는 운영진 계정 Collaborator 초대를 완료하세요.
````
````
