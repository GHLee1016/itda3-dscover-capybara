# 제출본 점검 및 보완 기록

대회 설명서(제3회 ITDA 연합학술제 참가자 안내서)를 기준으로 2026-09-08에 점검했습니다.

## 반영한 필수 수정

- `predict.ipynb`의 **첫 번째 셀을 설정 코드 셀**로 변경했습니다. 이전에는 제목 Markdown 셀이 앞에 있어, 안내서의 “첫 번째 코드셀” 규정을 엄격하게 해석하는 자동 검수에서 문제가 될 여지가 있었습니다.
- 설정 셀은 `ITDA_INPUT_DIR`와 `ITDA_OUTPUT_PATH`를 `os.environ.get`으로 그대로 읽습니다.
- 날짜 유효성 검사를 달력 기준으로 바꿨습니다. 따라서 `2026-02-30`, `2026-04-31` 같은 OCR 오인식은 후보에서 제외됩니다.
- 이미지당 OCR 후보를 34개에서 16개로 줄이고, EasyOCR 인식을 배치 처리합니다. 실패 시 전체 이미지를 다시 검출하는 대신 상위 6개 후보 ROI만 CLAHE 대비 보정해 재인식합니다.
- 시간의 78%를 사용한 뒤에는 대비 보정 재시도만 생략하고 fast-path는 끝까지 수행합니다. 따라서 막판 이미지가 통째로 `NONE` 처리되는 비율을 낮춥니다.
- 입력 탐색에 BMP/WEBP 확장자를 추가했습니다. JPEG/PNG가 아닌 평가 이미지도 누락되지 않습니다.

## 대회 규격 대조 결과

| 항목 | 상태 |
| --- | --- |
| `predict.ipynb` 첫 설정 셀 | 충족 |
| 환경변수 입력/출력 경로 사용 | 충족 |
| 출력 스키마 5개 컬럼, `index=False` | 충족 |
| CPU 전용, 추론 중 네트워크 호출 없음 | 충족 (`gpu=False`, `download_enabled=False`) |
| `requirements.txt`에 `nbconvert`, `ipykernel` 명시 | 충족 |
| 2,400초 내 실제 전체 검증셋 Run All | 별도 실행 필요 |

## 제출 직전 필수 확인

`download_weights.sh`로 EasyOCR 가중치 두 개를 `weights/`에 먼저 준비한 뒤, 인터넷을 끈 상태에서 아래처럼 실행하세요. 가중치 파일은 Git에 커밋하지 않습니다.

```bash
export ITDA_INPUT_DIR=/eval/val_images
export ITDA_OUTPUT_PATH=/tmp/submission.csv
jupyter nbconvert --to notebook --execute predict.ipynb \
  --ExecutePreprocessor.timeout=2400 \
  --output /tmp/executed.ipynb
```

그 다음 `/tmp/submission.csv`의 행 수가 입력 이미지 수와 같고, 헤더가 정확히 `image_id,year,month,day,final_date`인지 확인하세요.

> 이 작업 환경에는 2GB 이미지 아카이브 전체와 EasyOCR 가중치를 동시에 풀어 실제 OCR 성능/완주 시간을 재현하지 않았습니다. 위 오프라인 Run All 검증이 최종 제출 전 반드시 필요합니다.
