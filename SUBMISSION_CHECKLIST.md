# 제출 체크리스트 — 1차 예선

마감 **2026.09.15(화) 23:59:59** · 1시간 이내 지각 −3점, 이후 실격

---

## 1. 코드 (GitHub)

- [ ] `predict.ipynb` 첫 코드셀이 `os.environ.get` 형태 그대로인가
      (값을 직접 대입하면 운영진 주입이 무시되어 채점 불가)
- [ ] 로컬 경로(`./data`, `/content/drive`) 하드코딩이 없는가
- [ ] 최종 셀에서 `df.to_csv(OUTPUT_PATH, index=False)` 인가
- [ ] `month`/`day` 가 2자리 zero-padding 문자열인가 (`5` ❌ / `05` ✅)
- [ ] 미인식 값이 빈 문자열/NaN 이 아니라 문자열 `NONE` 인가
- [ ] `image_id` 가 파일명 문자열 그대로인가 (`000001` ≠ `1`)
- [ ] `requirements.txt` 에 `nbconvert`, `ipykernel` 이 있는가
- [ ] 패키지 버전이 고정되어 있는가
- [ ] 노트북에 `input()` 등 입력 대기 코드가 없는가
- [ ] 대용량 가중치가 `.gitignore` 에 등록되어 있는가
- [ ] **직접 학습한 `weights/yolo/date_v1.onnx` (11MB) 를 Release Assets 에 올렸는가**
- [ ] **`download_weights.sh` 의 `YOLO_URL` 을 실제 Release 주소로 채웠는가**
      (비워 둬도 추론은 실패하지 않는다 — 보조 경로가 det 단독으로 내려갈 뿐이고,
       보조 경로 자체가 전체의 5%에만 도므로 영향이 작다)
- [ ] **`requirements.txt` 에 `torch`/`ultralytics` 가 없는가**
      (임포트되는 것만으로 2.4배 느려진다 — 학습용은 `requirements-train.txt`)
- [ ] `weights/rapidocr/*.onnx` (30MB) 가 동봉되어 있는가
      (없으면 RapidOCR 이 최초 실행 때 자동 다운로드한다 — 네트워크가 없으면 정확도가 크게 떨어진다)
- [ ] `README.md` 만 보고 처음부터 재현이 가능한가

> 위 항목 중 형식 관련 5개는 노트북 마지막 셀의 `assert` 가 자동 검증한다.

### 자동 점검 (제출 직전 반드시 실행)

```powershell
python check_submission.py
```

노트북 형식·의존성·가중치·PDF·소스 구문을 한 번에 확인한다 (48개 항목).
눈으로 훑으면 놓치는 것들이라 기계에 맡긴다.

### 검증 명령 (그대로 실행)

```bash
export ITDA_INPUT_DIR=/tmp/test_imgs
export ITDA_OUTPUT_PATH=/tmp/out.csv
jupyter nbconvert --to notebook --execute predict.ipynb \
    --ExecutePreprocessor.timeout=2400 --output /tmp/executed.ipynb
```

에러 없이 끝나고 `/tmp/out.csv` 가 생성되어야 한다.

---

## 2. 요약서 PDF (정성 40점 — 가장 큰 배점)

- [ ] 파일명 `[DScover]_팀명_아키텍처구조도.pdf`
- [ ] **A4 2장 이내**
- [ ] 1페이지: 전체 파이프라인 아키텍처 구조도
- [ ] 2페이지: 설계 논리 + CPU 최적화 전략 + 활용 계획

> 쓸 재료는 [METHOD.md](METHOD.md) 에 다 있다. 특히 **기각된 선택지 목록**
> (해상도·임계값·2단계 재인식·전처리 13종)이 "논리성" 항목의 직접 증거다.
> 전처리를 n=100 에서 n=400 으로 늘려 결론을 뒤집은 사례도 그대로 쓸 수 있다.

- [ ] 가산점(최대 5점)을 노린다면 직접 수집/라벨링 증빙을 2페이지에 넣는다
      → `label/` 도구와 라벨링 규모를 근거로 제시

---

## 3. 제출 메일

- 수신: `dataitda5@gmail.com`
- 제목: `[1차예선제출] [DScover]_팀명`
- 첨부: `[DScover]_팀명_아키텍처구조도.pdf`

```
1. 팀명: [DScover]_팀명
2. 팀원 이름: 홍길동(대표자/팀장), ...
3. GitHub Repository URL: https://github.com/<user>/<repo>
4. 최종 제출 Commit Hash: <git rev-parse HEAD 실행 결과>
5. 첨부파일: [DScover]_팀명_아키텍처구조도.pdf
```

- [ ] Commit Hash 를 `git rev-parse HEAD` 로 실제 확인해 기재
- [ ] 첨부파일명이 규정과 일치
- [ ] 마감 여유 있게 발송

---

## 4. 운영진에 확인할 것

규정에 없어 전략이 갈리는 항목이다. 오픈채팅에 질의할 가치가 있다.

1. 정확도 채점이 `final_date` **완전일치**인가, `year`/`month`/`day` **필드별 부분점수**인가
2. **오답 감점**이 있는가 (있다면 확신 없을 때 `NONE` 이 유리하다)
3. 비공개 평가셋의 **대략적 규모** (속도 예산 계산의 전제)

---

## 5. 현재 상태

| 항목 | 상태 |
| --- | --- |
| `predict.ipynb` | ✅ 채점 명령으로 실행 확인 (2026-09-14, RapidOCR+보조 구성), 형식 검증 통과 |
| 정확도 | **82.3%** (홀드아웃2 198장, 부분점수) · 직전 구성 66.8% 대비 **+15.5%p** |
| 속도 | ✅ 단일 프로세스 0.53초/장 → 500장 약 267초 (제한 2,400초의 **11%**) |
| 라벨링 | ✅ **383장** (튜닝 137 / 홀드아웃1 48 / 홀드아웃2 198) |
| YOLO 가중치 배포 | ❌ **Release Assets 업로드 + `YOLO_URL` 기입 필요** |
| RapidOCR 모델 | ✅ `weights/rapidocr/` 에 동봉(30MB), 없으면 자동 다운로드로 강하 |
| 요약서 PDF | ⚠️ **구조·수치가 바뀌어 재작성 필요** (배점 40점) |

> ⚠️ 홀드아웃2 는 보조 경로의 설정 선택에 여러 차례 쓰여 그쪽에 낙관 편향이 있다.
> 다만 최종 채택한 주력 경로(RapidOCR)는 이 표본에 한 번도 노출되지 않았으므로,
> 82.3% 자체는 크게 부풀려진 값이 아니다.
>
> ⚠️ 과거에 적혀 있던 "0.99초/장, 제한의 21%" 는 4프로세스 병렬 처리량을
> 잘못 쓴 값이었다. 위 수치가 단일 프로세스 실측이다.
