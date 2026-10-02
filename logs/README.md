# 새 가상환경 재현성 검증 로그

규정 요구사항 — "제출 전 아래 절차를 새 가상환경에서 반드시 수행하고,
실행 로그를 저장소에 포함해 주세요."

생성 명령

```powershell
powershell -ExecutionPolicy Bypass -File verify_fresh.ps1 -N 500
```

새 venv 생성 → `pip install -r requirements.txt` → `bash download_weights.sh`
→ `check_submission.py` → `audit_notebook.py` → `nbconvert --execute` 500장
전 과정을 기록한다.

---

## 로그 세 개를 모두 남긴 이유

**실패한 로그를 지우지 않았다.** 이 절차가 실제로 무엇을 잡아냈는지가
재현성의 증거라고 보기 때문이다. 세 번 돌려 **제출을 망칠 결함 다섯 개**를
찾았고, 마지막 로그가 그것들을 모두 고친 뒤의 결과다.

| 로그 | 결과 | 무엇을 잡았나 |
| --- | --- | --- |
| `fresh_run_20261001_221252.log` | 실패 | ① requirements 의 한글 주석 → pip 이 cp949 로 읽다 죽음<br>② `download_weights.sh` 가 `python` 을 못 찾아 주력 가중치 110MB 를 조용히 건너뜀 |
| `fresh_run_20261001_221651.log` | 실패 | ③ opencv 4.10 고정이 paddleocr 상한(4.6.0.66)과 충돌 → `ResolutionImpossible`, 설치 자체가 불가능 |
| `fresh_run_20261002_161451.log` | 부분 실패 | ④ WSL bash 의 `/mnt/c/...` 경로를 Windows python 에 넘겨 가중치 복사 실패 → 주력 엔진 없이 보조 경로만으로 돌아 **1,629초(제한의 65%)**<br>⑤ 한국어 rec 를 PP-OCRv6 으로 요청해 실패(실제 파일은 v5 mobile) |
| **`fresh_run_20261002_164805.log`** | **성공** | 위 다섯 가지 수정 후 전 단계 통과 |

### 다섯 결함의 공통점

**전부 개발 환경에서는 보이지 않는 종류다.**

- ①③ 은 패키지를 하나씩 점진적으로 깔아 둔 환경에서는 발생하지 않는다
- ②④ 는 파일이 이미 있어 `skip (exists)` 로 빠지면 가려진다
- ⑤ 는 예전에 받아 둔 파일이 남아 있어 가려졌다

노트북을 500장 완주시켜 "검증 완료" 라고 적어 둔 적이 있는데, 그것은
**이미 세팅된 환경에 한정된 검증**이었다. 규정이 "새 가상환경에서 반드시" 를
요구한 이유가 여기 있었다.

재발 방지로 `check_submission.py` 검사를 49 → 56개로 늘렸다
(requirements ASCII 여부, 다섯 로케일 파싱, opencv 세 계열 버전 일치,
paddleocr 상한 준수).

### 채점 환경과 같은 x86 리눅스에서의 확인 (GitHub Codespaces)

위 로그는 전부 Windows 개발 PC 에서 나왔다. 채점 환경(Ubuntu · x86 · CPU 4코어)에
가까운 조건에서 같은 절차를 따로 돌렸다.

| 로그 | 환경 | 결과 |
| --- | --- | --- |
| `repro_codespace_20261002_0719.log` | 4 vCPU x86 · Ubuntu 24.04 · Python 3.10 · 커밋 `258a929` | **성공** · 새 venv → `pip install` → `download_weights.sh` 가중치 11개 전부 수신(**Release 의 YOLO·도트 판독기 포함**) → `nbconvert --execute` `sample/` 50장 **168초** 완주, 50행 출력 |
| `speed_codespace_500img_20261002.log` | 4 vCPU x86(AMD EPYC 7763) · Ubuntu 24.04 · 커밋 `24e8d36` | **속도 실측** · 배포 사진 500장(6장 간격 추출) 공식 명령 실행, 추론 **1,169초**(장당 2.34초) · 전체 1,178초 = **제한 2,500초의 47%**, 500행 출력 |

리눅스에서는 `opencv-python`(paddleocr 2.7.3 이 강제 설치)이 `libgl1` 을 요구해
`apt install libgl1 libglib2.0-0` 이 먼저 필요했다. 속도 실측은 위 재현 로그에서 만든
환경을 그대로 썼다. 두 커밋과 현재 커밋 사이에 `predict.ipynb` · `src/` ·
`requirements.txt` · `download_weights.sh` 변경은 없다.

---

## 성공 로그 요약 — `fresh_run_20261002_164805.log`

| 단계 | 결과 |
| --- | --- |
| 환경 | Windows 11 · AMD Ryzen 5 7500F · Python 3.10 · GPU 미사용 |
| `pip install -r requirements.txt` | 통과 |
| `bash download_weights.sh` | RapidOCR 6개 + PaddleOCR 3세트 + **Release 에서 YOLO·도트 판독기** 수신 |
| `check_submission.py` | 통과 |
| `audit_notebook.py` | 통과 (노트북이 네트워크에 나가지 않음) |
| 엔진 로딩 | **8개 전부 `bundled`** (주력·det medium·rec 한국어·rec 영문·도트·YOLO·보조) |
| 추론 | 500장 **522초** (장당 1.043초) |
| **전체 벽시계** | **538초 = 제한 2,500초의 22%** |
| 출력 | `submission.csv` **500행** · 값 없음 20행 |

### 개발 환경과 동일하게 재현됐다

| 항목 | 개발 환경 | 새 가상환경 |
| --- | --- | --- |
| 칸별 처리 건수 | rapid 409 / adaptive 11 / retry 16 / clahe 3 / rec_ko 5 / rec_en 1 / dot 3 / partial 28 / fallback 4 / none 20 | **동일** |
| 필드 정확도(라벨 127장) | 94.0% | **94.0%** |
| 완전일치 | 92.1% | **92.1%** |
| 벽시계 | 614초 (25%) | 538초 (22%) |

처리 경로가 한 자리도 다르지 않다. 속도만 12% 차이가 나는데 같은 기계의
부하 차이로 본다. **보고에는 더 느린 614초 / 25% 를 쓴다.**

> 위 127장은 설계에 쓴 표본이 섞여 있어 성능 근거가 아니다.
> 성능 근거는 설계에 한 번도 쓰지 않은 248장에서 잰 **92.3%** 다(METHOD.md 4-2).
