# requirements 고정값의 근거

`requirements.txt` 와 `requirements-train.txt` 는 **순수 ASCII 로 유지한다.**
한글 주석을 그 파일에 넣으면 안 된다. 근거는 이 문서에 적는다.

---

## 왜 ASCII 만 쓰는가 — 실제로 설치가 깨졌다

pip 은 requirements 파일에 **BOM 이 없으면 시스템 기본 인코딩으로 읽는다**
(`pip/_internal/utils/encoding.py` 의 `auto_decode`). 한국어 Windows 에서는
그 값이 **cp949** 라, UTF-8 로 저장된 한글 주석을 만나면 다음과 같이 죽는다.

```
UnicodeDecodeError: 'cp949' codec can't decode byte 0xec in position 2:
illegal multibyte sequence
```

**설치가 한 줄도 진행되지 않는다.** 채점 환경이 한글·일본어·중국어 Windows 면
`pip install -r requirements.txt` 단계에서 끝나고 정량 전량 0점이 된다.

이 문제는 **새 가상환경에서 돌려 보기 전까지 보이지 않았다.** 개발용 `.venv` 는
오래전에 만들어 둔 것이라 requirements 를 다시 읽지 않았기 때문이다.
규정이 "제출 전 새 가상환경에서 반드시 수행" 을 요구하는 이유가 이것이다.

- 발견 기록: `logs/fresh_run_20261001_221252.log`
- 재발 방지: `check_submission.py` 가 두 requirements 파일의 ASCII 여부를 검사한다

> BOM 을 붙이는 것도 방법이지만, 일부 도구가 BOM 을 주석이 아닌 패키지명으로
> 읽는 사례가 있어 **ASCII 로 만드는 쪽**을 택했다. 로케일·pip 버전과 무관하게
> 안전하다.

---

## 추론 의존성 (`requirements.txt`)

| 패키지 | 고정값 | 이유 |
| --- | --- | --- |
| `numpy` | 1.26.4 | **paddlepaddle 2.6.x 는 numpy 1.x 를 요구한다.** numpy 2.x 를 설치하면 `numpy.core.multiarray failed to import` 로 import 단계에서 죽는다 |
| `opencv-python`<br>`opencv-contrib-python`<br>`opencv-python-headless` | **4.6.0.66**<br>(세 계열 동일) | 아래 "opencv 를 4.6 으로 내린 이유" 참고 |
| `rapidocr` | 3.9.2 | **주력 엔진.** PP-OCRv6 det/rec 를 onnxruntime 으로 돌린다. torch 를 싣지 않는 것이 핵심 |
| `paddlepaddle`<br>`paddleocr` | 2.6.2 / 2.7.3 | 보조 경로. 주력이 아무것도 읽지 못한 이미지에만 돈다 |
| `setuptools` | 84.0.0 | `paddle.utils.cpp_extension` 이 import 한다. 빼면 paddle import 가 실패한다 |
| `onnxruntime` | 1.19.2 | 날짜 검출기(YOLOv8n → ONNX)와 RapidOCR 이 함께 쓴다 |
| `nbconvert`<br>`ipykernel` | 7.17.1 / 7.3.0 | 규정 §7: 채점이 nbconvert 로 노트북을 실행한다. 둘 다 필수 |
| `pandas` | 2.3.3 | 제출 CSV 생성 |

### opencv 를 4.6.0.66 으로 내린 이유 — 안 그러면 설치가 아예 안 된다

개발·측정은 `4.10.0.84` 로 했다. 그런데 **새 가상환경에서 표준 명령으로는
설치가 불가능했다.**

```
ERROR: Cannot install -r requirements.txt (line 28), (line 32)
       and opencv-python==4.10.0.84 because these package versions have
       conflicting dependencies.
ERROR: ResolutionImpossible
```

`paddleocr 2.7.3` 이 상한을 선언하고 있다.

| 패키지 | 요구 |
| --- | --- |
| `paddleocr 2.7.3` | `opencv-python <=4.6.0.66` · `opencv-contrib-python <=4.6.0.66` |
| `rapidocr 3.9.2` | `opencv_python >=4.5.1.48` |

개발 환경은 패키지를 **하나씩 점진적으로** 깔아서 pip 이 전체를 한 번에 해결할
일이 없었다. 그래서 이 결함을 오래 안고 있으면서도 보이지 않았다.
`4.6.0.66` 으로 내리면 두 조건을 모두 만족한다(휠은 `cp36-abi3` 라 3.10 에서 정상).

### 세 계열을 같은 버전으로 고정하는 이유

`opencv-python` / `opencv-contrib-python` / `opencv-python-headless` 는
**같은 `cv2` 디렉터리에 설치된다.** 그래서 마지막에 설치된 것이 이기고, 버전이
다르면 **실제 cv2 버전이 설치 순서에 따라 달라진다.**

실제로 관측했다 — `paddleocr` → `pdf2docx` 가 `opencv-python-headless>=4.5` 를
요구하는데, pip 이 거기에 **4.11.0.86** 을 꽂는다. 다른 두 개는 4.6.0.66 인
상태로 공존한다. 테스트 머신에서는 4.6 이 이겼지만 순서가 바뀌면 4.11 이 된다.

세 계열을 모두 `4.6.0.66` 으로 고정하면 순서와 무관하게 `cv2.__version__` 이
`4.6.0` 으로 결정된다. `check_submission.py` 가 세 계열의 버전 일치와
paddleocr 상한 준수를 검사한다.

### torch / ultralytics 를 넣으면 안 된다

추론에는 필요 없고, **임포트되는 것만으로** 파이프라인이 2.4배 느려진다.

| 환경 | 장당 | 500장 | 제한 대비 |
| --- | ---: | ---: | ---: |
| torch 없음 (제출 구성) | 1.20초 | 614초 | 26% |
| torch 설치됨 | 5.88초 | 2,939초 | **122% — 초과** |

paddle 과 torch 가 각자 OpenMP 런타임을 싣고 스레드를 다투기 때문이다.
`set_num_threads(1)` · `KMP_BLOCKTIME=0` · `OMP_WAIT_POLICY=PASSIVE` 를 모두
시도했으나 해결되지 않았다 (EDA 52~53). 그래서 학습·ONNX 내보내기 의존성은
`requirements-train.txt` 로 분리했고, 제출 노트북에는 두 패키지의 import 가
한 줄도 없다(`check_submission.py` 가 검사한다).

---

## 학습 의존성 (`requirements-train.txt`)

채점에는 필요 없다. YOLOv8n 파인튜닝과 ONNX 내보내기를 재현할 때만 쓴다.
재현 절차는 그 파일 주석과 `JOURNAL.md` 의 EDA 45~56 절에 있다.
