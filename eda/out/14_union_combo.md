# EDA 14 — 전처리 조합(합집합) 분석

- 평가 표본 100장, 기준선 `none|both` coverage **31.0%**
- 새 OCR 실행 없음 (10단계 수집 결과 재사용)

> 12단계에서 전 설정 합집합이 63.0% 였다. 단일 최고가 40.0% 인데 합집합이 그보다 훨씬 높다는 것은 **전처리마다 잡아내는 이미지가 다르다**는 뜻이다. det 가 입력의 작은 변화에 민감하다.

## 단일 설정

| 설정 | coverage | plausible | 후보/장 | 초/장 |
| --- | ---: | ---: | ---: | ---: |
| `adapt_bin|det_only` | 40.0% | 89.6% | 0.48 | 5.11 |
| `illum+clahe|det_only` | 38.0% | 85.1% | 0.47 | 6.32 |
| `illum_norm|both` | 35.0% | 82.2% | 0.45 | 7.17 |
| `illum_norm|det_only` | 35.0% | 83.7% | 0.43 | 6.82 |
| `illum+clahe|both` | 35.0% | 90.7% | 0.43 | 6.54 |
| `blackhat|det_only` | 34.0% | 92.5% | 0.40 | 4.71 |
| `clahe|det_only` | 33.0% | 86.0% | 0.43 | 6.50 |
| `dot_close3|both` | 33.0% | 89.5% | 0.38 | 6.05 |
| `blackhat|both` | 33.0% | 92.1% | 0.38 | 5.12 |
| `clahe|both` | 32.0% | 85.7% | 0.42 | 7.10 |
| `none|both` | 31.0% | 87.2% | 0.39 | 8.23 |
| `adapt_bin|both` | 31.0% | 88.6% | 0.35 | 5.62 |
| `dot_close3|det_only` | 29.0% | 83.8% | 0.37 | 5.69 |

## 2개 조합 — 상위 12

| 조합 | coverage | Δ(기준선) | plausible | 초/장 | 시간당 이득 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `illum+clahe|det_only` + `adapt_bin|det_only` | **51.0%** | +20.0%p | 87.1% | 11.43 | 1.75 %p/초 |
| `illum+clahe|both` + `adapt_bin|det_only` | **49.0%** | +18.0%p | 87.3% | 11.65 | 1.55 %p/초 |
| `illum+clahe|det_only` + `adapt_bin|both` | **48.0%** | +17.0%p | 83.3% | 11.94 | 1.42 %p/초 |
| `clahe|det_only` + `adapt_bin|det_only` | **47.0%** | +16.0%p | 86.7% | 11.61 | 1.38 %p/초 |
| `illum_norm|both` + `adapt_bin|det_only` | **47.0%** | +16.0%p | 82.0% | 12.28 | 1.30 %p/초 |
| `illum_norm|det_only` + `adapt_bin|det_only` | **47.0%** | +16.0%p | 84.7% | 11.93 | 1.34 %p/초 |
| `blackhat|both` + `adapt_bin|det_only` | **47.0%** | +16.0%p | 87.9% | 10.23 | 1.56 %p/초 |
| `clahe|both` + `adapt_bin|det_only` | **46.0%** | +15.0%p | 82.3% | 12.21 | 1.23 %p/초 |
| `illum_norm|both` + `adapt_bin|both` | **46.0%** | +15.0%p | 81.4% | 12.79 | 1.17 %p/초 |
| `illum_norm|det_only` + `adapt_bin|both` | **46.0%** | +15.0%p | 82.5% | 12.44 | 1.21 %p/초 |
| `illum+clahe|det_only` + `dot_close3|both` | **46.0%** | +15.0%p | 84.7% | 12.37 | 1.21 %p/초 |
| `dot_close3|both` + `adapt_bin|det_only` | **46.0%** | +15.0%p | 87.1% | 11.16 | 1.34 %p/초 |

> 비용 대비 효율 1위 조합: `illum+clahe|det_only` + `adapt_bin|det_only` — coverage 51.0% (+20.0%p), 11.43초/장

## 3개 조합 — 상위 6

| 조합 | coverage | Δ | 초/장 |
| --- | ---: | ---: | ---: |
| `illum+clahe|det_only` + `dot_close3|both` + `adapt_bin|det_only` | **56.0%** | +25.0%p | 17.48 |
| `illum_norm|both` + `illum+clahe|det_only` + `adapt_bin|det_only` | **54.0%** | +23.0%p | 18.60 |
| `illum+clahe|both` + `dot_close3|both` + `adapt_bin|det_only` | **54.0%** | +23.0%p | 17.70 |
| `illum+clahe|det_only` + `dot_close3|both` + `adapt_bin|both` | **54.0%** | +23.0%p | 17.99 |
| `none|both` + `illum+clahe|det_only` + `adapt_bin|det_only` | **53.0%** | +22.0%p | 19.66 |
| `clahe|both` + `illum+clahe|det_only` + `adapt_bin|det_only` | **53.0%** | +22.0%p | 18.53 |

- 전 설정(13개) 합집합 상한: **63.0%**, 비용 81.0초/장 (비현실적)

## 해석

조합은 coverage 를 크게 올리지만 비용이 선형으로 늘어난다. CPU 예산(0.72초/장)을 감안하면 **2개까지가 현실적 상한**이고, 그것도 박스 필터로 rec 비용을 3배 줄인다는 전제에서만 가능하다.

> ⚠️ 더 근본적인 문제: 합집합이 이렇게 크게 오른다는 것은 **det 가 불안정하다**는 신호다. 같은 날짜가 전처리에 따라 잡히기도 하고 안 잡히기도 한다. 안정적인 검출기를 쓰는 것이 조합을 늘리는 것보다 낫다.