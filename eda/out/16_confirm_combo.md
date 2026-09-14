# EDA 14 — 전처리 조합(합집합) 분석

- 평가 표본 400장, 기준선 `none|both` coverage **36.8%**
- 새 OCR 실행 없음 (10단계 수집 결과 재사용)

> 12단계에서 전 설정 합집합이 63.0% 였다. 단일 최고가 40.0% 인데 합집합이 그보다 훨씬 높다는 것은 **전처리마다 잡아내는 이미지가 다르다**는 뜻이다. det 가 입력의 작은 변화에 민감하다.

## 단일 설정

| 설정 | coverage | plausible | 후보/장 | 초/장 |
| --- | ---: | ---: | ---: | ---: |
| `adapt_bin|det_only` | 39.8% | 88.4% | 0.50 | 4.80 |
| `illum+clahe|det_only` | 37.5% | 85.6% | 0.47 | 5.49 |
| `none|both` | 36.8% | 85.6% | 0.45 | 7.15 |
| `illum+clahe|both` | 35.0% | 87.2% | 0.45 | 5.89 |
| `adapt_bin|both` | 33.2% | 88.0% | 0.42 | 5.34 |

## 2개 조합 — 상위 12

| 조합 | coverage | Δ(기준선) | plausible | 초/장 | 시간당 이득 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `none|both` + `adapt_bin|det_only` | **47.5%** | +10.7%p | 84.8% | 11.95 | 0.90 %p/초 |
| `adapt_bin|det_only` + `illum+clahe|det_only` | **46.5%** | +9.8%p | 84.0% | 10.29 | 0.95 %p/초 |
| `none|both` + `adapt_bin|both` | **46.2%** | +9.5%p | 85.1% | 12.49 | 0.76 %p/초 |
| `adapt_bin|both` + `illum+clahe|det_only` | **45.5%** | +8.8%p | 84.4% | 10.83 | 0.81 %p/초 |
| `adapt_bin|det_only` + `illum+clahe|both` | **44.8%** | +8.0%p | 84.2% | 10.69 | 0.75 %p/초 |
| `adapt_bin|both` + `illum+clahe|both` | **43.5%** | +6.8%p | 85.1% | 11.23 | 0.60 %p/초 |
| `adapt_bin|both` + `adapt_bin|det_only` | **42.8%** | +6.0%p | 86.6% | 10.14 | 0.59 %p/초 |
| `none|both` + `illum+clahe|det_only` | **42.0%** | +5.2%p | 82.7% | 12.64 | 0.42 %p/초 |
| `none|both` + `illum+clahe|both` | **41.5%** | +4.7%p | 82.7% | 13.04 | 0.36 %p/초 |
| `illum+clahe|both` + `illum+clahe|det_only` | **39.5%** | +2.8%p | 82.6% | 11.38 | 0.24 %p/초 |

> 비용 대비 효율 1위 조합: `adapt_bin|det_only` + `illum+clahe|det_only` — coverage 46.5% (+9.8%p), 10.29초/장

## 3개 조합 — 상위 6

| 조합 | coverage | Δ | 초/장 |
| --- | ---: | ---: | ---: |
| `none|both` + `adapt_bin|det_only` + `illum+clahe|det_only` | **49.8%** | +13.0%p | 17.44 |
| `none|both` + `adapt_bin|both` + `adapt_bin|det_only` | **49.5%** | +12.8%p | 17.29 |
| `none|both` + `adapt_bin|both` + `illum+clahe|det_only` | **49.2%** | +12.5%p | 17.98 |
| `none|both` + `adapt_bin|det_only` + `illum+clahe|both` | **49.0%** | +12.2%p | 17.84 |
| `none|both` + `adapt_bin|both` + `illum+clahe|both` | **48.8%** | +12.0%p | 18.38 |
| `adapt_bin|both` + `adapt_bin|det_only` + `illum+clahe|det_only` | **48.8%** | +12.0%p | 15.63 |

- 전 설정(5개) 합집합 상한: **52.8%**, 비용 28.7초/장 (비현실적)

## 해석

조합은 coverage 를 크게 올리지만 비용이 선형으로 늘어난다. CPU 예산(0.72초/장)을 감안하면 **2개까지가 현실적 상한**이고, 그것도 박스 필터로 rec 비용을 3배 줄인다는 전제에서만 가능하다.

> ⚠️ 더 근본적인 문제: 합집합이 이렇게 크게 오른다는 것은 **det 가 불안정하다**는 신호다. 같은 날짜가 전처리에 따라 잡히기도 하고 안 잡히기도 한다. 안정적인 검출기를 쓰는 것이 조합을 늘리는 것보다 낫다.