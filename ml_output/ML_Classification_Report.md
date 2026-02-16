# BITalino 실험 단계 분류 기계학습 분석 보고서

**생성 일시**: 2026-02-16 02:35:22

---

## 1. 요약 (Abstract)

본 연구에서는 BITalino 생체신호 측정 장치를 사용하여 수집된 ECG(심전도), EMG(근전도), EDA(피부전도도) 데이터를 기반으로 실험 단계를 자동으로 분류하는 기계학습 모델을 개발하였습니다.

- **총 피험자 수**: 8명
- **분류 대상 단계**: BASELINE, TASK, PUZZLE, CHASE
- **사용 특징 수**: 20개
- **최고 성능 모델**: Random Forest
- **LOSO CV 정확도**: 0.6774
- **F1-Score**: 0.6655

## 2. 방법론 (Methodology)

### 2.1 데이터셋

| 피험자 | 샘플 수 |
|--------|--------|
| hase1 | 4 |
| hase2 | 4 |
| ishikawa | 4 |
| masaya | 3 |
| matsumoto | 4 |
| sensei | 4 |
| takamiya | 4 |
| takamiya2 | 4 |
| **합계** | **31** |

### 2.2 특징 (Features)

총 15개의 생리학적 특징을 추출하였습니다:

#### ECG 특징 (심전도)
- `ECG_heart_rate_bpm`: 심박수 (beats per minute)
- `ECG_hrv_rmssd_ms`: 심박변이도 RMSSD (milliseconds)
- `ECG_num_r_peaks`: R-peak 개수
- `ECG_mean_rr_ms`: 평균 RR 간격 (milliseconds)
- `ECG_std_rr_ms`: RR 간격 표준편차 (milliseconds)

#### EMG 특징 (근전도)
- `EMG_mean_uV`: 평균 근전도 값 (microvolts)
- `EMG_std_uV`: 근전도 표준편차 (microvolts)
- `EMG_max_uV`: 최대 근전도 값 (microvolts)
- `EMG_rms_uV`: 근전도 RMS 값 (microvolts)
- `EMG_integrated_emg`: 적분 근전도 값

#### EDA 특징 (피부전도도)
- `EDA_mean_SCL_uS`: 평균 피부 전도도 레벨 (microsiemens)
- `EDA_std_SCL_uS`: 피부 전도도 표준편차 (microsiemens)
- `EDA_mean_SCR_uS`: 평균 피부 전도 반응 (microsiemens)
- `EDA_max_SCR_uS`: 최대 피부 전도 반응 (microsiemens)
- `EDA_num_SCR_peaks`: 피부 전도 반응 피크 수

### 2.3 교차 검증 전략

**Leave-One-Subject-Out (LOSO) 교차 검증**을 사용하였습니다.

- 한 피험자의 모든 데이터를 테스트 셋으로 사용
- 나머지 피험자 데이터로 모델 학습
- 피험자 간 개인차를 고려한 일반화 성능 평가
- 총 8회의 교차 검증 수행

### 2.4 분류 알고리즘

5가지 기계학습 알고리즘을 비교 평가하였습니다:

#### 1) Random Forest (랜덤 포레스트)
- 다수의 결정 트리를 학습하여 앙상블하는 방법
- 과적합 방지에 효과적이며 특징 중요도 분석 가능
- 파라미터: `n_estimators=100`, `max_depth=10`

#### 2) Gradient Boosting (그래디언트 부스팅)
- 순차적으로 약한 학습기를 추가하여 오차를 줄여나가는 방식
- 높은 예측 성능
- 파라미터: `n_estimators=100`, `learning_rate=0.1`, `max_depth=5`

#### 3) Support Vector Machine (서포트 벡터 머신)
- 고차원 공간에서 최적의 결정 경계를 찾는 방법
- RBF(Radial Basis Function) 커널 사용
- 파라미터: `kernel='rbf'`, `C=1.0`, `gamma='scale'`

#### 4) k-Nearest Neighbors (k-최근접 이웃)
- 새로운 샘플과 가장 가까운 k개 이웃의 다수결로 분류
- 파라미터: `n_neighbors=5`, `weights='distance'`

#### 5) Logistic Regression (로지스틱 회귀)
- 선형 분류기로 베이스라인 모델로 사용
- 파라미터: `C=1.0`, `multi_class='multinomial'`

## 3. 결과 (Results)

### 3.1 모델 성능 비교

| 모델 | Accuracy | F1-Score | Precision | Recall |
|------|----------|----------|-----------|--------|
| Random Forest | 0.6774 | 0.6655 | 0.7043 | 0.6774 |
| Gradient Boosting | 0.6452 | 0.6370 | 0.6547 | 0.6452 |
| SVM (RBF) | 0.4516 | 0.4653 | 0.5072 | 0.4516 |
| k-NN | 0.4516 | 0.4207 | 0.4485 | 0.4516 |
| Logistic Regression | 0.4839 | 0.4946 | 0.5161 | 0.4839 |

**최고 성능 모델**: Random Forest

### 3.2 혼동 행렬

**Random Forest** 모델의 혼동 행렬:

| | Predicted |
|---|BASELINE|CHASE|PUZZLE|TASK|
|---|---|---|---|---|
| **BASELINE** |7|0|1|0|
| **CHASE** |1|5|1|0|
| **PUZZLE** |0|1|6|1|
| **TASK** |4|0|1|3|

### 3.3 상세 분류 보고서

```
              precision    recall  f1-score   support

    BASELINE       0.58      0.88      0.70         8
       CHASE       0.83      0.71      0.77         7
      PUZZLE       0.67      0.75      0.71         8
        TASK       0.75      0.38      0.50         8

    accuracy                           0.68        31
   macro avg       0.71      0.68      0.67        31
weighted avg       0.70      0.68      0.67        31
```

## 4. 생성된 파일

| 파일명 | 설명 |
|--------|------|
| `model_comparison.png` | 모델별 성능 비교 차트 |
| `confusion_matrices.png` | 각 모델의 혼동 행렬 |
| `feature_importance.png` | 특징 중요도 (Random Forest) |
| `subject_accuracy.png` | 피험자별 정확도 |
| `trained_model.joblib` | 학습된 최종 모델 |
| `scaler.joblib` | 특징 스케일러 |
| `label_encoder.joblib` | 레이블 인코더 |
| `subject_validation_results.csv` | 피험자별 검증 결과 |

## 5. 결론 (Conclusion)

본 연구에서 Random Forest 모델이 LOSO 교차 검증에서 67.7%의 정확도와 66.6%의 F1-Score를 달성하였습니다.

이는 BITalino 센서에서 추출된 생리학적 특징만으로도 실험 단계를 효과적으로 분류할 수 있음을 보여줍니다.

### 향후 연구 방향

1. 더 많은 피험자 데이터 수집을 통한 모델 일반화 성능 향상
2. 딥러닝 기반 시계열 분석 방법 적용 (LSTM, Transformer 등)
3. 실시간 분류 시스템 개발
4. 다른 생리학적 신호 추가 (호흡, 체온 등)

---

*본 보고서는 자동 생성되었습니다. (2026-02-16 02:35:22)*
