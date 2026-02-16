#!/usr/bin/env python3
"""
BITalino 실험 단계 분류 기계학습 모델
=====================================

이 스크립트는 BITalino 센서(ECG, EMG, EDA)에서 수집된 생체 신호 데이터를 
기반으로 실험 단계(BASELINE, TASK, PUZZLE, CHASE)를 자동으로 분류하는
기계학습 모델을 학습하고 검증합니다.

Author: Research Team
Date: 2026-02-05
Version: 1.0

=== 기계학습 방법론 (Methodology) ===

1. 데이터셋 구성
   - 8명의 피험자 (hase1, hase2, ishikawa, masaya, matsumoto, sensei, takamiya, takamiya2)
   - 각 피험자당 4개의 실험 단계 (BASELINE, TASK, PUZZLE, CHASE)
   - 총 32개 샘플 (일부 피험자는 CHASE 단계 데이터 없음)

2. 특징 (Features)
   - ECG: 심박수(BPM), HRV(RMSSD), R-peak 수, 평균 RR 간격, RR 표준편차
   - EMG: 평균값, 표준편차, 최대값, RMS, 적분값
   - EDA: 평균 SCL, SCL 표준편차, 평균 SCR, 최대 SCR, SCR 피크 수
   - 총 17개 특징

3. 교차 검증 전략
   - Leave-One-Subject-Out (LOSO) 교차 검증
   - 한 피험자를 테스트 셋으로 사용하고 나머지로 학습
   - 피험자 간 개인차를 고려한 일반화 성능 평가

4. 분류 알고리즘
   - Random Forest: 앙상블 기반 결정 트리
   - Gradient Boosting: 순차적 부스팅 기반 앙상블
   - Support Vector Machine (SVM): 커널 기반 분류기
   - k-Nearest Neighbors (k-NN): 거리 기반 분류기
   - Logistic Regression: 선형 분류기 (베이스라인)

5. 평가 지표
   - Accuracy: 전체 정확도
   - Precision: 정밀도 (각 클래스별)
   - Recall: 재현율 (각 클래스별)
   - F1-Score: 정밀도와 재현율의 조화 평균
   - Confusion Matrix: 혼동 행렬

=== 실행 방법 ===
    python ml_phase_classifier.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score
)
from sklearn.pipeline import Pipeline
import joblib

# 플롯 설정
plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150


class MLPhaseClassifier:
    """
    BITalino 실험 단계 분류기
    
    이 클래스는 BITalino 센서 데이터에서 추출된 특징을 기반으로
    실험 단계를 분류하는 기계학습 모델을 학습하고 평가합니다.
    
    Attributes:
        base_dir (str): 프로젝트 기본 디렉토리
        output_dir (str): 결과 출력 디렉토리
        features_file (str): 특징 데이터 CSV 파일 경로
        
    Methods:
        load_data(): 특징 데이터 로드
        prepare_features(): 특징 및 레이블 준비
        train_and_evaluate(): 모델 학습 및 평가
        generate_report(): 결과 보고서 생성
    """
    
    # 사용할 특징 목록
    FEATURE_COLUMNS = [
        # ECG 심박수 특징
        'ECG_heart_rate_bpm',   # 심박수 (BPM)
        
        # ECG 시간 영역 HRV 특징
        'ECG_hrv_rmssd_ms',     # RMSSD: 부교감신경 활동 지표 (ms)
        'ECG_hrv_sdnn_ms',      # SDNN: 전반적 HRV 지표 (ms)
        'ECG_pnn50_percent',    # pNN50: 부교감신경 활동 비율 (%)
        
        # ECG 주파수 영역 HRV 특징
        'ECG_lf_power_ms2',     # LF 파워: 교감+부교감 (0.04-0.15Hz)
        'ECG_hf_power_ms2',     # HF 파워: 부교감 (0.15-0.40Hz)
        'ECG_lf_hf_ratio',      # LF/HF 비율: 자율신경 균형 (스트레스 지표)
        
        # ECG R-peak 통계
        'ECG_num_r_peaks',      # R-peak 개수
        'ECG_mean_rr_ms',       # 평균 RR 간격 (ms)
        'ECG_std_rr_ms',        # RR 간격 표준편차 (ms)
        
        # EMG 특징
        'EMG_mean_uV',          # EMG 평균값 (μV)
        'EMG_std_uV',           # EMG 표준편차 (μV)
        'EMG_max_uV',           # EMG 최대값 (μV)
        'EMG_rms_uV',           # EMG RMS (μV)
        'EMG_integrated_emg',   # 적분 EMG
        
        # EDA 특징
        'EDA_mean_SCL_uS',      # 평균 피부 전도도 (μS)
        'EDA_std_SCL_uS',       # 피부 전도도 표준편차 (μS)
        'EDA_mean_SCR_uS',      # 평균 SCR (μS)
        'EDA_max_SCR_uS',       # 최대 SCR (μS)
        'EDA_num_SCR_peaks',    # SCR 피크 수
    ]
    
    # 분류할 실험 단계
    PHASE_LABELS = ['BASELINE', 'TASK', 'PUZZLE', 'CHASE']
    
    # 피험자 목록
    SUBJECTS = ['hase1', 'hase2', 'ishikawa', 'masaya', 'matsumoto', 'sensei', 'takamiya', 'takamiya2']
    
    def __init__(self, base_dir=None):
        """
        초기화
        
        Args:
            base_dir (str): 프로젝트 기본 디렉토리. None이면 현재 디렉토리 사용
        """
        self.base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.join(self.base_dir, 'ml_output')
        self.features_file = os.path.join(self.base_dir, 'analysis_output', 'summary', 'deep_research_features.csv')
        
        # 출력 디렉토리 생성
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 데이터 저장 변수
        self.raw_data = None
        self.X = None
        self.y = None
        self.subjects = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
        # 모델 정의
        self.models = self._define_models()
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def _define_models(self):
        """
        분류 모델 정의
        
        Returns:
            dict: 모델 이름과 모델 객체 딕셔너리
            
        각 모델에 대한 설명:
        
        1. Random Forest (랜덤 포레스트)
           - 다수의 결정 트리를 학습하여 앙상블하는 방법
           - 과적합 방지에 효과적
           - 특징 중요도 분석 가능
           - 파라미터: n_estimators=100 (트리 개수), max_depth=10 (트리 깊이 제한)
           
        2. Gradient Boosting (그래디언트 부스팅)
           - 순차적으로 약한 학습기를 추가하여 오차를 줄여나가는 방식
           - 높은 예측 성능
           - 파라미터: n_estimators=100, learning_rate=0.1, max_depth=5
           
        3. Support Vector Machine (서포트 벡터 머신)
           - 고차원 공간에서 최적의 결정 경계를 찾는 방법
           - RBF(Radial Basis Function) 커널 사용
           - 파라미터: kernel='rbf', C=1.0, gamma='scale'
           
        4. k-Nearest Neighbors (k-최근접 이웃)
           - 새로운 샘플과 가장 가까운 k개 이웃의 다수결로 분류
           - 간단하고 직관적
           - 파라미터: n_neighbors=5, weights='distance'
           
        5. Logistic Regression (로지스틱 회귀)
           - 선형 분류기로 베이스라인 모델로 사용
           - 다중 클래스 분류를 위해 multinomial 사용
           - 파라미터: C=1.0, max_iter=1000
        """
        return {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ),
            'SVM (RBF)': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            ),
            'k-NN': KNeighborsClassifier(
                n_neighbors=5,
                weights='distance',
                n_jobs=-1
            ),
            'Logistic Regression': LogisticRegression(
                C=1.0,
                max_iter=1000,
                multi_class='multinomial',
                random_state=42
            )
        }
    
    def load_data(self):
        """
        특징 데이터 로드
        
        Returns:
            pd.DataFrame: 로드된 데이터프레임
        """
        print("=" * 60)
        print("1. 데이터 로드")
        print("=" * 60)
        
        if not os.path.exists(self.features_file):
            raise FileNotFoundError(f"특징 파일을 찾을 수 없습니다: {self.features_file}")
        
        self.raw_data = pd.read_csv(self.features_file)
        print(f"   파일: {self.features_file}")
        print(f"   전체 샘플 수: {len(self.raw_data)}")
        print(f"   피험자 수: {self.raw_data['subject_id'].nunique()}")
        print(f"   단계별 샘플 수:")
        for phase in self.PHASE_LABELS:
            count = len(self.raw_data[self.raw_data['phase'] == phase])
            print(f"      - {phase}: {count}")
        
        return self.raw_data
    
    def prepare_features(self):
        """
        학습을 위한 특징 및 레이블 준비
        
        Returns:
            tuple: (X, y, subjects) - 특징 행렬, 레이블 배열, 피험자 배열
        """
        print("\n" + "=" * 60)
        print("2. 특징 및 레이블 준비")
        print("=" * 60)
        
        # 결측치 처리
        df = self.raw_data.dropna(subset=self.FEATURE_COLUMNS + ['phase', 'subject_id'])
        print(f"   결측치 제거 후 샘플 수: {len(df)}")
        
        # 특징 추출
        self.X = df[self.FEATURE_COLUMNS].values
        print(f"   특징 수: {len(self.FEATURE_COLUMNS)}")
        print(f"   특징 행렬 shape: {self.X.shape}")
        
        # 레이블 인코딩
        self.y = self.label_encoder.fit_transform(df['phase'].values)
        self.subjects = df['subject_id'].values
        
        print(f"   클래스 레이블: {list(self.label_encoder.classes_)}")
        print(f"   피험자 목록: {list(set(self.subjects))}")
        
        # 특징 통계 출력
        print("\n   특징 통계:")
        for i, col in enumerate(self.FEATURE_COLUMNS):
            print(f"      {col}: mean={self.X[:, i].mean():.2f}, std={self.X[:, i].std():.2f}")
        
        return self.X, self.y, self.subjects
    
    def train_and_evaluate(self):
        """
        모델 학습 및 평가
        
        Leave-One-Subject-Out (LOSO) 교차 검증을 사용하여
        각 모델의 성능을 평가합니다.
        
        Returns:
            dict: 각 모델의 평가 결과
        """
        print("\n" + "=" * 60)
        print("3. 모델 학습 및 평가 (Leave-One-Subject-Out)")
        print("=" * 60)
        
        # LOSO 교차 검증 설정
        logo = LeaveOneGroupOut()
        
        for model_name, model in self.models.items():
            print(f"\n   [{model_name}]")
            
            # 각 폴드별 예측 저장
            all_y_true = []
            all_y_pred = []
            fold_scores = []
            
            # LOSO 교차 검증 수행
            for train_idx, test_idx in logo.split(self.X, self.y, self.subjects):
                X_train, X_test = self.X[train_idx], self.X[test_idx]
                y_train, y_test = self.y[train_idx], self.y[test_idx]
                
                # 스케일링
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # 학습 및 예측
                model_clone = self._clone_model(model)
                model_clone.fit(X_train_scaled, y_train)
                y_pred = model_clone.predict(X_test_scaled)
                
                # 결과 저장
                all_y_true.extend(y_test)
                all_y_pred.extend(y_pred)
                fold_scores.append(accuracy_score(y_test, y_pred))
            
            # 전체 성능 계산
            accuracy = accuracy_score(all_y_true, all_y_pred)
            f1 = f1_score(all_y_true, all_y_pred, average='weighted')
            precision = precision_score(all_y_true, all_y_pred, average='weighted')
            recall = recall_score(all_y_true, all_y_pred, average='weighted')
            
            self.results[model_name] = {
                'accuracy': accuracy,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'fold_scores': fold_scores,
                'y_true': all_y_true,
                'y_pred': all_y_pred,
                'confusion_matrix': confusion_matrix(all_y_true, all_y_pred)
            }
            
            print(f"      Accuracy: {accuracy:.4f} (±{np.std(fold_scores):.4f})")
            print(f"      F1-Score: {f1:.4f}")
            print(f"      Precision: {precision:.4f}")
            print(f"      Recall: {recall:.4f}")
        
        # 최고 성능 모델 선택
        best_name = max(self.results.keys(), key=lambda x: self.results[x]['f1_score'])
        self.best_model_name = best_name
        print(f"\n   [최고 성능 모델: {best_name}]")
        print(f"      F1-Score: {self.results[best_name]['f1_score']:.4f}")
        
        return self.results
    
    def _clone_model(self, model):
        """모델 복제"""
        from sklearn.base import clone
        return clone(model)
    
    def train_final_model(self):
        """
        전체 데이터로 최종 모델 학습
        
        Returns:
            tuple: (trained_model, scaler)
        """
        print("\n" + "=" * 60)
        print("4. 최종 모델 학습 (전체 데이터)")
        print("=" * 60)
        
        # 최고 성능 모델로 전체 데이터 학습
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(self.X)
        
        self.best_model = self._clone_model(self.models[self.best_model_name])
        self.best_model.fit(X_scaled, self.y)
        
        print(f"   모델: {self.best_model_name}")
        print(f"   학습 샘플 수: {len(self.X)}")
        
        # 모델 저장
        model_path = os.path.join(self.output_dir, 'trained_model.joblib')
        scaler_path = os.path.join(self.output_dir, 'scaler.joblib')
        encoder_path = os.path.join(self.output_dir, 'label_encoder.joblib')
        
        joblib.dump(self.best_model, model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.label_encoder, encoder_path)
        
        print(f"   모델 저장: {model_path}")
        print(f"   스케일러 저장: {scaler_path}")
        print(f"   레이블 인코더 저장: {encoder_path}")
        
        return self.best_model, self.scaler
    
    def validate_all_subjects(self):
        """
        모든 피험자에 대해 개별 검증 수행
        
        Returns:
            pd.DataFrame: 피험자별 검증 결과
        """
        print("\n" + "=" * 60)
        print("5. 피험자별 개별 검증")
        print("=" * 60)
        
        results_list = []
        
        for subject in self.SUBJECTS:
            # 해당 피험자 데이터 추출
            mask = self.subjects == subject
            if not np.any(mask):
                print(f"   [{subject}] 데이터 없음")
                continue
            
            X_subject = self.X[mask]
            y_subject = self.y[mask]
            
            # 나머지 피험자로 학습
            X_train = self.X[~mask]
            y_train = self.y[~mask]
            
            # 스케일링
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_subject_scaled = scaler.transform(X_subject)
            
            # 최고 성능 모델로 학습 및 예측
            model = self._clone_model(self.models[self.best_model_name])
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_subject_scaled)
            
            # 결과 계산
            accuracy = accuracy_score(y_subject, y_pred)
            
            # 개별 단계 정확도
            phase_results = {'subject': subject, 'accuracy': accuracy}
            for phase_idx, phase_name in enumerate(self.label_encoder.classes_):
                phase_mask = y_subject == phase_idx
                if np.any(phase_mask):
                    phase_acc = accuracy_score(y_subject[phase_mask], y_pred[phase_mask])
                    phase_results[f'{phase_name}_correct'] = phase_acc
                else:
                    phase_results[f'{phase_name}_correct'] = np.nan
            
            results_list.append(phase_results)
            
            print(f"   [{subject}] Accuracy: {accuracy:.4f}")
            for phase_idx, phase_name in enumerate(self.label_encoder.classes_):
                phase_mask = y_subject == phase_idx
                if np.any(phase_mask):
                    pred_phase = self.label_encoder.inverse_transform(y_pred[phase_mask])
                    print(f"      {phase_name}: 예측 → {pred_phase}")
        
        subject_results = pd.DataFrame(results_list)
        subject_results.to_csv(os.path.join(self.output_dir, 'subject_validation_results.csv'), index=False)
        
        return subject_results
    
    def generate_visualizations(self):
        """
        결과 시각화 생성
        """
        print("\n" + "=" * 60)
        print("6. 시각화 생성")
        print("=" * 60)
        
        # 1. 모델 비교 차트
        self._plot_model_comparison()
        
        # 2. 혼동 행렬
        self._plot_confusion_matrix()
        
        # 3. 특징 중요도 (Random Forest인 경우)
        if 'Random Forest' in self.models:
            self._plot_feature_importance()
        
        # 4. 피험자별 결과
        self._plot_subject_results()
        
        print("   시각화 완료")
    
    def _plot_model_comparison(self):
        """모델 비교 막대 그래프"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        models = list(self.results.keys())
        metrics = ['accuracy', 'f1_score', 'precision', 'recall']
        x = np.arange(len(models))
        width = 0.2
        
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
        
        for i, metric in enumerate(metrics):
            values = [self.results[m][metric] for m in models]
            bars = ax.bar(x + i*width, values, width, label=metric.replace('_', ' ').title(), color=colors[i])
            
            # 값 표시
            for bar, val in zip(bars, values):
                ax.annotate(f'{val:.2f}', 
                           xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           xytext=(0, 3), textcoords='offset points',
                           ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison (LOSO Cross-Validation)')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend(loc='lower right')
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'model_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("   - model_comparison.png 저장됨")
    
    def _plot_confusion_matrix(self):
        """혼동 행렬 시각화"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # 논리적 순서 및 새로운 레이블 정의
        logical_order = ['BASELINE', 'TASK', 'PUZZLE', 'CHASE']
        new_labels = ['Phase 0', 'Phase 1', 'Phase 2', 'Phase 3']
        
        # 현재 인코더의 클래스 순서
        current_classes = list(self.label_encoder.classes_)
        
        # 인덱스 매핑 생성 (logical_order의 각 항목이 current_classes에서 어디에 있는지)
        order_indices = [current_classes.index(label) for label in logical_order]
        
        for idx, (model_name, result) in enumerate(self.results.items()):
            if idx >= len(axes):
                break
                
            cm = result['confusion_matrix']
            
            # 행렬 재정렬 (행과 열 모두 logical_order 순서로)
            cm_reordered = cm[order_indices][:, order_indices]
            
            # 정규화
            cm_normalized = cm_reordered.astype('float') / cm_reordered.sum(axis=1)[:, np.newaxis]
            
            # 정규화 (0으로 나누기 방지)
            row_sums = cm_reordered.sum(axis=1)
            # 0인 합계는 1로 대체하여 나누기 오류 방지 (해당 행은 어차피 0이 됨)
            row_sums_safe = row_sums.copy()
            row_sums_safe[row_sums_safe == 0] = 1
            cm_normalized = cm_reordered.astype('float') / row_sums_safe[:, np.newaxis]
            
            cm_plot = np.nan_to_num(cm_normalized, nan=0.0, posinf=0.0, neginf=0.0)
            # 히트맵 그리기 (주석은 직접 텍스트로 덧붙임)
            sns.heatmap(
                cm_plot,
                annot=False,
                cmap='Blues',
                vmin=0,
                vmax=1,
                cbar=False,
                xticklabels=new_labels,
                yticklabels=new_labels,
                ax=axes[idx])

            # 셀별 값 강제 표시 (좌표 기반 수동 렌더링)
            for i in range(cm_plot.shape[0]):
                for j in range(cm_plot.shape[1]):
                    val = cm_plot[i, j]
                    text_color = 'white' if val >= 0.65 else 'black'
                    axes[idx].text(
                        j + 0.5,
                        i + 0.5,
                        f'{val:.2f}',
                        ha='center',
                        va='center',
                        fontsize=10,
                        fontweight='bold',
                        color=text_color
                    )
            axes[idx].set_title(f'{model_name}\n(Accuracy: {result["accuracy"]:.3f})')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        # 빈 서브플롯 숨기기
        for idx in range(len(self.results), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Confusion Matrices (Normalized)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrices.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("   - confusion_matrices.png 저장됨")
    
    def _plot_feature_importance(self):
        """Random Forest 특징 중요도"""
        # 전체 데이터로 Random Forest 학습
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_scaled, self.y)
        
        # 특징 중요도 추출
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(self.FEATURE_COLUMNS)))
        
        bars = ax.barh(range(len(self.FEATURE_COLUMNS)), 
                       importances[indices][::-1],
                       color=colors)
        
        ax.set_yticks(range(len(self.FEATURE_COLUMNS)))
        ax.set_yticklabels([self.FEATURE_COLUMNS[i] for i in indices[::-1]])
        ax.set_xlabel('Feature Importance')
        ax.set_title('Feature Importance (Random Forest)')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'feature_importance.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("   - feature_importance.png 저장됨")
    
    def _plot_subject_results(self):
        """피험자별 결과 시각화"""
        # LOSO 결과에서 피험자별 정확도 추출
        logo = LeaveOneGroupOut()
        subject_accuracy = {}
        
        model = self._clone_model(self.models[self.best_model_name])
        
        for train_idx, test_idx in logo.split(self.X, self.y, self.subjects):
            subject = self.subjects[test_idx[0]]
            
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model_clone = self._clone_model(model)
            model_clone.fit(X_train_scaled, y_train)
            y_pred = model_clone.predict(X_test_scaled)
            
            subject_accuracy[subject] = accuracy_score(y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        subjects = list(subject_accuracy.keys())
        accuracies = list(subject_accuracy.values())
        
        colors = ['#3498db' if acc >= 0.5 else '#e74c3c' for acc in accuracies]
        
        bars = ax.bar(subjects, accuracies, color=colors, edgecolor='black', linewidth=1)
        
        # 평균선
        mean_acc = np.mean(accuracies)
        ax.axhline(y=mean_acc, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_acc:.3f}')
        
        # 값 표시
        for bar, acc in zip(bars, accuracies):
            ax.annotate(f'{acc:.2f}', 
                       xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 3), textcoords='offset points',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Subject')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Per-Subject Accuracy ({self.best_model_name})')
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'subject_accuracy.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("   - subject_accuracy.png 저장됨")
    
    def generate_report(self):
        """
        논문용 상세 보고서 생성
        """
        print("\n" + "=" * 60)
        print("7. 논문용 보고서 생성")
        print("=" * 60)
        
        report_path = os.path.join(self.output_dir, 'ML_Classification_Report.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# BITalino 실험 단계 분류 기계학습 분석 보고서\n\n")
            f.write(f"**생성 일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("---\n\n")
            
            # 1. 요약
            f.write("## 1. 요약 (Abstract)\n\n")
            f.write("본 연구에서는 BITalino 생체신호 측정 장치를 사용하여 수집된 ")
            f.write("ECG(심전도), EMG(근전도), EDA(피부전도도) 데이터를 기반으로 ")
            f.write("실험 단계를 자동으로 분류하는 기계학습 모델을 개발하였습니다.\n\n")
            f.write(f"- **총 피험자 수**: {len(self.SUBJECTS)}명\n")
            f.write(f"- **분류 대상 단계**: {', '.join(self.PHASE_LABELS)}\n")
            f.write(f"- **사용 특징 수**: {len(self.FEATURE_COLUMNS)}개\n")
            f.write(f"- **최고 성능 모델**: {self.best_model_name}\n")
            f.write(f"- **LOSO CV 정확도**: {self.results[self.best_model_name]['accuracy']:.4f}\n")
            f.write(f"- **F1-Score**: {self.results[self.best_model_name]['f1_score']:.4f}\n\n")
            
            # 2. 방법론
            f.write("## 2. 방법론 (Methodology)\n\n")
            
            f.write("### 2.1 데이터셋\n\n")
            f.write("| 피험자 | 샘플 수 |\n")
            f.write("|--------|--------|\n")
            for subject in self.SUBJECTS:
                count = np.sum(self.subjects == subject) if self.subjects is not None else 0
                f.write(f"| {subject} | {count} |\n")
            f.write(f"| **합계** | **{len(self.raw_data)}** |\n\n")
            
            f.write("### 2.2 특징 (Features)\n\n")
            f.write("총 15개의 생리학적 특징을 추출하였습니다:\n\n")
            f.write("#### ECG 특징 (심전도)\n")
            f.write("- `ECG_heart_rate_bpm`: 심박수 (beats per minute)\n")
            f.write("- `ECG_hrv_rmssd_ms`: 심박변이도 RMSSD (milliseconds)\n")
            f.write("- `ECG_num_r_peaks`: R-peak 개수\n")
            f.write("- `ECG_mean_rr_ms`: 평균 RR 간격 (milliseconds)\n")
            f.write("- `ECG_std_rr_ms`: RR 간격 표준편차 (milliseconds)\n\n")
            
            f.write("#### EMG 특징 (근전도)\n")
            f.write("- `EMG_mean_uV`: 평균 근전도 값 (microvolts)\n")
            f.write("- `EMG_std_uV`: 근전도 표준편차 (microvolts)\n")
            f.write("- `EMG_max_uV`: 최대 근전도 값 (microvolts)\n")
            f.write("- `EMG_rms_uV`: 근전도 RMS 값 (microvolts)\n")
            f.write("- `EMG_integrated_emg`: 적분 근전도 값\n\n")
            
            f.write("#### EDA 특징 (피부전도도)\n")
            f.write("- `EDA_mean_SCL_uS`: 평균 피부 전도도 레벨 (microsiemens)\n")
            f.write("- `EDA_std_SCL_uS`: 피부 전도도 표준편차 (microsiemens)\n")
            f.write("- `EDA_mean_SCR_uS`: 평균 피부 전도 반응 (microsiemens)\n")
            f.write("- `EDA_max_SCR_uS`: 최대 피부 전도 반응 (microsiemens)\n")
            f.write("- `EDA_num_SCR_peaks`: 피부 전도 반응 피크 수\n\n")
            
            f.write("### 2.3 교차 검증 전략\n\n")
            f.write("**Leave-One-Subject-Out (LOSO) 교차 검증**을 사용하였습니다.\n\n")
            f.write("- 한 피험자의 모든 데이터를 테스트 셋으로 사용\n")
            f.write("- 나머지 피험자 데이터로 모델 학습\n")
            f.write("- 피험자 간 개인차를 고려한 일반화 성능 평가\n")
            f.write("- 총 8회의 교차 검증 수행\n\n")
            
            f.write("### 2.4 분류 알고리즘\n\n")
            f.write("5가지 기계학습 알고리즘을 비교 평가하였습니다:\n\n")
            
            f.write("#### 1) Random Forest (랜덤 포레스트)\n")
            f.write("- 다수의 결정 트리를 학습하여 앙상블하는 방법\n")
            f.write("- 과적합 방지에 효과적이며 특징 중요도 분석 가능\n")
            f.write("- 파라미터: `n_estimators=100`, `max_depth=10`\n\n")
            
            f.write("#### 2) Gradient Boosting (그래디언트 부스팅)\n")
            f.write("- 순차적으로 약한 학습기를 추가하여 오차를 줄여나가는 방식\n")
            f.write("- 높은 예측 성능\n")
            f.write("- 파라미터: `n_estimators=100`, `learning_rate=0.1`, `max_depth=5`\n\n")
            
            f.write("#### 3) Support Vector Machine (서포트 벡터 머신)\n")
            f.write("- 고차원 공간에서 최적의 결정 경계를 찾는 방법\n")
            f.write("- RBF(Radial Basis Function) 커널 사용\n")
            f.write("- 파라미터: `kernel='rbf'`, `C=1.0`, `gamma='scale'`\n\n")
            
            f.write("#### 4) k-Nearest Neighbors (k-최근접 이웃)\n")
            f.write("- 새로운 샘플과 가장 가까운 k개 이웃의 다수결로 분류\n")
            f.write("- 파라미터: `n_neighbors=5`, `weights='distance'`\n\n")
            
            f.write("#### 5) Logistic Regression (로지스틱 회귀)\n")
            f.write("- 선형 분류기로 베이스라인 모델로 사용\n")
            f.write("- 파라미터: `C=1.0`, `multi_class='multinomial'`\n\n")
            
            # 3. 결과
            f.write("## 3. 결과 (Results)\n\n")
            
            f.write("### 3.1 모델 성능 비교\n\n")
            f.write("| 모델 | Accuracy | F1-Score | Precision | Recall |\n")
            f.write("|------|----------|----------|-----------|--------|\n")
            for model_name, result in self.results.items():
                f.write(f"| {model_name} | {result['accuracy']:.4f} | ")
                f.write(f"{result['f1_score']:.4f} | ")
                f.write(f"{result['precision']:.4f} | ")
                f.write(f"{result['recall']:.4f} |\n")
            f.write("\n")
            
            f.write(f"**최고 성능 모델**: {self.best_model_name}\n\n")
            
            f.write("### 3.2 혼동 행렬\n\n")
            best_cm = self.results[self.best_model_name]['confusion_matrix']
            f.write(f"**{self.best_model_name}** 모델의 혼동 행렬:\n\n")
            f.write("| | Predicted |\n")
            f.write("|---|" + "|".join(self.label_encoder.classes_) + "|\n")
            f.write("|---|" + "|".join(["---"] * len(self.label_encoder.classes_)) + "|\n")
            for i, phase in enumerate(self.label_encoder.classes_):
                f.write(f"| **{phase}** |")
                f.write("|".join([str(best_cm[i, j]) for j in range(len(self.label_encoder.classes_))]))
                f.write("|\n")
            f.write("\n")
            
            f.write("### 3.3 상세 분류 보고서\n\n")
            f.write("```\n")
            f.write(classification_report(
                self.results[self.best_model_name]['y_true'],
                self.results[self.best_model_name]['y_pred'],
                target_names=self.label_encoder.classes_
            ))
            f.write("```\n\n")
            
            # 4. 생성된 파일
            f.write("## 4. 생성된 파일\n\n")
            f.write("| 파일명 | 설명 |\n")
            f.write("|--------|------|\n")
            f.write("| `model_comparison.png` | 모델별 성능 비교 차트 |\n")
            f.write("| `confusion_matrices.png` | 각 모델의 혼동 행렬 |\n")
            f.write("| `feature_importance.png` | 특징 중요도 (Random Forest) |\n")
            f.write("| `subject_accuracy.png` | 피험자별 정확도 |\n")
            f.write("| `trained_model.joblib` | 학습된 최종 모델 |\n")
            f.write("| `scaler.joblib` | 특징 스케일러 |\n")
            f.write("| `label_encoder.joblib` | 레이블 인코더 |\n")
            f.write("| `subject_validation_results.csv` | 피험자별 검증 결과 |\n\n")
            
            # 5. 결론
            f.write("## 5. 결론 (Conclusion)\n\n")
            f.write(f"본 연구에서 {self.best_model_name} 모델이 ")
            f.write(f"LOSO 교차 검증에서 {self.results[self.best_model_name]['accuracy']:.1%}의 정확도와 ")
            f.write(f"{self.results[self.best_model_name]['f1_score']:.1%}의 F1-Score를 달성하였습니다.\n\n")
            f.write("이는 BITalino 센서에서 추출된 생리학적 특징만으로도 ")
            f.write("실험 단계를 효과적으로 분류할 수 있음을 보여줍니다.\n\n")
            
            f.write("### 향후 연구 방향\n\n")
            f.write("1. 더 많은 피험자 데이터 수집을 통한 모델 일반화 성능 향상\n")
            f.write("2. 딥러닝 기반 시계열 분석 방법 적용 (LSTM, Transformer 등)\n")
            f.write("3. 실시간 분류 시스템 개발\n")
            f.write("4. 다른 생리학적 신호 추가 (호흡, 체온 등)\n\n")
            
            f.write("---\n\n")
            f.write(f"*본 보고서는 자동 생성되었습니다. ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})*\n")
        
        print(f"   보고서 저장: {report_path}")
        
        # 텍스트 형식 상세 보고서도 생성
        txt_report_path = os.path.join(self.output_dir, 'classification_report.txt')
        with open(txt_report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("BITalino Phase Classification Report\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Best Model: {self.best_model_name}\n")
            f.write(f"Accuracy: {self.results[self.best_model_name]['accuracy']:.4f}\n")
            f.write(f"F1-Score: {self.results[self.best_model_name]['f1_score']:.4f}\n\n")
            
            f.write("Classification Report:\n")
            f.write("-" * 60 + "\n")
            f.write(classification_report(
                self.results[self.best_model_name]['y_true'],
                self.results[self.best_model_name]['y_pred'],
                target_names=self.label_encoder.classes_
            ))
        
        print(f"   텍스트 보고서 저장: {txt_report_path}")
        
        return report_path
    
    def run(self):
        """
        전체 분석 파이프라인 실행
        """
        print("\n" + "=" * 60)
        print("BITalino 실험 단계 분류 기계학습 분석")
        print("=" * 60)
        print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 1. 데이터 로드
        self.load_data()
        
        # 2. 특징 준비
        self.prepare_features()
        
        # 3. 모델 학습 및 평가
        self.train_and_evaluate()
        
        # 4. 최종 모델 학습
        self.train_final_model()
        
        # 5. 피험자별 검증
        self.validate_all_subjects()
        
        # 6. 시각화 생성
        self.generate_visualizations()
        
        # 7. 보고서 생성
        self.generate_report()
        
        print("\n" + "=" * 60)
        print("분석 완료")
        print("=" * 60)
        print(f"결과 저장 위치: {self.output_dir}")
        print(f"종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return self.results


def main():
    """메인 함수"""
    classifier = MLPhaseClassifier()
    results = classifier.run()
    return results


if __name__ == '__main__':
    main()
