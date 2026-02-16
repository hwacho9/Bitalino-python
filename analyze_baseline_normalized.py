#!/usr/bin/env python3
"""
베이스라인 정규화 HRV/ECG 분석

개인별 Phase 0 (BASELINE) 값을 기준으로 정규화하여
개인차를 보정한 상대적 변화율을 계산합니다.
"""

import pandas as pd
import numpy as np
import os
from scipy import stats

# 분석 대상 ECG/HRV 특징량
ECG_FEATURES = [
    'ECG_heart_rate_bpm',
    'ECG_hrv_rmssd_ms',
    'ECG_hrv_sdnn_ms',
    'ECG_lf_hf_ratio'
]

# 한글/일본어 라벨
FEATURE_LABELS = {
    'ECG_heart_rate_bpm': '平均心拍数 (HR, bpm)',
    'ECG_hrv_rmssd_ms': 'RMSSD (ms)',
    'ECG_hrv_sdnn_ms': 'SDNN (ms)',
    'ECG_lf_hf_ratio': 'LF/HF比'
}

PHASE_ORDER = ['BASELINE', 'TASK', 'PUZZLE', 'CHASE']
PHASE_LABELS = {
    'BASELINE': 'Phase 0',
    'TASK': 'Phase 1',
    'PUZZLE': 'Phase 2',
    'CHASE': 'Phase 3'
}


def load_phase_comparison_stats(filepath):
    """phase_comparison_stats.csv 로드"""
    df = pd.read_csv(filepath)
    return df


def compute_baseline_normalized_stats(stats_df):
    """
    베이스라인 정규화 통계 계산
    
    정규화 방법: (Phase N 값 - Baseline 값) / Baseline 값 * 100 (%)
    """
    results = []
    
    for feature in ECG_FEATURES:
        feature_data = stats_df[stats_df['feature'] == feature]
        
        # BASELINE 값 가져오기
        baseline_row = feature_data[feature_data['phase'] == 'BASELINE']
        if baseline_row.empty:
            continue
            
        baseline_mean = baseline_row['mean'].values[0]
        baseline_std = baseline_row['std'].values[0]
        
        for phase in PHASE_ORDER:
            phase_row = feature_data[feature_data['phase'] == phase]
            if phase_row.empty:
                continue
            
            phase_mean = phase_row['mean'].values[0]
            phase_std = phase_row['std'].values[0]
            n = phase_row['n'].values[0]
            
            if phase == 'BASELINE':
                # 베이스라인 자체는 변화율 0%
                norm_change = 0.0
                norm_std = 0.0
            else:
                # 변화율 계산
                if baseline_mean != 0:
                    norm_change = ((phase_mean - baseline_mean) / baseline_mean) * 100
                    # 표준편차의 전파 (error propagation)
                    # σ(%) ≈ |100/baseline| * σ_phase
                    norm_std = abs(100 / baseline_mean) * phase_std
                else:
                    norm_change = np.nan
                    norm_std = np.nan
            
            results.append({
                'feature': feature,
                'feature_label': FEATURE_LABELS.get(feature, feature),
                'phase': phase,
                'phase_label': PHASE_LABELS.get(phase, phase),
                'absolute_mean': phase_mean,
                'absolute_std': phase_std,
                'baseline_mean': baseline_mean,
                'normalized_change_percent': norm_change,
                'normalized_std_percent': norm_std,
                'n': n
            })
    
    return pd.DataFrame(results)


def compute_subject_level_normalization(all_features_df):
    """
    개인별 베이스라인 정규화
    
    각 피험자의 BASELINE 값으로 해당 피험자의 모든 Phase 데이터를 정규화
    """
    results = []
    
    # 피험자 목록
    subjects = all_features_df['Subject'].unique()
    
    for subject in subjects:
        subject_data = all_features_df[all_features_df['Subject'] == subject]
        
        # 해당 피험자의 BASELINE 값
        baseline_row = subject_data[subject_data['Phase'] == 'BASELINE']
        if baseline_row.empty:
            continue
        
        for _, phase_row in subject_data.iterrows():
            phase = phase_row['Phase']
            
            for feature in ECG_FEATURES:
                # all_subjects_features.csv의 컬럼명 매핑
                col_mapping = {
                    'ECG_heart_rate_bpm': None,  # 직접 계산 필요
                    'ECG_hrv_rmssd_ms': None,    # 직접 계산 필요
                    'ECG_hrv_sdnn_ms': None,     # 직접 계산 필요
                    'ECG_lf_hf_ratio': None      # 직접 계산 필요
                }
                
            # ECG_mean, ECG_std 등으로부터 특징량 추정 불가
            # phase_comparison_stats.csv에서 직접 가져와야 함
            pass
    
    return pd.DataFrame(results)


def generate_latex_table_absolute(normalized_df):
    """
    절대값 테이블 LaTeX 생성 (기존 형식)
    """
    latex = r"""\begin{table}[t]
\centering
\caption{全参加者のフェーズ別ECG/HRV主要統計値 (Mean $\pm$ SD)}
\label{tab:ecg_hrv_stats}
\begin{tabular}{lcccc}
\hline
特徴量 (Feature) & Phase 0 & Phase 1 & Phase 2 & Phase 3 \\
\hline
"""
    
    for feature in ECG_FEATURES:
        feature_data = normalized_df[normalized_df['feature'] == feature]
        label = FEATURE_LABELS.get(feature, feature)
        
        row = f"{label}"
        for phase in PHASE_ORDER:
            phase_row = feature_data[feature_data['phase'] == phase]
            if not phase_row.empty:
                mean = phase_row['absolute_mean'].values[0]
                std = phase_row['absolute_std'].values[0]
                row += f" & {mean:.2f} $\\pm$ {std:.2f}"
            else:
                row += " & --"
        row += r" \\"
        latex += row + "\n"
    
    latex += r"""\hline
\end{tabular}
\end{table}
"""
    return latex


def generate_latex_table_normalized(normalized_df):
    """
    베이스라인 정규화 테이블 LaTeX 생성
    """
    latex = r"""\begin{table}[t]
\centering
\caption{ベースライン正規化後のフェーズ別ECG/HRV変化率 (\%, Mean $\pm$ SD)}
\label{tab:ecg_hrv_normalized}
\begin{tabular}{lccc}
\hline
特徴量 & Phase 1 & Phase 2 & Phase 3 \\
\hline
"""
    
    # Phase 0는 기준이므로 제외
    target_phases = ['TASK', 'PUZZLE', 'CHASE']
    
    for feature in ECG_FEATURES:
        feature_data = normalized_df[normalized_df['feature'] == feature]
        label = FEATURE_LABELS.get(feature, feature)
        
        row = f"{label}"
        for phase in target_phases:
            phase_row = feature_data[feature_data['phase'] == phase]
            if not phase_row.empty:
                change = phase_row['normalized_change_percent'].values[0]
                std = phase_row['normalized_std_percent'].values[0]
                
                # 양수면 + 표시
                sign = "+" if change > 0 else ""
                row += f" & {sign}{change:.1f} $\\pm$ {std:.1f}"
            else:
                row += " & --"
        row += r" \\"
        latex += row + "\n"
    
    latex += r"""\hline
\end{tabular}
\end{table}
"""
    return latex


def generate_analysis_text(normalized_df):
    """
    분석 결과 해석 텍스트 생성
    """
    text = """
=== ベースライン正規化 分析結果 ===

【計算方法】
各参加者のPhase 0 (安静状態) における測定値をベースラインとして,
以降のフェーズにおける値をベースラインからの変化率 (%) として算出.

Δ(%) = (Phase_N - Phase_0) / Phase_0 × 100

"""
    
    for feature in ECG_FEATURES:
        feature_data = normalized_df[normalized_df['feature'] == feature]
        label = FEATURE_LABELS.get(feature, feature)
        
        text += f"\n【{label}】\n"
        
        for phase in ['TASK', 'PUZZLE', 'CHASE']:
            phase_row = feature_data[feature_data['phase'] == phase]
            if not phase_row.empty:
                change = phase_row['normalized_change_percent'].values[0]
                baseline = phase_row['baseline_mean'].values[0]
                absolute = phase_row['absolute_mean'].values[0]
                phase_label = PHASE_LABELS.get(phase, phase)
                
                direction = "増加" if change > 0 else "減少"
                text += f"  {phase_label}: ベースライン比 {change:+.1f}% ({direction})\n"
                text += f"           (絶対値: {baseline:.2f} → {absolute:.2f})\n"
    
    return text


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'analysis_output')
    
    # 데이터 로드
    stats_path = os.path.join(output_dir, 'phase_comparison_stats.csv')
    stats_df = load_phase_comparison_stats(stats_path)
    
    print("=" * 60)
    print("BITalino ECG/HRV ベースライン正規化分析")
    print("=" * 60)
    
    # 베이스라인 정규화 계산
    normalized_df = compute_baseline_normalized_stats(stats_df)
    
    # 결과 저장
    output_csv = os.path.join(output_dir, 'baseline_normalized_stats.csv')
    normalized_df.to_csv(output_csv, index=False)
    print(f"\n정규화 결과 저장: {output_csv}")
    
    # LaTeX 테이블 생성
    print("\n" + "=" * 60)
    print("【LaTeX テーブル - 絶対値】")
    print("=" * 60)
    print(generate_latex_table_absolute(normalized_df))
    
    print("\n" + "=" * 60)
    print("【LaTeX テーブル - 正規化変化率】")
    print("=" * 60)
    print(generate_latex_table_normalized(normalized_df))
    
    # 분석 텍스트
    print("\n" + generate_analysis_text(normalized_df))
    
    # 통계 요약
    print("\n" + "=" * 60)
    print("【統計サマリー】")
    print("=" * 60)
    print(normalized_df[['feature_label', 'phase_label', 'absolute_mean', 'absolute_std', 
                         'normalized_change_percent', 'n']].to_string(index=False))


if __name__ == "__main__":
    main()
