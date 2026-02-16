#!/usr/bin/env python3
"""
BITalino 센서 데이터 동기화 및 시각화/분석 스크립트

기능:
1. 이벤트 로그와 센서 데이터 동기화
2. 실험 단계별 (Baseline, Task, Puzzle, Chase) 시각화
3. 논문용 전처리 및 통계 분석
4. 결과 파일 저장 (CSV, PNG)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from datetime import datetime
from scipy import signal
from scipy.stats import zscore
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정 (macOS)
plt.rcParams['font.family'] = ['AppleGothic', 'Malgun Gothic', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


class SensorDataAnalyzer:
    """BITalino 센서 데이터 분석 클래스"""
    
    # 실험 단계 정의
    PHASES = {
        'BASELINE': ('PHASE0_BASELINE_START', 'PHASE0_BASELINE_END'),
        'TASK': ('PHASE1_TASK_START', 'PHASE1_TASK_END'),
        'PUZZLE': ('PHASE2_SCENE_START', 'PHASE3_START'),
        'CHASE': ('CHASE_START', 'EXPERIMENT_END')
    }
    
    # 채널 정의
    CHANNELS = {
        'EMG': 'A1_(EMG)',
        'EDA': 'A2_(EDA)',
        'ECG': 'A3_(ECG)'
    }
    
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.subjects = {}
        self.output_dir = self.base_dir / 'analysis_output'
        self.output_dir.mkdir(exist_ok=True)
        
    def find_subject_folders(self):
        """피험자 폴더 찾기"""
        folders = []
        for item in self.base_dir.iterdir():
            if item.is_dir():
                events_files = list(item.glob('events_log_*.csv'))
                sensor_files = list(item.glob('sensor_data_*.csv'))
                if events_files and sensor_files:
                    folders.append(item.name)
        return sorted(folders)
    
    def load_events(self, subject_folder):
        """이벤트 로그 파일 로드"""
        folder_path = self.base_dir / subject_folder
        events_files = list(folder_path.glob('events_log_*.csv'))
        
        if not events_files:
            return None
        
        events_file = events_files[0]
        df = pd.read_csv(events_file, encoding='utf-8-sig')
        
        # 컬럼명 정리
        df.columns = df.columns.str.strip()
        
        # 빈 행 제거 및 정리
        df = df.dropna(subset=['label'])
        df = df[df['label'].str.strip() != '']
        
        # 데이터 타입 변환
        df['unix_time'] = pd.to_numeric(df['unix_time'], errors='coerce')
        df['sample_index'] = pd.to_numeric(df['sample_index'], errors='coerce')
        df['label'] = df['label'].str.strip()
        
        return df.dropna(subset=['unix_time'])
    
    def load_sensor_data(self, subject_folder):
        """센서 데이터 파일 로드"""
        folder_path = self.base_dir / subject_folder
        sensor_files = list(folder_path.glob('sensor_data_*.csv'))
        
        if not sensor_files:
            return None
        
        sensor_file = sensor_files[0]
        
        # 헤더 먼저 읽어서 예상 컬럼 수 확인
        with open(sensor_file, 'r', encoding='utf-8-sig') as f:
            header_line = f.readline().strip()
        
        expected_cols = [c.strip() for c in header_line.split(',')]
        num_expected = len(expected_cols)
        
        # 실제 데이터 컬럼 수가 다를 수 있으므로 헤더 수만큼만 사용
        df = pd.read_csv(sensor_file, encoding='utf-8-sig', 
                        usecols=range(num_expected), 
                        names=expected_cols, 
                        skiprows=1)
        
        # 컬럼명 정리
        df.columns = df.columns.str.strip()
        
        return df
    
    def get_phase_indices(self, events_df, phase_name):
        """실험 단계의 시작/끝 sample_index 가져오기"""
        if phase_name not in self.PHASES:
            return None, None
        
        start_label, end_label = self.PHASES[phase_name]
        
        start_row = events_df[events_df['label'] == start_label]
        end_row = events_df[events_df['label'] == end_label]
        
        if start_row.empty or end_row.empty:
            return None, None
        
        start_idx = int(start_row.iloc[0]['sample_index'])
        end_idx = int(end_row.iloc[0]['sample_index'])
        
        return start_idx, end_idx
    
    def extract_phase_data(self, sensor_df, events_df, phase_name):
        """특정 단계의 센서 데이터 추출"""
        start_idx, end_idx = self.get_phase_indices(events_df, phase_name)
        
        if start_idx is None or end_idx is None:
            return None
        
        # sample_index 기준으로 필터링
        phase_data = sensor_df[(sensor_df['sample_index'] >= start_idx) & 
                               (sensor_df['sample_index'] <= end_idx)].copy()
        
        return phase_data
    
    def preprocess_signal(self, data, channel_type):
        """신호 전처리 (채널 타입에 따른 필터링)"""
        if data is None or len(data) == 0:
            return data
        
        data = np.array(data, dtype=float)
        
        # NaN 처리
        if np.any(np.isnan(data)):
            data = pd.Series(data).interpolate().values
        
        fs = 1000  # 샘플링 레이트
        
        if channel_type == 'EMG':
            # EMG: 20-450Hz 밴드패스 필터 + 전파정류 + 이동평균
            nyq = fs / 2
            low = 20 / nyq
            high = min(450 / nyq, 0.99)
            b, a = signal.butter(4, [low, high], btype='band')
            filtered = signal.filtfilt(b, a, data)
            rectified = np.abs(filtered)
            # 50ms 이동평균
            window = int(fs * 0.05)
            kernel = np.ones(window) / window
            envelope = np.convolve(rectified, kernel, mode='same')
            return envelope
            
        elif channel_type == 'EDA':
            # EDA: 0.05Hz 로우패스 필터
            nyq = fs / 2
            cutoff = 0.5 / nyq
            b, a = signal.butter(4, cutoff, btype='low')
            return signal.filtfilt(b, a, data)
            
        elif channel_type == 'ECG':
            # ECG: 0.5-40Hz 밴드패스 필터
            nyq = fs / 2
            low = 0.5 / nyq
            high = 40 / nyq
            b, a = signal.butter(4, [low, high], btype='band')
            return signal.filtfilt(b, a, data)
        
        return data
    
    def compute_features(self, data, channel_type):
        """특징값 계산"""
        if data is None or len(data) == 0:
            return {}
        
        data = np.array(data, dtype=float)
        data = data[~np.isnan(data)]
        
        if len(data) == 0:
            return {}
        
        features = {
            'mean': np.mean(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data),
            'range': np.max(data) - np.min(data),
            'median': np.median(data),
            'rms': np.sqrt(np.mean(data**2)),
        }
        
        if channel_type == 'EDA':
            # SCL (Skin Conductance Level) - 평균값
            features['SCL'] = np.mean(data)
            # SCR (Skin Conductance Response) - 표준편차로 근사
            features['SCR_proxy'] = np.std(data)
        
        return features
    
    def analyze_subject(self, subject_name):
        """단일 피험자 분석"""
        print(f"\n{'='*60}")
        print(f"Analyzing: {subject_name}")
        print('='*60)
        
        events_df = self.load_events(subject_name)
        sensor_df = self.load_sensor_data(subject_name)
        
        if events_df is None or sensor_df is None:
            print(f"  Error: Could not load data for {subject_name}")
            return None
        
        print(f"  Events: {len(events_df)} rows")
        print(f"  Sensor data: {len(sensor_df)} samples")
        
        results = {
            'subject': subject_name,
            'phases': {}
        }
        
        # 각 단계별 분석
        for phase_name in self.PHASES.keys():
            phase_data = self.extract_phase_data(sensor_df, events_df, phase_name)
            
            if phase_data is None or len(phase_data) == 0:
                print(f"  {phase_name}: No data found")
                continue
            
            print(f"  {phase_name}: {len(phase_data)} samples ({len(phase_data)/1000:.1f}s)")
            
            phase_results = {
                'samples': len(phase_data),
                'duration_sec': len(phase_data) / 1000,
                'channels': {}
            }
            
            # 각 채널 분석
            for ch_name, ch_col in self.CHANNELS.items():
                if ch_col in phase_data.columns:
                    raw_data = phase_data[ch_col].values
                    processed = self.preprocess_signal(raw_data, ch_name)
                    features = self.compute_features(processed, ch_name)
                    
                    phase_results['channels'][ch_name] = {
                        'raw': raw_data,
                        'processed': processed,
                        'features': features
                    }
            
            results['phases'][phase_name] = phase_results
        
        # 추가 이벤트 정보
        results['events'] = events_df
        results['sensor_data'] = sensor_df
        
        self.subjects[subject_name] = results
        return results
    
    def plot_subject_overview(self, subject_name, save=True):
        """피험자 데이터 개요 시각화"""
        if subject_name not in self.subjects:
            self.analyze_subject(subject_name)
        
        results = self.subjects[subject_name]
        events_df = results['events']
        sensor_df = results['sensor_data']
        
        fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
        
        time_sec = (sensor_df['sample_index'] - sensor_df['sample_index'].min()) / 1000
        
        # 색상 정의
        phase_colors = {
            'BASELINE': '#90EE90',  # 연두색
            'TASK': '#FFD700',      # 금색
            'PUZZLE': '#87CEEB',    # 하늘색
            'CHASE': '#FF6B6B'      # 빨간색
        }
        
        # 각 채널 플롯
        for i, (ch_name, ch_col) in enumerate(self.CHANNELS.items()):
            ax = axes[i]
            if ch_col in sensor_df.columns:
                ax.plot(time_sec, sensor_df[ch_col].values, 
                       linewidth=0.3, alpha=0.7, color='navy')
            ax.set_ylabel(ch_name, fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # 단계별 배경색
            for phase_name, color in phase_colors.items():
                start_idx, end_idx = self.get_phase_indices(events_df, phase_name)
                if start_idx is not None and end_idx is not None:
                    start_sec = (start_idx - sensor_df['sample_index'].min()) / 1000
                    end_sec = (end_idx - sensor_df['sample_index'].min()) / 1000
                    ax.axvspan(start_sec, end_sec, alpha=0.2, color=color)
        
        # 이벤트 마커 (마지막 축)
        ax = axes[3]
        ax.set_ylabel('Events', fontsize=11, fontweight='bold')
        
        # 주요 이벤트만 표시
        important_events = ['PHASE0_BASELINE_START', 'PHASE0_BASELINE_END', 
                           'PHASE1_TASK_START', 'PHASE1_TASK_END',
                           'PHASE2_SCENE_START', 'PHASE3_START', 'CHASE_START',
                           'EXPERIMENT_END', 'P1_HIT_CORRECT', 'P1_NO_RESPONSE',
                           'DOOR_OPEN', 'P2_PIECE_1', 'P2_PIECE_2', 'P2_PIECE_3', 'P2_PIECE_4']
        
        event_y = {'DOOR_OPEN': 1, 'P1_HIT_CORRECT': 2, 'P1_NO_RESPONSE': 3, 
                   'P2_PIECE_1': 4, 'P2_PIECE_2': 4, 'P2_PIECE_3': 4, 'P2_PIECE_4': 4}
        
        for _, row in events_df.iterrows():
            label = row['label']
            if label in important_events:
                sample_idx = row['sample_index']
                t = (sample_idx - sensor_df['sample_index'].min()) / 1000
                y = event_y.get(label, 0)
                color = 'red' if 'NO_RESPONSE' in label else 'green'
                ax.scatter(t, y, s=20, color=color, alpha=0.6)
        
        ax.set_ylim(-1, 5)
        ax.set_xlabel('Time (seconds)', fontsize=11)
        
        # 범례
        legend_patches = [mpatches.Patch(color=c, alpha=0.3, label=p) 
                         for p, c in phase_colors.items()]
        axes[0].legend(handles=legend_patches, loc='upper right', ncol=4)
        
        plt.suptitle(f'BITalino Sensor Data Overview - {subject_name}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save:
            output_path = self.output_dir / f'{subject_name}_overview.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {output_path}")
        
        plt.close()
        return fig
    
    def plot_phase_comparison(self, subject_name, save=True):
        """단계별 비교 시각화 (전처리된 데이터)"""
        if subject_name not in self.subjects:
            self.analyze_subject(subject_name)
        
        results = self.subjects[subject_name]
        phases = results['phases']
        
        fig, axes = plt.subplots(3, 4, figsize=(16, 10))
        
        phase_colors = {'BASELINE': '#2E86AB', 'TASK': '#F18F01', 
                       'PUZZLE': '#C73E1D', 'CHASE': '#A23B72'}
        
        for i, (ch_name, _) in enumerate(self.CHANNELS.items()):
            for j, phase_name in enumerate(self.PHASES.keys()):
                ax = axes[i, j]
                
                if phase_name in phases and ch_name in phases[phase_name]['channels']:
                    data = phases[phase_name]['channels'][ch_name]['processed']
                    time_sec = np.arange(len(data)) / 1000
                    
                    ax.plot(time_sec, data, linewidth=0.5, 
                           color=phase_colors[phase_name], alpha=0.8)
                    
                    features = phases[phase_name]['channels'][ch_name]['features']
                    ax.axhline(features.get('mean', 0), color='red', 
                              linestyle='--', alpha=0.5, linewidth=1)
                
                if i == 0:
                    ax.set_title(phase_name, fontweight='bold')
                if j == 0:
                    ax.set_ylabel(ch_name, fontweight='bold')
                if i == 2:
                    ax.set_xlabel('Time (s)')
                
                ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Phase Comparison (Preprocessed) - {subject_name}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save:
            output_path = self.output_dir / f'{subject_name}_phase_comparison.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {output_path}")
        
        plt.close()
        return fig
    
    def export_preprocessed_data(self, subject_name):
        """전처리된 데이터를 CSV로 저장"""
        if subject_name not in self.subjects:
            self.analyze_subject(subject_name)
        
        results = self.subjects[subject_name]
        
        # 단계별 전처리 데이터 저장
        for phase_name, phase_data in results['phases'].items():
            dfs = []
            for ch_name in self.CHANNELS.keys():
                if ch_name in phase_data['channels']:
                    processed = phase_data['channels'][ch_name]['processed']
                    df = pd.DataFrame({
                        f'{ch_name}_raw': phase_data['channels'][ch_name]['raw'],
                        f'{ch_name}_processed': processed
                    })
                    dfs.append(df)
            
            if dfs:
                combined = pd.concat(dfs, axis=1)
                combined['time_sec'] = np.arange(len(combined)) / 1000
                
                output_path = self.output_dir / f'{subject_name}_{phase_name}_preprocessed.csv'
                combined.to_csv(output_path, index=False)
                print(f"  Saved: {output_path}")
        
        return True
    
    def generate_feature_summary(self):
        """모든 피험자의 특징값 요약 테이블 생성"""
        rows = []
        
        for subject_name, results in self.subjects.items():
            for phase_name, phase_data in results['phases'].items():
                row = {
                    'Subject': subject_name,
                    'Phase': phase_name,
                    'Duration_sec': phase_data['duration_sec']
                }
                
                for ch_name in self.CHANNELS.keys():
                    if ch_name in phase_data['channels']:
                        features = phase_data['channels'][ch_name]['features']
                        for feat_name, feat_val in features.items():
                            row[f'{ch_name}_{feat_name}'] = feat_val
                
                rows.append(row)
        
        df = pd.DataFrame(rows)
        
        output_path = self.output_dir / 'all_subjects_features.csv'
        df.to_csv(output_path, index=False, float_format='%.4f')
        print(f"\nSaved feature summary: {output_path}")
        
        return df
    
    def run_full_analysis(self):
        """전체 분석 실행"""
        print("\n" + "="*70)
        print("BITalino Sensor Data Analysis - Full Run")
        print("="*70)
        
        subjects = self.find_subject_folders()
        print(f"\nFound {len(subjects)} subjects: {subjects}")
        
        for subject in subjects:
            # 분석
            self.analyze_subject(subject)
            
            # 시각화
            self.plot_subject_overview(subject)
            self.plot_phase_comparison(subject)
            
            # 전처리 데이터 저장
            self.export_preprocessed_data(subject)
        
        # 전체 요약
        summary_df = self.generate_feature_summary()
        
        print("\n" + "="*70)
        print("Analysis Complete!")
        print(f"Output directory: {self.output_dir}")
        print("="*70)
        
        return summary_df


def main():
    base_dir = Path(__file__).parent
    
    analyzer = SensorDataAnalyzer(base_dir)
    summary = analyzer.run_full_analysis()
    
    print("\n\n" + "="*70)
    print("Feature Summary Preview:")
    print("="*70)
    print(summary.to_string())


if __name__ == '__main__':
    main()
