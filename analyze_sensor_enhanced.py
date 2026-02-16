#!/usr/bin/env python3
"""
BITalino 센서 데이터 분석 - 전처리 시각화 및 PDF 보고서 생성

기능:
1. RAW → 물리적 단위 변환 (EMG: μV, EDA: μS, ECG: mV)
2. 전처리된 데이터로 시각화
3. 딥 리서치용 종합 PDF 보고서 생성
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from datetime import datetime
from scipy import signal
from scipy.stats import zscore, ttest_rel, mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

# 폰트 설정
plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100


class BITalinoDataConverter:
    """BITalino RAW 데이터를 물리적 단위로 변환"""
    
    # BITalino 사양
    ADC_RESOLUTION = 10  # 10-bit ADC
    VCC = 3.3  # 공급 전압 (V)
    
    # 센서별 변환 파라미터 (BITalino 공식 문서 기반)
    SENSOR_PARAMS = {
        'EMG': {
            'gain': 1000,  # EMG 센서 증폭률
            'unit': 'μV',
            'unit_full': 'microvolts',
            'description': 'Electromyography - Muscle electrical activity'
        },
        'EDA': {
            'unit': 'μS',
            'unit_full': 'microsiemens', 
            'description': 'Electrodermal Activity - Skin conductance (stress/arousal)'
        },
        'ECG': {
            'gain': 1100,  # ECG 센서 증폭률
            'offset': 0.5,  # DC offset
            'unit': 'mV',
            'unit_full': 'millivolts',
            'description': 'Electrocardiography - Heart electrical activity'
        }
    }
    
    @staticmethod
    def raw_to_voltage(raw_value, n_bits=10):
        """RAW ADC 값을 전압으로 변환"""
        return (raw_value / (2**n_bits)) * BITalinoDataConverter.VCC
    
    @staticmethod
    def convert_emg(raw_data):
        """EMG: RAW → μV"""
        # EMG 센서 전달 함수: EMG(μV) = ((ADC/2^n - 0.5) * VCC) / G_EMG * 10^6
        voltage = BITalinoDataConverter.raw_to_voltage(raw_data)
        emg_uv = ((voltage - BITalinoDataConverter.VCC/2) / 
                  BITalinoDataConverter.SENSOR_PARAMS['EMG']['gain']) * 1e6
        return emg_uv
    
    @staticmethod
    def convert_eda(raw_data):
        """EDA: RAW → μS"""
        # EDA 센서 전달 함수: EDA(μS) = ((ADC/2^n) * VCC) / 0.12
        voltage = BITalinoDataConverter.raw_to_voltage(raw_data)
        eda_us = voltage / 0.12
        return eda_us
    
    @staticmethod
    def convert_ecg(raw_data):
        """ECG: RAW → mV"""
        # ECG 센서 전달 함수: ECG(mV) = ((ADC/2^n - 0.5) * VCC) / G_ECG * 1000
        voltage = BITalinoDataConverter.raw_to_voltage(raw_data)
        ecg_mv = ((voltage - BITalinoDataConverter.VCC/2) / 
                  BITalinoDataConverter.SENSOR_PARAMS['ECG']['gain']) * 1000
        return ecg_mv


class SignalProcessor:
    """신호 전처리 클래스"""
    
    SAMPLING_RATE = 1000  # Hz
    
    @staticmethod
    def bandpass_filter(data, lowcut, highcut, order=4):
        """밴드패스 필터"""
        nyq = SignalProcessor.SAMPLING_RATE / 2
        low = lowcut / nyq
        high = min(highcut / nyq, 0.99)
        b, a = signal.butter(order, [low, high], btype='band')
        return signal.filtfilt(b, a, data)
    
    @staticmethod
    def lowpass_filter(data, cutoff, order=4):
        """로우패스 필터"""
        nyq = SignalProcessor.SAMPLING_RATE / 2
        normalized_cutoff = cutoff / nyq
        b, a = signal.butter(order, normalized_cutoff, btype='low')
        return signal.filtfilt(b, a, data)
    
    @staticmethod
    def highpass_filter(data, cutoff, order=4):
        """하이패스 필터"""
        nyq = SignalProcessor.SAMPLING_RATE / 2
        normalized_cutoff = cutoff / nyq
        b, a = signal.butter(order, normalized_cutoff, btype='high')
        return signal.filtfilt(b, a, data)
    
    @staticmethod
    def process_emg(raw_data):
        """EMG 전처리: 밴드패스 → 정류 → 엔벨로프"""
        # 1. 물리 단위 변환
        emg_uv = BITalinoDataConverter.convert_emg(raw_data)
        
        # 2. 20-450Hz 밴드패스 필터 (근육 신호 주파수 대역)
        filtered = SignalProcessor.bandpass_filter(emg_uv, 20, 450)
        
        # 3. 전파 정류
        rectified = np.abs(filtered)
        
        # 4. 엔벨로프 추출 (50ms 이동평균)
        window = int(SignalProcessor.SAMPLING_RATE * 0.05)
        kernel = np.ones(window) / window
        envelope = np.convolve(rectified, kernel, mode='same')
        
        return {
            'raw_uv': emg_uv,
            'filtered': filtered,
            'envelope': envelope,
            'unit': 'μV'
        }
    
    @staticmethod
    def process_eda(raw_data):
        """EDA 전처리: 로우패스 → SCL/SCR 분리"""
        # 1. 물리 단위 변환
        eda_us = BITalinoDataConverter.convert_eda(raw_data)
        
        # 2. 0.5Hz 로우패스 필터 (노이즈 제거)
        filtered = SignalProcessor.lowpass_filter(eda_us, 0.5)
        
        # 3. SCL (Tonic) - 매우 느린 성분 (0.05Hz 이하)
        scl = SignalProcessor.lowpass_filter(eda_us, 0.05)
        
        # 4. SCR (Phasic) - SCL을 뺀 나머지
        scr = filtered - scl
        
        return {
            'raw_us': eda_us,
            'filtered': filtered,
            'SCL': scl,
            'SCR': scr,
            'unit': 'μS'
        }
    
    @staticmethod
    def process_ecg(raw_data):
        """ECG 전처리: 밴드패스 → R-peak 검출 → HRV 분석"""
        # 1. 물리 단위 변환
        ecg_mv = BITalinoDataConverter.convert_ecg(raw_data)
        
        # 2. 0.5-40Hz 밴드패스 필터 (ECG 주파수 대역)
        filtered = SignalProcessor.bandpass_filter(ecg_mv, 0.5, 40)
        
        # 3. R-peak 검출을 위한 추가 처리
        # 미분으로 기울기 강조
        diff_ecg = np.diff(filtered)
        diff_ecg = np.append(diff_ecg, diff_ecg[-1])
        
        # 제곱으로 양수화 및 강조
        squared = diff_ecg ** 2
        
        # 이동평균으로 스무딩
        window = int(SignalProcessor.SAMPLING_RATE * 0.15)
        kernel = np.ones(window) / window
        integrated = np.convolve(squared, kernel, mode='same')
        
        # R-peak 검출 (간단한 임계값 방식)
        threshold = np.mean(integrated) + 0.5 * np.std(integrated)
        peaks, _ = signal.find_peaks(integrated, height=threshold, 
                                      distance=int(SignalProcessor.SAMPLING_RATE * 0.5))
        
        # RR 간격 및 HRV 지표 계산
        rr_intervals = np.diff(peaks) / SignalProcessor.SAMPLING_RATE * 1000  # ms
        
        if len(rr_intervals) > 0:
            mean_hr = 60000 / np.mean(rr_intervals)  # bpm
            
            # RMSSD: 연속 RR 간격 차이의 제곱평균제곱근 (부교감신경 지표)
            if len(rr_intervals) > 1:
                rr_diffs = np.diff(rr_intervals)
                hrv_rmssd = np.sqrt(np.mean(rr_diffs**2))
            else:
                hrv_rmssd = 0
            
            # SDNN: NN 간격의 표준편차 (전반적 HRV 지표)
            hrv_sdnn = np.std(rr_intervals)
            
            # pNN50: 50ms 이상 차이나는 연속 RR 간격의 비율 (부교감신경 지표)
            if len(rr_intervals) > 1:
                rr_diffs = np.diff(rr_intervals)
                pnn50 = np.sum(np.abs(rr_diffs) > 50) / len(rr_diffs) * 100  # 퍼센트
            else:
                pnn50 = 0
            
            # LF/HF 비율 계산 (주파수 영역 분석)
            lf_power, hf_power, lf_hf_ratio = SignalProcessor._compute_lf_hf_ratio(rr_intervals)
        else:
            mean_hr = 0
            hrv_rmssd = 0
            hrv_sdnn = 0
            pnn50 = 0
            lf_power = 0
            hf_power = 0
            lf_hf_ratio = 0
        
        return {
            'raw_mv': ecg_mv,
            'filtered': filtered,
            'r_peaks': peaks,
            'rr_intervals_ms': rr_intervals,
            'heart_rate_bpm': mean_hr,
            'hrv_rmssd_ms': hrv_rmssd,
            'hrv_sdnn_ms': hrv_sdnn,
            'pnn50_percent': pnn50,
            'lf_power_ms2': lf_power,
            'hf_power_ms2': hf_power,
            'lf_hf_ratio': lf_hf_ratio,
            'unit': 'mV'
        }
    
    @staticmethod
    def _compute_lf_hf_ratio(rr_intervals):
        """
        RR 간격으로부터 LF/HF 비율 계산
        
        LF (Low Frequency): 0.04-0.15 Hz - 교감+부교감 신경 활동
        HF (High Frequency): 0.15-0.40 Hz - 부교감 신경 활동 (호흡성 동성 부정맥)
        LF/HF 비율: 자율신경계 균형 지표 (높을수록 교감신경 우세 = 스트레스)
        """
        if len(rr_intervals) < 4:
            return 0, 0, 0
        
        # RR 간격을 초 단위로 변환
        rr_sec = rr_intervals / 1000.0
        
        # 불규칙 간격 → 규칙 간격으로 보간 (4Hz 리샘플링)
        # 원래 시간축 생성
        time_orig = np.cumsum(rr_sec) - rr_sec[0]
        
        # 보간을 위한 규칙적인 시간축 (4Hz)
        fs_interp = 4.0  # Hz
        time_interp = np.arange(0, time_orig[-1], 1/fs_interp)
        
        if len(time_interp) < 4:
            return 0, 0, 0
        
        # 선형 보간
        rr_interp = np.interp(time_interp, time_orig, rr_sec)
        
        # 평균 제거 (detrend)
        rr_detrend = rr_interp - np.mean(rr_interp)
        
        # FFT를 이용한 파워 스펙트럼 밀도 계산
        n = len(rr_detrend)
        fft_vals = np.fft.fft(rr_detrend)
        fft_power = np.abs(fft_vals)**2 / n
        freqs = np.fft.fftfreq(n, 1/fs_interp)
        
        # 양의 주파수만 사용
        pos_mask = freqs > 0
        freqs = freqs[pos_mask]
        fft_power = fft_power[pos_mask]
        
        # LF 대역 (0.04-0.15 Hz)
        lf_mask = (freqs >= 0.04) & (freqs < 0.15)
        lf_power = np.trapz(fft_power[lf_mask], freqs[lf_mask]) * 1000**2 if np.any(lf_mask) else 0
        
        # HF 대역 (0.15-0.40 Hz)
        hf_mask = (freqs >= 0.15) & (freqs < 0.40)
        hf_power = np.trapz(fft_power[hf_mask], freqs[hf_mask]) * 1000**2 if np.any(hf_mask) else 0
        
        # LF/HF 비율
        lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 0
        
        return lf_power, hf_power, lf_hf_ratio


class SensorDataAnalyzer:
    """센서 데이터 분석 및 보고서 생성"""
    
    PHASES = {
        'BASELINE': ('PHASE0_BASELINE_START', 'PHASE0_BASELINE_END'),
        'TASK': ('PHASE1_TASK_START', 'PHASE1_TASK_END'),
        'PUZZLE': ('PHASE2_SCENE_START', 'PHASE3_START'),
        'CHASE': ('CHASE_START', 'EXPERIMENT_END')
    }
    
    PHASE_DESCRIPTIONS = {
        'BASELINE': 'Rest period (60s) - Baseline measurement',
        'TASK': 'Go/NoGo Task - Cognitive load & inhibition',
        'PUZZLE': 'Puzzle Phase - Problem solving in horror environment',
        'CHASE': 'Chase Phase - High stress escape scenario'
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
            if item.is_dir() and item.name not in ['analysis_output', 'samples', '__pycache__', 'venv']:
                events_files = list(item.glob('events_log_*.csv'))
                sensor_files = list(item.glob('sensor_data_*.csv'))
                if events_files and sensor_files:
                    folders.append(item.name)
        return sorted(folders)
    
    def load_events(self, subject_folder):
        """이벤트 로그 로드"""
        folder_path = self.base_dir / subject_folder
        events_files = list(folder_path.glob('events_log_*.csv'))
        if not events_files:
            return None
        
        df = pd.read_csv(events_files[0], encoding='utf-8-sig')
        df.columns = df.columns.str.strip()
        df = df.dropna(subset=['label'])
        df['unix_time'] = pd.to_numeric(df['unix_time'], errors='coerce')
        df['sample_index'] = pd.to_numeric(df['sample_index'], errors='coerce')
        df['label'] = df['label'].str.strip()
        return df.dropna(subset=['unix_time'])
    
    def load_sensor_data(self, subject_folder):
        """센서 데이터 로드"""
        folder_path = self.base_dir / subject_folder
        sensor_files = list(folder_path.glob('sensor_data_*.csv'))
        if not sensor_files:
            return None
        
        sensor_file = sensor_files[0]
        with open(sensor_file, 'r', encoding='utf-8-sig') as f:
            header_line = f.readline().strip()
        
        expected_cols = [c.strip() for c in header_line.split(',')]
        num_expected = len(expected_cols)
        
        df = pd.read_csv(sensor_file, encoding='utf-8-sig',
                        usecols=range(num_expected),
                        names=expected_cols,
                        skiprows=1)
        df.columns = df.columns.str.strip()
        return df
    
    def get_phase_indices(self, events_df, phase_name):
        """실험 단계 인덱스 가져오기"""
        if phase_name not in self.PHASES:
            return None, None
        
        start_label, end_label = self.PHASES[phase_name]
        start_row = events_df[events_df['label'] == start_label]
        end_row = events_df[events_df['label'] == end_label]
        
        if start_row.empty or end_row.empty:
            return None, None
        
        return int(start_row.iloc[0]['sample_index']), int(end_row.iloc[0]['sample_index'])
    
    def analyze_subject(self, subject_name):
        """피험자 분석"""
        print(f"\nAnalyzing: {subject_name}")
        
        events_df = self.load_events(subject_name)
        sensor_df = self.load_sensor_data(subject_name)
        
        if events_df is None or sensor_df is None:
            return None
        
        results = {
            'subject': subject_name,
            'phases': {},
            'events': events_df,
            'sensor_data': sensor_df
        }
        
        # 채널 컬럼 확인
        emg_col = 'A1_(EMG)' if 'A1_(EMG)' in sensor_df.columns else None
        eda_col = 'A2_(EDA)' if 'A2_(EDA)' in sensor_df.columns else None
        ecg_col = 'A3_(ECG)' if 'A3_(ECG)' in sensor_df.columns else None
        
        for phase_name in self.PHASES.keys():
            start_idx, end_idx = self.get_phase_indices(events_df, phase_name)
            if start_idx is None:
                continue
            
            phase_data = sensor_df[(sensor_df['sample_index'] >= start_idx) & 
                                   (sensor_df['sample_index'] <= end_idx)].copy()
            
            if len(phase_data) == 0:
                continue
            
            phase_results = {
                'samples': len(phase_data),
                'duration_sec': len(phase_data) / 1000,
                'channels': {}
            }
            
            # EMG 처리
            if emg_col:
                raw_emg = phase_data[emg_col].values
                processed = SignalProcessor.process_emg(raw_emg)
                phase_results['channels']['EMG'] = {
                    'processed': processed,
                    'features': self._compute_emg_features(processed)
                }
            
            # EDA 처리
            if eda_col:
                raw_eda = phase_data[eda_col].values
                processed = SignalProcessor.process_eda(raw_eda)
                phase_results['channels']['EDA'] = {
                    'processed': processed,
                    'features': self._compute_eda_features(processed)
                }
            
            # ECG 처리
            if ecg_col:
                raw_ecg = phase_data[ecg_col].values
                processed = SignalProcessor.process_ecg(raw_ecg)
                phase_results['channels']['ECG'] = {
                    'processed': processed,
                    'features': self._compute_ecg_features(processed)
                }
            
            results['phases'][phase_name] = phase_results
            print(f"  {phase_name}: {len(phase_data)} samples ({len(phase_data)/1000:.1f}s)")
        
        self.subjects[subject_name] = results
        return results
    
    def _compute_emg_features(self, processed):
        """EMG 특징값"""
        envelope = processed['envelope']
        return {
            'mean_uV': np.mean(envelope),
            'std_uV': np.std(envelope),
            'max_uV': np.max(envelope),
            'rms_uV': np.sqrt(np.mean(envelope**2)),
            'integrated_emg': np.trapz(envelope) / 1000  # 적분값
        }
    
    def _compute_eda_features(self, processed):
        """EDA 특징값"""
        return {
            'mean_SCL_uS': np.mean(processed['SCL']),
            'std_SCL_uS': np.std(processed['SCL']),
            'mean_SCR_uS': np.mean(np.abs(processed['SCR'])),
            'max_SCR_uS': np.max(np.abs(processed['SCR'])),
            'num_SCR_peaks': len(signal.find_peaks(processed['SCR'], height=0.01)[0])
        }
    
    def _compute_ecg_features(self, processed):
        """ECG 특징값 (시간 영역 + 주파수 영역 HRV 지표)"""
        return {
            # 기본 심박수
            'heart_rate_bpm': processed['heart_rate_bpm'],
            
            # 시간 영역 HRV 지표
            'hrv_rmssd_ms': processed['hrv_rmssd_ms'],      # RMSSD: 부교감신경 활동 지표
            'hrv_sdnn_ms': processed['hrv_sdnn_ms'],        # SDNN: 전반적 HRV (교감+부교감)
            'pnn50_percent': processed['pnn50_percent'],    # pNN50: 부교감신경 활동 비율
            
            # 주파수 영역 HRV 지표
            'lf_power_ms2': processed['lf_power_ms2'],      # LF 파워: 교감+부교감 (0.04-0.15Hz)
            'hf_power_ms2': processed['hf_power_ms2'],      # HF 파워: 부교감 (0.15-0.40Hz)
            'lf_hf_ratio': processed['lf_hf_ratio'],        # LF/HF 비율: 자율신경 균형 (스트레스 지표)
            
            # R-peak 통계
            'num_r_peaks': len(processed['r_peaks']),
            'mean_rr_ms': np.mean(processed['rr_intervals_ms']) if len(processed['rr_intervals_ms']) > 0 else 0,
            'std_rr_ms': np.std(processed['rr_intervals_ms']) if len(processed['rr_intervals_ms']) > 0 else 0
        }
    
    def plot_processed_overview(self, subject_name, save=True):
        """전처리된 데이터 시각화"""
        if subject_name not in self.subjects:
            self.analyze_subject(subject_name)
        
        results = self.subjects[subject_name]
        
        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        
        phase_colors = {
            'BASELINE': '#2E86AB',
            'TASK': '#F18F01',
            'PUZZLE': '#C73E1D',
            'CHASE': '#A23B72'
        }
        
        for j, phase_name in enumerate(self.PHASES.keys()):
            if phase_name not in results['phases']:
                continue
            
            phase_data = results['phases'][phase_name]
            
            # Row 0: EMG Envelope
            ax = axes[0, j]
            if 'EMG' in phase_data['channels']:
                emg = phase_data['channels']['EMG']['processed']
                time_sec = np.arange(len(emg['envelope'])) / 1000
                ax.plot(time_sec, emg['envelope'], color=phase_colors[phase_name], linewidth=0.5)
                ax.set_title(f"{phase_name}\n({phase_data['duration_sec']:.0f}s)", fontweight='bold')
                features = phase_data['channels']['EMG']['features']
                ax.axhline(features['mean_uV'], color='red', linestyle='--', alpha=0.5)
            if j == 0:
                ax.set_ylabel('EMG Envelope (μV)', fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Row 1: EDA SCL
            ax = axes[1, j]
            if 'EDA' in phase_data['channels']:
                eda = phase_data['channels']['EDA']['processed']
                time_sec = np.arange(len(eda['SCL'])) / 1000
                ax.plot(time_sec, eda['SCL'], color=phase_colors[phase_name], linewidth=1, label='SCL')
                ax.fill_between(time_sec, eda['SCL'] - np.abs(eda['SCR']), 
                               eda['SCL'] + np.abs(eda['SCR']), alpha=0.3, label='SCR')
            if j == 0:
                ax.set_ylabel('EDA SCL (μS)', fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=8)
            
            # Row 2: ECG Filtered
            ax = axes[2, j]
            if 'ECG' in phase_data['channels']:
                ecg = phase_data['channels']['ECG']['processed']
                time_sec = np.arange(len(ecg['filtered'])) / 1000
                ax.plot(time_sec, ecg['filtered'], color=phase_colors[phase_name], linewidth=0.3)
                # R-peak 표시
                if len(ecg['r_peaks']) > 0:
                    r_times = ecg['r_peaks'] / 1000
                    r_values = ecg['filtered'][ecg['r_peaks']]
                    ax.scatter(r_times, r_values, color='red', s=10, zorder=5, label='R-peaks')
                features = phase_data['channels']['ECG']['features']
                ax.text(0.02, 0.98, f"HR: {features['heart_rate_bpm']:.1f} bpm\nHRV: {features['hrv_rmssd_ms']:.1f} ms",
                       transform=ax.transAxes, fontsize=8, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            if j == 0:
                ax.set_ylabel('ECG Filtered (mV)', fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Row 3: Feature Summary Bar
            ax = axes[3, j]
            if 'ECG' in phase_data['channels']:
                ecg_feat = phase_data['channels']['ECG']['features']
                emg_feat = phase_data['channels']['EMG']['features'] if 'EMG' in phase_data['channels'] else {}
                eda_feat = phase_data['channels']['EDA']['features'] if 'EDA' in phase_data['channels'] else {}
                
                # Normalize for visualization
                labels = ['HR\n(bpm)', 'HRV\n(ms)', 'EMG\n(μV)', 'SCL\n(μS)']
                values = [
                    ecg_feat.get('heart_rate_bpm', 0),
                    ecg_feat.get('hrv_rmssd_ms', 0),
                    emg_feat.get('mean_uV', 0),
                    eda_feat.get('mean_SCL_uS', 0)
                ]
                bars = ax.bar(labels, values, color=phase_colors[phase_name], alpha=0.7)
                ax.set_ylabel('Value', fontweight='bold')
                
                # 값 표시
                for bar, val in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                           f'{val:.1f}', ha='center', va='bottom', fontsize=8)
            
            ax.set_xlabel('Features', fontweight='bold')
        
        plt.suptitle(f'Preprocessed Sensor Data - {subject_name}', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        if save:
            output_path = self.output_dir / f'{subject_name}_preprocessed_overview.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {output_path}")
        
        plt.close()
        return fig
    
    def generate_pdf_report(self):
        """종합 PDF 보고서 생성"""
        pdf_path = self.output_dir / 'BITalino_Analysis_Report.pdf'
        
        with PdfPages(pdf_path) as pdf:
            # Page 1: Title & Study Overview
            fig, ax = plt.subplots(figsize=(11.69, 8.27))  # A4 landscape
            ax.axis('off')
            
            title_text = """
BITalino Physiological Sensor Data Analysis Report
═══════════════════════════════════════════════════════════════

Study Overview
───────────────────────────────────────────────────────────────
• Sensors: EMG (Electromyography), EDA (Electrodermal Activity), ECG (Electrocardiography)
• Sampling Rate: 1000 Hz
• ADC Resolution: 10-bit (0-1023)

Data Units & Conversions
───────────────────────────────────────────────────────────────
• EMG: Raw ADC → μV (microvolts) - Muscle electrical activity
  - Preprocessing: 20-450Hz bandpass → Rectification → Envelope extraction
  
• EDA: Raw ADC → μS (microsiemens) - Skin conductance
  - Preprocessing: 0.5Hz lowpass → SCL/SCR decomposition
  - SCL (Tonic): Slow-varying baseline (arousal level)
  - SCR (Phasic): Rapid fluctuations (event-related responses)
  
• ECG: Raw ADC → mV (millivolts) - Heart electrical activity
  - Preprocessing: 0.5-40Hz bandpass → R-peak detection → HR/HRV extraction
  - Heart Rate (HR): beats per minute (bpm)
  - HRV RMSSD: Root Mean Square of Successive RR Differences (ms)

Experimental Phases
───────────────────────────────────────────────────────────────
1. BASELINE (60s): Rest period for baseline physiological measurement
2. TASK: Go/NoGo cognitive task - measures inhibitory control
3. PUZZLE: Problem-solving in horror VR environment
4. CHASE: High-stress escape scenario - peak arousal expected
"""
            ax.text(0.5, 0.95, title_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top', horizontalalignment='center',
                   family='monospace')
            
            ax.text(0.5, 0.05, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                   transform=ax.transAxes, fontsize=8, ha='center')
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Page 2: All Subjects Summary Table
            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            ax.axis('off')
            
            # 데이터 수집
            summary_data = []
            for subj_name, results in self.subjects.items():
                for phase_name, phase_data in results['phases'].items():
                    row = {'Subject': subj_name, 'Phase': phase_name}
                    
                    if 'ECG' in phase_data['channels']:
                        ecg_f = phase_data['channels']['ECG']['features']
                        row['HR (bpm)'] = f"{ecg_f['heart_rate_bpm']:.1f}"
                        row['HRV (ms)'] = f"{ecg_f['hrv_rmssd_ms']:.1f}"
                    
                    if 'EMG' in phase_data['channels']:
                        emg_f = phase_data['channels']['EMG']['features']
                        row['EMG (μV)'] = f"{emg_f['mean_uV']:.2f}"
                    
                    if 'EDA' in phase_data['channels']:
                        eda_f = phase_data['channels']['EDA']['features']
                        row['SCL (μS)'] = f"{eda_f['mean_SCL_uS']:.2f}"
                    
                    summary_data.append(row)
            
            if summary_data:
                df = pd.DataFrame(summary_data)
                
                # 테이블 그리기
                ax.text(0.5, 0.98, 'Summary of All Subjects - Key Features by Phase',
                       transform=ax.transAxes, fontsize=14, fontweight='bold',
                       ha='center', va='top')
                
                table = ax.table(cellText=df.values,
                                colLabels=df.columns,
                                cellLoc='center',
                                loc='center',
                                colColours=['#4472C4']*len(df.columns))
                table.auto_set_font_size(False)
                table.set_fontsize(8)
                table.scale(1.2, 1.5)
                
                # 헤더 스타일
                for j in range(len(df.columns)):
                    table[(0, j)].set_text_props(color='white', fontweight='bold')
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Individual Subject Pages
            for subj_name in self.subjects.keys():
                # Overview plot
                fig = self.plot_processed_overview(subj_name, save=False)
                if fig:
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close()
        
        print(f"\n✓ PDF Report saved: {pdf_path}")
        return pdf_path
    
    def export_for_deep_research(self):
        """딥 리서치용 CSV 내보내기"""
        all_features = []
        
        for subj_name, results in self.subjects.items():
            for phase_name, phase_data in results['phases'].items():
                row = {
                    'subject_id': subj_name,
                    'phase': phase_name,
                    'duration_sec': phase_data['duration_sec']
                }
                
                # ECG features
                if 'ECG' in phase_data['channels']:
                    for k, v in phase_data['channels']['ECG']['features'].items():
                        row[f'ECG_{k}'] = v
                
                # EMG features
                if 'EMG' in phase_data['channels']:
                    for k, v in phase_data['channels']['EMG']['features'].items():
                        row[f'EMG_{k}'] = v
                
                # EDA features
                if 'EDA' in phase_data['channels']:
                    for k, v in phase_data['channels']['EDA']['features'].items():
                        row[f'EDA_{k}'] = v
                
                all_features.append(row)
        
        df = pd.DataFrame(all_features)
        
        # 전체 특징값 저장
        output_path = self.output_dir / 'deep_research_features.csv'
        df.to_csv(output_path, index=False, float_format='%.4f')
        print(f"✓ Deep research CSV saved: {output_path}")
        
        # Phase별 비교 통계
        stats_path = self.output_dir / 'phase_comparison_stats.csv'
        stats_rows = []
        
        for phase in ['BASELINE', 'TASK', 'PUZZLE', 'CHASE']:
            phase_df = df[df['phase'] == phase]
            if len(phase_df) > 0:
                for col in df.columns:
                    if col not in ['subject_id', 'phase', 'duration_sec']:
                        stats_rows.append({
                            'phase': phase,
                            'feature': col,
                            'mean': phase_df[col].mean(),
                            'std': phase_df[col].std(),
                            'min': phase_df[col].min(),
                            'max': phase_df[col].max(),
                            'n': len(phase_df)
                        })
        
        stats_df = pd.DataFrame(stats_rows)
        stats_df.to_csv(stats_path, index=False, float_format='%.4f')
        print(f"✓ Phase comparison stats saved: {stats_path}")
        
        return df
    
    def run_full_analysis(self):
        """전체 분석 실행"""
        print("\n" + "="*70)
        print("BITalino Sensor Data Analysis - Enhanced Version")
        print("="*70)
        
        subjects = self.find_subject_folders()
        print(f"\nFound {len(subjects)} subjects: {subjects}")
        
        for subject in subjects:
            self.analyze_subject(subject)
            self.plot_processed_overview(subject)
        
        # 딥 리서치용 데이터 내보내기
        self.export_for_deep_research()
        
        # PDF 보고서 생성
        self.generate_pdf_report()
        
        print("\n" + "="*70)
        print("Analysis Complete!")
        print(f"Output directory: {self.output_dir}")
        print("="*70)


def main():
    base_dir = Path(__file__).parent
    analyzer = SensorDataAnalyzer(base_dir)
    analyzer.run_full_analysis()


if __name__ == '__main__':
    main()
