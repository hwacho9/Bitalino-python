#!/usr/bin/env python3
"""
논문 데이터 검증 스크립트

논문에 기재된 모든 생체 데이터의 정확성을 검증하고,
수정이 필요한 경우 올바른 값을 제공합니다.
"""

import pandas as pd
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt, welch
from scipy.interpolate import interp1d
import os
import glob
import warnings
warnings.filterwarnings('ignore')

# 설정
SAMPLING_RATE = 1000  # Hz
BASE_DIR = '/Users/chosunghwa/Desktop/workspace/Bitalino-python'

# 피험자 목록
SUBJECTS = ['hase1', 'hase2', 'ishikawa', 'masaya', 'matsumoto', 'sensei', 'takamiya', 'takamiya2']

# 피험자 매핑 (논문용)
SUBJECT_MAPPING = {
    'hase1': 'A',
    'hase2': 'B', 
    'ishikawa': 'C',
    'masaya': 'D',
    'matsumoto': 'E1',
    'sensei': 'F',
    'takamiya': 'G',
    'takamiya2': 'E2'
}

# Phase 정의
PHASES = {
    'BASELINE': ('PHASE0_BASELINE_START', 'PHASE0_BASELINE_END'),
    'TASK': ('PHASE1_TASK_START', 'PHASE1_TASK_END'),
    'PUZZLE': ('PHASE2_SCENE_START', 'CHASE_START'),  # PUZZLE은 PHASE2_SCENE_START ~ CHASE_START
    'CHASE': ('CHASE_START', 'EXPERIMENT_END')
}

class BITalinoDataConverter:
    """BITalino RAW 데이터를 물리적 단위로 변환"""
    ADC_RESOLUTION = 10
    VCC = 3.3
    
    @staticmethod
    def raw_to_voltage(raw_value, n_bits=10):
        return (raw_value / (2**n_bits)) * BITalinoDataConverter.VCC
    
    @staticmethod
    def convert_emg(raw_data):
        """EMG: RAW → μV"""
        voltage = BITalinoDataConverter.raw_to_voltage(np.array(raw_data))
        offset = BITalinoDataConverter.VCC / 2
        gain = 1000
        emg_mv = ((voltage - offset) / gain) * 1000
        return np.abs(emg_mv) * 1000  # μV
    
    @staticmethod
    def convert_eda(raw_data):
        """EDA: RAW → μS"""
        voltage = BITalinoDataConverter.raw_to_voltage(np.array(raw_data))
        eda_us = ((voltage / 0.12) - 0.25) * 1000
        return np.clip(eda_us, 0, None)
    
    @staticmethod
    def convert_ecg(raw_data):
        """ECG: RAW → mV"""
        voltage = BITalinoDataConverter.raw_to_voltage(np.array(raw_data))
        offset = BITalinoDataConverter.VCC / 2
        gain = 1100
        ecg_mv = ((voltage - offset) * 1000) / gain
        return ecg_mv


class SignalProcessor:
    """신호 처리 클래스"""
    
    @staticmethod
    def bandpass_filter(data, lowcut, highcut, fs=1000, order=4):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)
    
    @staticmethod
    def lowpass_filter(data, cutoff, fs=1000, order=4):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low')
        return filtfilt(b, a, data)
    
    @staticmethod
    def process_emg(raw_data):
        """EMG 전처리"""
        emg_uv = BITalinoDataConverter.convert_emg(raw_data)
        filtered = SignalProcessor.bandpass_filter(emg_uv, 20, 450)
        rectified = np.abs(filtered)
        envelope = SignalProcessor.lowpass_filter(rectified, 6)
        return {
            'raw': emg_uv,
            'filtered': filtered,
            'envelope': envelope,
            'mean': np.mean(envelope),
            'std': np.std(envelope),
            'max': np.max(envelope),
            'rms': np.sqrt(np.mean(envelope**2))
        }
    
    @staticmethod
    def process_eda(raw_data):
        """EDA 전처리"""
        eda_us = BITalinoDataConverter.convert_eda(raw_data)
        scl = SignalProcessor.lowpass_filter(eda_us, 0.5)
        scr = eda_us - scl
        
        # SCR 피크 검출
        scr_peaks, _ = find_peaks(scr, height=0.01, distance=1000, prominence=0.01)
        
        return {
            'raw': eda_us,
            'scl': scl,
            'scr': scr,
            'mean_scl': np.mean(scl),
            'std_scl': np.std(scl),
            'num_scr_peaks': len(scr_peaks)
        }
    
    @staticmethod
    def process_ecg(raw_data, min_hr=30, max_hr=200):
        """ECG 전처리 및 HRV 분석"""
        ecg_mv = BITalinoDataConverter.convert_ecg(raw_data)
        
        # 밴드패스 필터
        ecg_filtered = SignalProcessor.bandpass_filter(ecg_mv, 5, 40)
        
        # R-peak 검출 - 적응적 threshold
        # 먼저 넓은 범위로 검출 시도
        min_distance = int(60 / max_hr * SAMPLING_RATE)  # 최대 HR 기준 최소 거리
        
        peaks, properties = find_peaks(
            ecg_filtered, 
            distance=min_distance,
            prominence=np.std(ecg_filtered) * 0.3
        )
        
        if len(peaks) < 2:
            # 더 민감한 설정으로 재시도
            peaks, properties = find_peaks(
                ecg_filtered, 
                distance=int(min_distance * 0.8),
                prominence=np.std(ecg_filtered) * 0.2
            )
        
        result = {
            'raw': ecg_mv,
            'filtered': ecg_filtered,
            'r_peaks': peaks,
            'num_r_peaks': len(peaks)
        }
        
        if len(peaks) > 1:
            rr_intervals = np.diff(peaks)  # in samples (ms at 1000Hz)
            
            # 이상치 제거 (생리학적 범위: 300-2000ms)
            valid_rr = rr_intervals[(rr_intervals > 300) & (rr_intervals < 2000)]
            
            if len(valid_rr) > 0:
                mean_rr = np.mean(valid_rr)
                hr = 60000 / mean_rr
                
                result['rr_intervals'] = valid_rr
                result['mean_rr_ms'] = mean_rr
                result['heart_rate_bpm'] = hr
                result['std_rr_ms'] = np.std(valid_rr)
                
                # RMSSD
                if len(valid_rr) > 1:
                    diff_rr = np.diff(valid_rr)
                    result['rmssd_ms'] = np.sqrt(np.mean(diff_rr**2))
                    result['sdnn_ms'] = np.std(valid_rr)
                    
                    # pNN50
                    pnn50 = np.sum(np.abs(diff_rr) > 50) / len(diff_rr) * 100
                    result['pnn50_percent'] = pnn50
                    
                    # LF/HF ratio
                    lf_hf = SignalProcessor._compute_lf_hf_ratio(valid_rr)
                    result.update(lf_hf)
            else:
                result['heart_rate_bpm'] = np.nan
                result['rmssd_ms'] = np.nan
                result['sdnn_ms'] = np.nan
                result['lf_hf_ratio'] = np.nan
        else:
            result['heart_rate_bpm'] = np.nan
            result['rmssd_ms'] = np.nan
            result['sdnn_ms'] = np.nan
            result['lf_hf_ratio'] = np.nan
        
        return result
    
    @staticmethod
    def _compute_lf_hf_ratio(rr_intervals):
        """LF/HF 비율 계산"""
        if len(rr_intervals) < 4:
            return {'lf_power': np.nan, 'hf_power': np.nan, 'lf_hf_ratio': np.nan}
        
        # RR 간격을 시간 축으로 변환
        rr_times = np.cumsum(rr_intervals) / 1000  # seconds
        rr_values = rr_intervals
        
        # 균일한 샘플링으로 보간 (4Hz)
        fs_new = 4.0
        t_interp = np.arange(0, rr_times[-1], 1/fs_new)
        
        if len(t_interp) < 4:
            return {'lf_power': np.nan, 'hf_power': np.nan, 'lf_hf_ratio': np.nan}
        
        try:
            interp_func = interp1d(rr_times, rr_values, kind='cubic', fill_value='extrapolate')
            rr_interp = interp_func(t_interp)
            
            # Welch PSD
            freqs, psd = welch(rr_interp, fs=fs_new, nperseg=min(len(rr_interp), 256))
            
            # LF (0.04-0.15 Hz), HF (0.15-0.40 Hz)
            lf_mask = (freqs >= 0.04) & (freqs < 0.15)
            hf_mask = (freqs >= 0.15) & (freqs <= 0.40)
            
            lf_power = np.trapz(psd[lf_mask], freqs[lf_mask])
            hf_power = np.trapz(psd[hf_mask], freqs[hf_mask])
            
            lf_hf_ratio = lf_power / hf_power if hf_power > 0 else np.nan
            
            return {
                'lf_power': lf_power,
                'hf_power': hf_power,
                'lf_hf_ratio': lf_hf_ratio
            }
        except:
            return {'lf_power': np.nan, 'hf_power': np.nan, 'lf_hf_ratio': np.nan}


def load_subject_data(subject_name):
    """피험자 데이터 로드"""
    subject_dir = os.path.join(BASE_DIR, subject_name)
    
    # 센서 데이터 파일 찾기
    sensor_files = glob.glob(os.path.join(subject_dir, 'sensor_data_*.csv'))
    event_files = glob.glob(os.path.join(subject_dir, 'events_log_*.csv'))
    
    if not sensor_files or not event_files:
        return None, None
    
    sensor_df = pd.read_csv(sensor_files[0])
    sensor_df.columns = sensor_df.columns.str.strip()
    
    events_df = pd.read_csv(event_files[0])
    events_df.columns = events_df.columns.str.strip()
    
    return sensor_df, events_df


def get_phase_data(sensor_df, events_df, phase_name):
    """특정 Phase의 데이터 추출"""
    if phase_name not in PHASES:
        return None
    
    start_label, end_label = PHASES[phase_name]
    
    # 시작/종료 이벤트 찾기
    start_event = events_df[events_df['label'].str.contains(start_label, na=False)]
    end_event = events_df[events_df['label'].str.contains(end_label, na=False)]
    
    if start_event.empty or end_event.empty:
        return None
    
    start_idx = start_event['sample_index'].values[0]
    end_idx = end_event['sample_index'].values[0]
    
    # 데이터 추출
    phase_data = sensor_df[
        (sensor_df['sample_index'] >= start_idx) & 
        (sensor_df['sample_index'] <= end_idx)
    ]
    
    return phase_data


def analyze_subject(subject_name):
    """피험자 분석"""
    sensor_df, events_df = load_subject_data(subject_name)
    
    if sensor_df is None:
        return None
    
    results = {
        'subject': subject_name,
        'subject_code': SUBJECT_MAPPING.get(subject_name, subject_name)
    }
    
    # 컬럼 매핑
    emg_col = 'A1_(EMG)' if 'A1_(EMG)' in sensor_df.columns else 'A1'
    eda_col = 'A2_(EDA)' if 'A2_(EDA)' in sensor_df.columns else 'A2'
    ecg_col = 'A3_(ECG)' if 'A3_(ECG)' in sensor_df.columns else 'A3'
    
    for phase_name in ['BASELINE', 'TASK', 'PUZZLE', 'CHASE']:
        phase_data = get_phase_data(sensor_df, events_df, phase_name)
        
        if phase_data is None or len(phase_data) < 1000:
            results[phase_name] = None
            continue
        
        phase_results = {
            'duration_sec': len(phase_data) / SAMPLING_RATE
        }
        
        # EMG 분석
        if emg_col in phase_data.columns:
            emg_result = SignalProcessor.process_emg(phase_data[emg_col].values)
            phase_results['EMG_mean_uV'] = emg_result['mean']
            phase_results['EMG_std_uV'] = emg_result['std']
            phase_results['EMG_max_uV'] = emg_result['max']
            phase_results['EMG_rms_uV'] = emg_result['rms']
        
        # EDA 분석
        if eda_col in phase_data.columns:
            eda_result = SignalProcessor.process_eda(phase_data[eda_col].values)
            phase_results['EDA_mean_SCL_uS'] = eda_result['mean_scl']
            phase_results['EDA_std_SCL_uS'] = eda_result['std_scl']
            phase_results['EDA_num_SCR_peaks'] = eda_result['num_scr_peaks']
        
        # ECG 분석
        if ecg_col in phase_data.columns:
            ecg_result = SignalProcessor.process_ecg(phase_data[ecg_col].values)
            phase_results['ECG_heart_rate_bpm'] = ecg_result.get('heart_rate_bpm', np.nan)
            phase_results['ECG_rmssd_ms'] = ecg_result.get('rmssd_ms', np.nan)
            phase_results['ECG_sdnn_ms'] = ecg_result.get('sdnn_ms', np.nan)
            phase_results['ECG_lf_hf_ratio'] = ecg_result.get('lf_hf_ratio', np.nan)
            phase_results['ECG_num_r_peaks'] = ecg_result.get('num_r_peaks', 0)
        
        results[phase_name] = phase_results
    
    return results


def generate_verification_report():
    """전체 검증 보고서 생성"""
    print("=" * 80)
    print("논문 데이터 검증 보고서")
    print("=" * 80)
    
    all_results = []
    
    for subject in SUBJECTS:
        print(f"\n분석 중: {subject} ({SUBJECT_MAPPING.get(subject, subject)})")
        result = analyze_subject(subject)
        if result:
            all_results.append(result)
    
    # 결과를 DataFrame으로 변환
    rows = []
    for result in all_results:
        subject = result['subject']
        subject_code = result['subject_code']
        
        for phase in ['BASELINE', 'TASK', 'PUZZLE', 'CHASE']:
            if result.get(phase):
                row = {
                    'Subject': subject,
                    'Subject_Code': subject_code,
                    'Phase': phase,
                    **result[phase]
                }
                rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # CSV 저장
    output_path = os.path.join(BASE_DIR, 'analysis_output', 'verified_data.csv')
    df.to_csv(output_path, index=False)
    print(f"\n검증된 데이터 저장: {output_path}")
    
    # 통계 요약
    print("\n" + "=" * 80)
    print("Phase별 통계 요약 (Mean ± SD)")
    print("=" * 80)
    
    summary_rows = []
    for phase in ['BASELINE', 'TASK', 'PUZZLE', 'CHASE']:
        phase_data = df[df['Phase'] == phase]
        if len(phase_data) == 0:
            continue
        
        summary = {'Phase': phase, 'N': len(phase_data)}
        
        # 각 특징량에 대해 평균±표준편차
        for col in ['ECG_heart_rate_bpm', 'ECG_rmssd_ms', 'ECG_sdnn_ms', 'ECG_lf_hf_ratio',
                    'EDA_mean_SCL_uS', 'EMG_mean_uV', 'EMG_rms_uV']:
            if col in phase_data.columns:
                values = phase_data[col].dropna()
                if len(values) > 0:
                    summary[f'{col}_mean'] = values.mean()
                    summary[f'{col}_std'] = values.std()
        
        summary_rows.append(summary)
    
    summary_df = pd.DataFrame(summary_rows)
    summary_output = os.path.join(BASE_DIR, 'analysis_output', 'verified_summary_stats.csv')
    summary_df.to_csv(summary_output, index=False)
    print(f"통계 요약 저장: {summary_output}")
    
    # 화면에 주요 결과 출력
    print("\n" + "-" * 80)
    print("ECG/HRV 검증 결과")
    print("-" * 80)
    
    for phase in ['BASELINE', 'TASK', 'PUZZLE', 'CHASE']:
        phase_data = df[df['Phase'] == phase]
        if len(phase_data) == 0:
            continue
        
        hr_mean = phase_data['ECG_heart_rate_bpm'].dropna().mean()
        hr_std = phase_data['ECG_heart_rate_bpm'].dropna().std()
        rmssd_mean = phase_data['ECG_rmssd_ms'].dropna().mean()
        rmssd_std = phase_data['ECG_rmssd_ms'].dropna().std()
        sdnn_mean = phase_data['ECG_sdnn_ms'].dropna().mean()
        sdnn_std = phase_data['ECG_sdnn_ms'].dropna().std()
        lf_hf_mean = phase_data['ECG_lf_hf_ratio'].dropna().mean()
        lf_hf_std = phase_data['ECG_lf_hf_ratio'].dropna().std()
        
        print(f"\n{phase} (N={len(phase_data)}):")
        print(f"  HR: {hr_mean:.2f} ± {hr_std:.2f} bpm")
        print(f"  RMSSD: {rmssd_mean:.2f} ± {rmssd_std:.2f} ms")
        print(f"  SDNN: {sdnn_mean:.2f} ± {sdnn_std:.2f} ms")
        print(f"  LF/HF: {lf_hf_mean:.2f} ± {lf_hf_std:.2f}")
    
    print("\n" + "-" * 80)
    print("EDA 검증 결과")
    print("-" * 80)
    
    for phase in ['BASELINE', 'TASK', 'PUZZLE', 'CHASE']:
        phase_data = df[df['Phase'] == phase]
        if len(phase_data) == 0:
            continue
        
        scl_mean = phase_data['EDA_mean_SCL_uS'].dropna().mean()
        scl_std = phase_data['EDA_mean_SCL_uS'].dropna().std()
        
        print(f"{phase}: SCL = {scl_mean:.2f} ± {scl_std:.2f} μS")
    
    print("\n" + "-" * 80)
    print("EMG 검증 결과")
    print("-" * 80)
    
    for phase in ['BASELINE', 'TASK', 'PUZZLE', 'CHASE']:
        phase_data = df[df['Phase'] == phase]
        if len(phase_data) == 0:
            continue
        
        emg_mean = phase_data['EMG_mean_uV'].dropna().mean()
        emg_std = phase_data['EMG_mean_uV'].dropna().std()
        emg_rms_mean = phase_data['EMG_rms_uV'].dropna().mean()
        emg_rms_std = phase_data['EMG_rms_uV'].dropna().std()
        
        print(f"{phase}: Mean = {emg_mean:.2f} ± {emg_std:.2f} μV, RMS = {emg_rms_mean:.2f} ± {emg_rms_std:.2f} μV")
    
    # 개별 피험자 상세 데이터 (Subject E2, D, F 포함)
    print("\n" + "=" * 80)
    print("개별 피험자 상세 데이터")
    print("=" * 80)
    
    for subject_code in ['E2', 'D', 'F']:
        subject_data = df[df['Subject_Code'] == subject_code]
        if len(subject_data) == 0:
            continue
        
        print(f"\n=== Subject {subject_code} ===")
        for _, row in subject_data.iterrows():
            phase = row['Phase']
            print(f"\n  {phase}:")
            print(f"    Duration: {row.get('duration_sec', 'N/A'):.1f} sec")
            print(f"    HR: {row.get('ECG_heart_rate_bpm', 'N/A'):.1f} bpm" if pd.notna(row.get('ECG_heart_rate_bpm')) else "    HR: N/A")
            print(f"    RMSSD: {row.get('ECG_rmssd_ms', 'N/A'):.2f} ms" if pd.notna(row.get('ECG_rmssd_ms')) else "    RMSSD: N/A")
            print(f"    SCL: {row.get('EDA_mean_SCL_uS', 'N/A'):.2f} μS" if pd.notna(row.get('EDA_mean_SCL_uS')) else "    SCL: N/A")
            print(f"    EMG RMS: {row.get('EMG_rms_uV', 'N/A'):.2f} μV" if pd.notna(row.get('EMG_rms_uV')) else "    EMG RMS: N/A")
    
    return df, summary_df


if __name__ == "__main__":
    df, summary_df = generate_verification_report()
