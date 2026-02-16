#!/usr/bin/env python3
"""원본 센서 데이터에서 직접 Phase별 통계 계산"""

import pandas as pd
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
import os
import glob

BASE_DIR = '/Users/chosunghwa/Desktop/workspace/Bitalino-python'
SUBJECTS = ['hase1', 'hase2', 'ishikawa', 'masaya', 'matsumoto', 'sensei', 'takamiya', 'takamiya2']
SUBJECT_MAP = {
    'masaya': ('Noguchi', 'A'),
    'hase1': ('Hase', 'B1'),
    'hase2': ('Hase', 'B2'),
    'matsumoto': ('Matsumoto', 'C'),
    'ishikawa': ('Ishikawa', 'D'),
    'takamiya': ('Takamiya', 'E1'),
    'takamiya2': ('Takamiya', 'E2'),
    'sensei': ('Sensei', 'F')
}

PHASES = {
    'BASELINE': ('PHASE0_BASELINE_START', 'PHASE0_BASELINE_END'),
    'TASK': ('PHASE1_TASK_START', 'PHASE1_TASK_END'),
    'PUZZLE': ('PHASE2_SCENE_START', 'CHASE_START'),
    'CHASE': ('CHASE_START', 'EXPERIMENT_END')
}

def bandpass(data, low, high, fs=1000, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, data)

def lowpass(data, cutoff, fs=1000, order=4):
    b, a = butter(order, cutoff/(0.5*fs), btype='low')
    return filtfilt(b, a, data)

def convert_emg(raw):
    vcc, n_bits, gain = 3.3, 10, 1000
    voltage = (raw / (2**n_bits)) * vcc
    emg_mv = ((voltage - vcc/2) / gain) * 1000
    return np.abs(emg_mv) * 1000

def convert_eda(raw):
    vcc, n_bits = 3.3, 10
    voltage = (raw / (2**n_bits)) * vcc
    return voltage / 0.132

def convert_ecg(raw):
    vcc, n_bits, gain = 3.3, 10, 1100
    voltage = (raw / (2**n_bits)) * vcc
    return ((voltage - vcc/2) * 1000) / gain

def analyze_emg(raw):
    if len(raw) == 0:
        return {'mean': np.nan, 'rms': np.nan, 'max': np.nan}
    emg = convert_emg(raw)
    try:
        filt = bandpass(emg, 20, 450)
        env = lowpass(np.abs(filt), 6)
    except:
        env = emg
    return {'mean': np.mean(env), 'rms': np.sqrt(np.mean(env**2)), 'max': np.max(env)}

def analyze_eda(raw):
    if len(raw) == 0:
        return {'scl_mean': np.nan}
    eda = convert_eda(raw)
    try:
        scl = lowpass(eda, 0.5)
    except:
        scl = eda
    return {'scl_mean': np.mean(scl)}

def analyze_ecg(raw):
    ecg = convert_ecg(raw)
    try:
        filt = bandpass(ecg, 5, 40)
    except:
        return {'hr': np.nan, 'rmssd': np.nan, 'sdnn': np.nan}
    
    peaks, _ = find_peaks(filt, distance=300, prominence=np.std(filt)*0.3)
    
    if len(peaks) < 3:
        return {'hr': np.nan, 'rmssd': np.nan, 'sdnn': np.nan}
    
    rr = np.diff(peaks)
    valid = rr[(rr > 300) & (rr < 2000)]
    
    if len(valid) < 2:
        return {'hr': np.nan, 'rmssd': np.nan, 'sdnn': np.nan}
    
    hr = 60000 / np.mean(valid)
    rmssd = np.sqrt(np.mean(np.diff(valid)**2)) if len(valid) > 2 else np.nan
    sdnn = np.std(valid)
    
    return {'hr': hr, 'rmssd': rmssd, 'sdnn': sdnn}

def load_subject(subj):
    subj_dir = os.path.join(BASE_DIR, subj)
    sensor_f = glob.glob(os.path.join(subj_dir, 'sensor_data_*.csv'))
    event_f = glob.glob(os.path.join(subj_dir, 'events_log_*.csv'))
    
    if not sensor_f or not event_f:
        return None, None
    
    sens = pd.read_csv(sensor_f[0])
    sens.columns = sens.columns.str.strip()
    evts = pd.read_csv(event_f[0])
    evts.columns = evts.columns.str.strip()
    return sens, evts

def get_phase(sens, evts, phase, subj_name):
    start_l, end_l = PHASES[phase]
    start_e = evts[evts['label'].str.contains(start_l, na=False)]
    end_e = evts[evts['label'].str.contains(end_l, na=False)]
    
    if start_e.empty or end_e.empty:
        # print(f"[{subj_name}] {phase}: Start({start_l})={len(start_e)}, End({end_l})={len(end_e)}")
        # Try simplified labels for Hase/Matsumoto if needed (e.g., just "BASELINE_START")
        if phase == 'BASELINE' and start_e.empty:
             start_e = evts[evts['label'].str.contains('BASELINE_START', na=False)]
        if phase == 'BASELINE' and end_e.empty:
             end_e = evts[evts['label'].str.contains('BASELINE_END', na=False)]
        
        if start_e.empty or end_e.empty:
             print(f"  FAILED {phase}: Labels not found. Start='{start_l}'(Found={len(start_e)}), End='{end_l}'(Found={len(end_e)})")
             return None
    
    s_idx = start_e['sample_index'].values[0]
    e_idx = end_e['sample_index'].values[0]
    
    # Check sensor range
    sens_min = sens['sample_index'].min()
    sens_max = sens['sample_index'].max()
    
    if s_idx < sens_min or e_idx > sens_max:
        # print(f"  FAILED {phase}: Index out of range. Event=({s_idx}, {e_idx}), Sensor=({sens_min}, {sens_max})")
        # Try to find closest? No, strict matching usually better.
        # But maybe offset?
        pass

    if e_idx <= s_idx:
        print(f"  FAILED {phase}: End <= Start ({e_idx} <= {s_idx})")
        return None
    
    return sens[(sens['sample_index'] >= s_idx) & (sens['sample_index'] <= e_idx)]

def analyze_ecg(raw, fs=1000):
    if len(raw) == 0:
        return {'hr': np.nan, 'rmssd': np.nan, 'sdnn': np.nan}
    
    ecg = convert_ecg(raw)
    
    # 1. Bandpass 5-40Hz
    try:
        filt = bandpass(ecg, 5, 40, fs)
    except:
        return {'hr': np.nan, 'rmssd': np.nan, 'sdnn': np.nan}
    
    # 2. Peak Detection (Prominence based)
    # Lower prominence to catch peaks in noisy data (Takamiya)
    # 0.2 * std might be better than 0.3
    peaks, _ = find_peaks(filt, distance=int(0.25 * fs), prominence=np.std(filt)*0.2)
    
    if len(peaks) < 2:
        return {'hr': np.nan, 'rmssd': np.nan, 'sdnn': np.nan}
    
    rr = np.diff(peaks) / fs * 1000  # ms
    
    # Filter valid RR (300ms ~ 1500ms = 40bpm ~ 200bpm)
    valid_rr = rr[(rr > 300) & (rr < 1500)]
    
    if len(valid_rr) < 2:
        if len(rr) >= 1: valid_rr = rr
        else: return {'hr': np.nan, 'rmssd': np.nan, 'sdnn': np.nan}

    hr = 60000 / np.mean(valid_rr)
    rmssd = np.sqrt(np.mean(np.diff(valid_rr)**2)) if len(valid_rr) > 1 else np.nan
    sdnn = np.std(valid_rr) if len(valid_rr) > 1 else 0
    
    return {'hr': hr, 'rmssd': rmssd, 'sdnn': sdnn}

# 분석 실행 (메인 루프 수정)
results = []
print("Subject | Phase | Duration | HR | RMSSD | EMG RMS | EDA SCL")

for subj in SUBJECTS:
    sens, evts = load_subject(subj)
    if sens is None:
        print(f"Skipping {subj}: Sensor/Events not found")
        continue

    # ... cols ...
    name, code = SUBJECT_MAP[subj]
    emg_col = [c for c in sens.columns if 'EMG' in c.upper()][0]
    eda_col = [c for c in sens.columns if 'EDA' in c.upper()][0]
    ecg_col = [c for c in sens.columns if 'ECG' in c.upper()][0]
    
    for phase in ['BASELINE', 'TASK', 'PUZZLE', 'CHASE']:
        pdata = get_phase(sens, evts, phase, name)
        if pdata is None:
            # CHASE 없는 경우 등
            results.append({
                'Subject': name, 'Code': code, 'Phase': phase,
                'Duration': 0,
                'EMG_mean': np.nan, 'EMG_rms': np.nan, 'EMG_max': np.nan,
                'EDA_scl': np.nan,
                'ECG_hr': np.nan, 'ECG_rmssd': np.nan, 'ECG_sdnn': np.nan
            })
            continue
        
        duration = len(pdata)/1000
        emg = analyze_emg(pdata[emg_col].values)
        eda = analyze_eda(pdata[eda_col].values)
        ecg = analyze_ecg(pdata[ecg_col].values)
        
        print(f"{code} ({subj}) | {phase} | {duration:.1f}s | {ecg['hr']:.1f} | {ecg['rmssd']:.1f} | {emg['rms']:.2f} | {eda['scl_mean']:.2f}")
        
        results.append({
            'Subject': name, 'Code': code, 'Phase': phase,
            'Duration': duration,
            'EMG_mean': emg['mean'], 'EMG_rms': emg['rms'], 'EMG_max': emg['max'],
            'EDA_scl': eda['scl_mean'],
            'ECG_hr': ecg['hr'], 'ECG_rmssd': ecg['rmssd'], 'ECG_sdnn': ecg['sdnn']
        })

df = pd.DataFrame(results)
df.to_csv(os.path.join(BASE_DIR, 'deepresearch_revision', 'calculated_from_raw.csv'), index=False)

# 통계 출력
print('=== Phase별 통계 (원본 데이터 기반) ===\n')

phase_names = {'BASELINE': 'Phase 0', 'TASK': 'Phase 1', 'PUZZLE': 'Phase 2', 'CHASE': 'Phase 3'}

print('## EMG (Mean ± SD)')
for phase in ['BASELINE', 'TASK', 'PUZZLE', 'CHASE']:
    p = df[df['Phase'] == phase]
    m, r, mx = p['EMG_mean'], p['EMG_rms'], p['EMG_max']
    print(f"{phase_names[phase]}: Mean={m.mean():.2f}±{m.std():.2f} | RMS={r.mean():.2f}±{r.std():.2f} | Max={mx.mean():.2f}±{mx.std():.2f}")

print('\n## ECG/HRV (Mean ± SD)')
for phase in ['BASELINE', 'TASK', 'PUZZLE', 'CHASE']:
    p = df[df['Phase'] == phase]
    hr, rmssd, sdnn = p['ECG_hr'].dropna(), p['ECG_rmssd'].dropna(), p['ECG_sdnn'].dropna()
    print(f"{phase_names[phase]} (N={len(hr)}): HR={hr.mean():.2f}±{hr.std():.2f} | RMSSD={rmssd.mean():.2f}±{rmssd.std():.2f} | SDNN={sdnn.mean():.2f}±{sdnn.std():.2f}")

print('\n## EDA SCL (Mean ± SD)')
for phase in ['BASELINE', 'TASK', 'PUZZLE', 'CHASE']:
    p = df[df['Phase'] == phase]
    scl = p['EDA_scl']
    print(f"{phase_names[phase]}: SCL={scl.mean():.2f}±{scl.std():.2f} μS")

# 정규화 변화율
print('\n## 베이스라인 정규화 변화율 (%)')
codes = df['Code'].unique()
norm = {p: {'hr': [], 'rmssd': [], 'sdnn': []} for p in ['TASK', 'PUZZLE', 'CHASE']}

for code in codes:
    cdata = df[df['Code'] == code]
    base = cdata[cdata['Phase'] == 'BASELINE']
    if len(base) == 0:
        continue
    
    hr0 = base['ECG_hr'].values[0]
    rmssd0 = base['ECG_rmssd'].values[0]
    sdnn0 = base['ECG_sdnn'].values[0]
    
    for phase in ['TASK', 'PUZZLE', 'CHASE']:
        phs = cdata[cdata['Phase'] == phase]
        if len(phs) == 0:
            continue
        
        hr, rmssd, sdnn = phs['ECG_hr'].values[0], phs['ECG_rmssd'].values[0], phs['ECG_sdnn'].values[0]
        
        if pd.notna(hr) and pd.notna(hr0) and hr0 > 0:
            norm[phase]['hr'].append(((hr - hr0) / hr0) * 100)
        if pd.notna(rmssd) and pd.notna(rmssd0) and rmssd0 > 0:
            norm[phase]['rmssd'].append(((rmssd - rmssd0) / rmssd0) * 100)
        if pd.notna(sdnn) and pd.notna(sdnn0) and sdnn0 > 0:
            norm[phase]['sdnn'].append(((sdnn - sdnn0) / sdnn0) * 100)

for phase in ['TASK', 'PUZZLE', 'CHASE']:
    hr = np.array(norm[phase]['hr'])
    rmssd = np.array(norm[phase]['rmssd'])
    sdnn = np.array(norm[phase]['sdnn'])
    print(f"{phase_names[phase]}: HR={np.mean(hr):+.1f}±{np.std(hr):.1f} | RMSSD={np.mean(rmssd):+.1f}±{np.std(rmssd):.1f} | SDNN={np.mean(sdnn):+.1f}±{np.std(sdnn):.1f}")
