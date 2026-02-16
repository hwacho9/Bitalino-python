import pandas as pd
import numpy as np

# Load data
file_path = '/Users/chosunghwa/Desktop/workspace/Bitalino-python/analysis_output/deep_research_features.csv'
df = pd.read_csv(file_path)

# Phase mapping
phase_map = {
    'BASELINE': 'Phase 0',
    'TASK': 'Phase 1',
    'PUZZLE': 'Phase 2',
    'CHASE': 'Phase 3'
}

# Ensure order
phase_order = ['BASELINE', 'TASK', 'PUZZLE', 'CHASE']

def format_mean_sd(mean, sd):
    return f"{mean:.1f} $\pm$ {sd:.1f}"

def format_val(val):
    return f"{val:.2f}"

print("--- Table: EDA SCL Stats ---")
# Table: tab:eda_scl_stats
# Columns: Phase, Mean SCL, SD, Min, Max
eda_stats = []
for phase in phase_order:
    phase_data = df[df['phase'] == phase]['EDA_mean_SCL_uS']
    mean_val = phase_data.mean()
    sd_val = phase_data.std()
    min_val = phase_data.min() # Min of the subject means? Or min of raw data? Paper says "Min value" usually means min of the calculated features or min of the group. Given the previous table values (Min 9.05, Max 28.02 for Phase 0), let's check the data.
    # In deep_research_features.csv, we have one row per subject per phase. 
    # The stats in the table seem to be aggregated across subjects. 
    # "Min" likely means the minimum subject mean SCL observed in that phase.
    max_val = phase_data.max()
    
    eda_stats.append({
        'Phase': phase_map[phase],
        'Mean': mean_val,
        'SD': sd_val,
        'Min': min_val,
        'Max': max_val
    })

print(f"Phase & Mean SCL ($\mu$S) & SD & Min & Max \\\\")
print("\\hline")
for row in eda_stats:
    print(f"{row['Phase']} & {row['Mean']:.2f} & {row['SD']:.2f} & {row['Min']:.2f} & {row['Max']:.2f} \\\\")
print("\\hline")
print("\n")


print("--- Table: ECG/HRV Stats ---")
# Table: tab:ecg_hrv_stats
# Features: HR (bpm), RMSSD (ms), SDNN (ms), LF/HF
metrics = {
    '平均心拍数 (HR, bpm)': 'ECG_heart_rate_bpm',
    'RMSSD (ms)': 'ECG_hrv_rmssd_ms', 
    'SDNN (ms)': 'ECG_hrv_sdnn_ms',
    'LF/HF比': 'ECG_lf_hf_ratio'
}

print("Feature & Phase 0 & Phase 1 & Phase 2 & Phase 3 \\\\")
print("\\hline")
for label, col in metrics.items():
    row_str = f"{label}"
    for phase in phase_order:
        phase_data = df[df['phase'] == phase][col]
        # Check for inf/nan
        phase_data = phase_data.replace([np.inf, -np.inf], np.nan).dropna()
        if len(phase_data) == 0:
            row_str += " & N/A"
            continue
            
        mean_val = phase_data.mean()
        sd_val = phase_data.std()
        row_str += f" & {format_mean_sd(mean_val, sd_val)}"
    print(row_str + " \\\\")
print("\\hline")
print("\n")


print("--- Table: Normalized ECG/HRV Stats ---")
# Table: tab:ecg_hrv_normalized
# Method: (Mean_Phase - Mean_Baseline) / Mean_Baseline * 100
# SD: abs(100 / Mean_Baseline) * SD_Phase (Error Propagation)

norm_metrics = {
    '心拍数変化率': 'ECG_heart_rate_bpm',
    'RMSSD変化率': 'ECG_hrv_rmssd_ms',
    'SDNN変化率': 'ECG_hrv_sdnn_ms',
    'LF/HF変化率': 'ECG_lf_hf_ratio'
}

print("Feature & Phase 1 & Phase 2 & Phase 3 \\\\")
print("\\hline")
for label, col in norm_metrics.items():
    row_str = f"{label}"
    
    # Get Baseline Stats
    baseline_data = df[df['phase'] == 'BASELINE'][col]
    baseline_mean = baseline_data.mean()
    
    for phase in ['TASK', 'PUZZLE', 'CHASE']:
        phase_data = df[df['phase'] == phase][col]
        # Handle outliers or missing data? The original script just used mean/std of the column
        phase_mean = phase_data.mean()
        phase_std = phase_data.std()
        
        if baseline_mean != 0:
            change = ((phase_mean - baseline_mean) / baseline_mean) * 100
            # Error propagation for SD
            sd_val = abs(100 / baseline_mean) * phase_std
        else:
            change = np.nan
            sd_val = np.nan
            
         # Format with + sign if positive
        mean_str = f"{change:+.1f}" if change >= 0 else f"{change:.1f}"
        row_str += f" & {mean_str} $\pm$ {sd_val:.1f}"
    print(row_str + " \\\\")
print("\\hline")
print("\n")


print("--- Table: EMG Stats ---")
# Table: tab:emg_stats
# Features: Mean, RMS, Max
emg_metrics = {
    '平均電圧 (Mean, $\mu$V)': 'EMG_mean_uV',
    'RMS電圧 (RMS, $\mu$V)': 'EMG_rms_uV',
    '最大電圧 (Max, $\mu$V)': 'EMG_max_uV'
}

print("Feature & Phase 0 & Phase 1 & Phase 2 & Phase 3 \\\\")
print("\\hline")
for label, col in emg_metrics.items():
    row_str = f"{label}"
    for phase in phase_order:
        phase_data = df[df['phase'] == phase][col]
        mean_val = phase_data.mean()
        sd_val = phase_data.std()
        row_str += f" & {format_mean_sd(mean_val, sd_val)}"
    print(row_str + " \\\\")
print("\\hline")
