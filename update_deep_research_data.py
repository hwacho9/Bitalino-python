import pandas as pd
import numpy as np

# Load source data
source_file = '/Users/chosunghwa/Desktop/workspace/Bitalino-python/analysis_output/deep_research_features.csv'
df_source = pd.read_csv(source_file)

# Define mappings
subject_map = {
    'masaya': {'Name': 'Noguchi', 'Code': 'A'},
    'hase1': {'Name': 'Hase', 'Code': 'B1'},
    'hase2': {'Name': 'Hase', 'Code': 'B2'},
    'matsumoto': {'Name': 'Matsumoto', 'Code': 'C'},
    'ishikawa': {'Name': 'Ishikawa', 'Code': 'D'},
    'takamiya': {'Name': 'Takamiya', 'Code': 'E1'},
    'takamiya2': {'Name': 'Takamiya', 'Code': 'E2'},
    'sensei': {'Name': 'Sensei', 'Code': 'F'}
}

# Define column mapping (Target -> Source)
col_map = {
    'Duration_sec': 'duration_sec',
    'EMG_mean': 'EMG_mean_uV',
    'EMG_rms': 'EMG_rms_uV',
    'EMG_max': 'EMG_max_uV',
    'EDA_SCL': 'EDA_mean_SCL_uS',
    'ECG_HR_bpm': 'ECG_heart_rate_bpm',
    'ECG_RMSSD_ms': 'ECG_hrv_rmssd_ms',
    'ECG_SDNN_ms': 'ECG_hrv_sdnn_ms'
}

# Prepare target rows
rows = []
phase_order = ['BASELINE', 'TASK', 'PUZZLE', 'CHASE']

for subject_id, info in subject_map.items():
    subject_data = df_source[df_source['subject_id'] == subject_id]
    
    for phase in phase_order:
        phase_row = subject_data[subject_data['phase'] == phase]
        
        row_data = {
            'Subject': info['Name'],
            'Code': info['Code'],
            'Session': subject_id,
            'Phase': phase,
            'Notes': 'Verified'
        }
        
        if not phase_row.empty:
            for target_col, source_col in col_map.items():
                val = phase_row[source_col].values[0]
                
                # DATA OVERRIDE: Takamiya2 (E2) CHASE - User provided corrected values
                if subject_id == 'takamiya2' and phase == 'CHASE':
                    if 'ECG_HR_bpm' in target_col:
                        row_data[target_col] = 100.40
                    elif 'ECG_RMSSD_ms' in target_col:
                        row_data[target_col] = 154.40
                    elif 'EMG_rms' in target_col:
                        row_data[target_col] = 57.01
                    elif 'EMG_mean' in target_col:
                        row_data[target_col] = '' # Clear others if not provided
                    elif 'EMG_max' in target_col:
                        row_data[target_col] = ''
                    elif 'ECG_SDNN' in target_col:
                        row_data[target_col] = ''
                    row_data['Notes'] = 'User Corrected'
                elif isinstance(val, (int, float)):
                    row_data[target_col] = round(val, 2)
                else:
                    row_data[target_col] = val
        else:
            # Handle missing phase (e.g. Noguchi CHASE)
             for target_col in col_map.keys():
                row_data[target_col] = ''
             row_data['Notes'] = 'No Data'
             if phase == 'CHASE' and subject_id == 'masaya':
                 row_data['Duration_sec'] = 0.0

        rows.append(row_data)

# Create DataFrame
df_target = pd.DataFrame(rows, columns=[
    'Subject', 'Code', 'Session', 'Phase', 'Duration_sec', 
    'EMG_mean', 'EMG_rms', 'EMG_max', 'EDA_SCL', 
    'ECG_HR_bpm', 'ECG_RMSSD_ms', 'ECG_SDNN_ms', 'Notes'
])

# Save to CSV
output_csv = '/Users/chosunghwa/Desktop/workspace/Bitalino-python/deepresearch_revision/all_subjects_complete_data.csv'
df_target.to_csv(output_csv, index=False)
print(f"Updated {output_csv}")
