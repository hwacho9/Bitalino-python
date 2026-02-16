import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def generate_continuous_plot(subject_name):
    base_dir = "analysis_output"
    phases = ["BASELINE", "TASK", "PUZZLE", "CHASE"]
    phase_colors = {
        'BASELINE': '#2E86AB',
        'TASK': '#F18F01',
        'PUZZLE': '#C73E1D',
        'CHASE': '#A23B72'
    }

    all_data = []
    current_time = 0
    phase_boundaries = []

    print(f"Loading data for {subject_name}...")

    for phase in phases:
        file_path = os.path.join(base_dir, f"{subject_name}_{phase}_preprocessed.csv")
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue

        df = pd.read_csv(file_path)
        
        # Adjust time to be continuous
        start_time = current_time
        df['continuous_time'] = df['time_sec'] + start_time
        
        # Update current_time for next phase
        if not df.empty:
            current_time = df['continuous_time'].iloc[-1]
            phase_boundaries.append((phase, start_time, current_time))
        
        df['phase'] = phase
        all_data.append(df)

    if not all_data:
        print("No data found.")
        return

    full_df = pd.concat(all_data, ignore_index=True)

    # Plotting
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    
    # 1. EMG
    ax = axes[0]
    ax.plot(full_df['continuous_time'], full_df['EMG_processed'], color='black', linewidth=0.3, alpha=0.8)
    for phase, start, end in phase_boundaries:
        ax.axvspan(start, end, color=phase_colors[phase], alpha=0.1)
        ax.text((start+end)/2, ax.get_ylim()[1], phase, ha='center', va='bottom', fontweight='bold', color=phase_colors[phase])
    ax.set_ylabel('EMG (uV)')
    ax.set_title(f'Continuous Sensor Data - {subject_name}')
    ax.grid(True, alpha=0.3)

    # 2. EDA
    ax = axes[1]
    ax.plot(full_df['continuous_time'], full_df['EDA_processed'], color='black', linewidth=0.8)
    for phase, start, end in phase_boundaries:
        ax.axvspan(start, end, color=phase_colors[phase], alpha=0.1)
    ax.set_ylabel('EDA (uS)')
    ax.grid(True, alpha=0.3)

    # 3. ECG
    ax = axes[2]
    ax.plot(full_df['continuous_time'], full_df['ECG_processed'], color='black', linewidth=0.3, alpha=0.8)
    for phase, start, end in phase_boundaries:
        ax.axvspan(start, end, color=phase_colors[phase], alpha=0.1)
    ax.set_ylabel('ECG (mV)')
    ax.set_xlabel('Time (s)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    
    output_dir = os.path.join(base_dir, "summary")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"subject_continuous_data_{subject_name}.png")
    
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    generate_continuous_plot("takamiya2")
