import csv
import sys
import matplotlib.pyplot as plt
import numpy as np

def visualize(csv_file):
    print(f"Loading {csv_file}...")
    
    times = []
    sample_indices = []
    analog_data = [[], [], [], [], [], []] # A1 to A6
    labels = ["A1 (EMG)", "A2 (EDA)", "A3 (ECG)", "A4", "A5", "A6"]
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader) # skip header
            
            # Header: unix_time(0), readable_time(1), sample_index(2), sequence(3), D0(4)... A1(8)...
            
            for row in reader:
                if not row: continue
                # Parse available columns
                times.append(float(row[0])) 
                # row[1] is readable_time string, skip for plotting
                sample_indices.append(int(row[2]))
                
                # A1 starts from index 8 (0-based)
                # D0=4, D1=5, D2=6, D3=7, A1=8...
                vals = row[8:]
                for i in range(len(vals)):
                    if i < 6:
                        analog_data[i].append(int(vals[i]))
                        
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # Convert to numpy for easier plotting
    times = np.array(times)
    # Start time from 0
    rel_times = times - times[0]
    
    # Plotting
    fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    if not isinstance(ax, list) and not isinstance(ax, np.ndarray):
        ax = [ax] 
        
    # Only A1, A2, A3
    channels_to_plot = [0, 1, 2] # indices in analog_data
    
    for i, ch_idx in enumerate(channels_to_plot):
        ax[i].plot(rel_times, analog_data[ch_idx], label=labels[ch_idx], color='tab:blue')
        ax[i].set_ylabel(labels[ch_idx])
        ax[i].grid(True)
        ax[i].legend(loc='upper right')
        
    ax[-1].set_xlabel('Time (seconds)')
    fig.suptitle(f'BITalino Sensor Data (A1-A3): {csv_file}')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_csv.py <csv_file>")
        # Default fallback for testing
        import glob
        files = glob.glob("sensor_data_*.csv")
        if files:
            visualize(files[-1]) # visualize last created file
        else:
            print("No CSV files found.")
    else:
        visualize(sys.argv[1])
