import numpy as np
from scipy.signal import butter, filtfilt
import os

# ========================= CONFIGURATION =========================
NPZ_PATH = "/home/atari/workspace/Behavior_Cloning/examples/data/behavior_cloning/trot/Apr_15_2025_10_21_38/dataset/experiment/traj_nominal_04_15_2025_10_21_45.npz"    # Input .npz file
OUTPUT_PATH = './data/example_traj_smoothed.npz'  # Output file
CUTOFF_HZ = 5.0    # Cutoff frequency (Hz)
FS = 100.0          # Sampling rate (Hz) — adjust to your data
ORDER = 2           # Filter orders
TARGET_KEYS = ['state', 'action']
# ================================================================
def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = 0.5 * fs
    norm_cutoff = cutoff / nyq
    b, a = butter(order, norm_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)

def main():
    data = np.load(NPZ_PATH)
    new_data = {}

    for key in data.files:
        arr = data[key]
        if key in TARGET_KEYS:
            if arr.shape[0] < (ORDER + 1) * 2:
                print(f"⚠️  {key} too short to filter; copying raw.")
                new_data[key] = arr
                continue
            print(f"Applying low-pass filter to: {key}, shape={arr.shape}")
            arr_filtered = butter_lowpass_filter(arr, CUTOFF_HZ, FS, ORDER)
            new_data[key] = arr_filtered
        else:
            new_data[key] = arr  # Copy other fields unchanged

    np.savez(OUTPUT_PATH, **new_data)
    print(f"\n✅ Low-pass filtered data saved to: {OUTPUT_PATH}")

if __name__ == '__main__':
    main()