import os
import glob
import numpy as np
import h5py
from scipy.signal import savgol_filter
from tqdm import tqdm

# ============================== CONFIGURATION ==============================
NPZ_DIR = '/home/atari/workspace/iterative_supervised_learning/examples/data/behavior_cloning/trot/Apr_01_2025_17_30_52/dataset/experiment'                      # Folder containing your .npz files
HDF5_OUTPUT_PATH = '/home/atari/workspace/iterative_supervised_learning/examples/data/behavior_cloning/trot/Apr_01_2025_17_30_52/dataset/smoothed_dataset.hdf5'     # Output path for merged and smoothed dataset
WINDOW = 5                              # Savitzky-Golay window size (must be odd)
POLYORDER = 2                           # Polynomial order for smoothing
SMOOTH_KEYS = ['state', 'action']      # Keys to smooth (if present in npz)
# ===========================================================================

def smooth_array(arr, window=5, polyorder=2):
    if arr.shape[0] < window:
        return arr  # Too short to filter
    return savgol_filter(arr, window_length=window, polyorder=polyorder, axis=0)

def main():
    npz_files = sorted(glob.glob(os.path.join(NPZ_DIR, '*.npz')))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {NPZ_DIR}")
    
    print(f"Found {len(npz_files)} .npz files. Processing...")

    # Placeholder to accumulate all data
    merged_data = {key: [] for key in SMOOTH_KEYS}

    for path in tqdm(npz_files):
        try:
            data = np.load(path)
        except Exception as e:
            print(f"Failed to read {path}: {e}")
            continue

        for key in SMOOTH_KEYS:
            if key not in data:
                print(f"Warning: {key} not in {path}")
                continue

            arr = data[key]
            arr_smoothed = smooth_array(arr, window=WINDOW, polyorder=POLYORDER)
            merged_data[key].append(arr_smoothed)

    # Concatenate all samples across files
    for key in merged_data:
        merged_data[key] = np.concatenate(merged_data[key], axis=0)

    # Save to HDF5
    with h5py.File(HDF5_OUTPUT_PATH, 'w') as f:
        for key, arr in merged_data.items():
            f.create_dataset(key, data=arr, compression='gzip')
    
    print(f"\n✅ Done. Smoothed data saved to: {HDF5_OUTPUT_PATH}")
    for key in merged_data:
        print(f" - {key}: {merged_data[key].shape}")

if __name__ == '__main__':
    main()
