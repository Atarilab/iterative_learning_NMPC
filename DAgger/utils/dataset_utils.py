import numpy as np
from typing import List

def merge_datasets(file_list: List[str], save_path: str):
    """Load all .npz files, concatenate along time axis, and save into one."""
    all_data = {}
    
    for file_path in file_list:
        data = np.load(file_path)
        for key in data.files:
            if key not in all_data:
                all_data[key] = []
            all_data[key].append(data[key])

    # Concatenate along the first axis (time dimension)
    merged_data = {k: np.concatenate(v, axis=0) for k, v in all_data.items()}

    # Save the merged dataset
    np.savez(save_path, **merged_data)
    print(f"✅ Merged dataset saved to: {save_path}")
