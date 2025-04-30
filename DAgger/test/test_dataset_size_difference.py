import h5py
import os

# ======= Define your paths here =======
original_db_path = "/home/atari/workspace/DAgger/example/data/SafeDagger/database_0.hdf5"
aggregated_db_path = "/home/atari/workspace/DAgger/example/data/SafeDagger/trot/Apr_29_2025_19_43_55/dataset/agg_dataset1.hdf5"

def get_dataset_length(h5_path):
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"File not found: {h5_path}")
    
    with h5py.File(h5_path, 'r') as f:
        if 'states' in f:
            return len(f['states'])
        else:
            raise KeyError(f"No 'states' dataset found in {h5_path}")

def main():
    print("Comparing dataset lengths...\n")
    
    len_orig = get_dataset_length(original_db_path)
    len_agg = get_dataset_length(aggregated_db_path)

    print(f"Original Database ({original_db_path}): {len_orig} samples")
    print(f"Aggregated Database ({aggregated_db_path}): {len_agg} samples")

    abs_diff = abs(len_agg - len_orig)
    rel_diff = (abs_diff / len_orig) * 100 if len_orig != 0 else float('inf')

    print(f"\nSize Difference: {abs_diff} samples")
    print(f"Relative Difference: {rel_diff:.2f}%")

    if len_orig > len_agg:
        print("→ Aggregated database is smaller than original.")
    elif len_agg > len_orig:
        print("→ Aggregated database is larger than original.")
    else:
        print("→ Both databases have the same length.")

if __name__ == "__main__":
    main()
