import h5py

# Replace with your actual path
file_path = "/home/atari/workspace/iterative_supervised_learning/examples/data/behavior_cloning/trot/Apr_09_2025_14_29_06/dataset/database_0.hdf5"

with h5py.File(file_path, 'r') as f:
    print("Top-level keys in the HDF5 file:")
    for key in f.keys():
        print(" -", key)

    # Optionally check shapes
    print("\nDataset shapes:")
    for key in f.keys():
        print(f"{key}: {f[key].shape}")
