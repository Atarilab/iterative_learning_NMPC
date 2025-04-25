import numpy as np

# === Define your npz file path and index here ===
npz_path = "/home/atari/workspace/iterative_supervised_learning/examples/data/behavior_cloning/trot/Apr_23_2025_08_57_33/dataset/experiment/traj_nominal_04_23_2025_08_57_39.npz"
n = 1944  # which entry you want to check

# === Load the npz file ===
data = np.load(npz_path)

# === Check if 'ctrl' key exists ===
if "ctrl" not in data:
    raise KeyError("Key 'ctrl' not found in the provided .npz file.")

ctrl_data = data["ctrl"]

# === Check if index is valid ===
if n < 0 or n >= len(ctrl_data):
    raise IndexError(f"Index {n} is out of bounds. 'ctrl' array has {len(ctrl_data)} entries.")

# === Print the nth 'ctrl' entry ===
print(f"ctrl[{n}] =\n{ctrl_data[n]}")
