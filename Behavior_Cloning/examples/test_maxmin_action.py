import numpy as np
import os

def extract_action_bounds(npz_path):
    # Load the trajectory data
    data = np.load(npz_path)
    
    if "action" not in data:
        raise KeyError(f"'action' key not found in file: {npz_path}")
    
    actions = data["action"]  # shape: (T, 12) where T = number of time steps

    # Compute min and max over the dataset
    min_action = np.min(actions, axis=0)  # shape: (12,)
    max_action = np.max(actions, axis=0)  # shape: (12,)

    print("Min action per joint:")
    print(min_action)
    print("\nMax action per joint:")
    print(max_action)
    
    return min_action, max_action

# Example usage
if __name__ == "__main__":
    npz_file_path = "/home/atari/workspace/Behavior_Cloning/examples/data/traj_nominal_04_07_2025_14_08_03.npz"  # ← replace with your actual file
    if not os.path.exists(npz_file_path):
        raise FileNotFoundError(f"Could not find file at: {npz_file_path}")
    extract_action_bounds(npz_file_path)
