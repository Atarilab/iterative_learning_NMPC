import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../..')
from iterative_supervised_learning.utils.database import Database

# --- Configuration ---
# dataset_path = "/home/atari/workspace/iterative_supervised_learning/examples/data/behavior_cloning/trot/Apr_16_2025_15_57_03/dataset/database_0.hdf5" 
dataset_path = "/home/atari/workspace/iterative_supervised_learning/examples/data/behavior_cloning/trot/Apr_22_2025_14_22_27/dataset/database_0.hdf5"
use_state_mask = True
distance_plot_save_path = "l2_distances_hist.png"

# --- Optional: mask certain dimensions of state (e.g., exclude phase %)
def get_state_mask(state_dim):
    mask = np.ones(state_dim, dtype=bool)
    mask[0] = False  # phase percentage (exclude)
    return mask

# --- Load database ---
db = Database(limit=1_000_000, norm_input=False)
db.load_saved_database(dataset_path)

# --- Preprocess ---
nominal_states = {}
for i in range(len(db)):
    if db.traj_ids[i] == 0:
        t = db.traj_times[i]
        nominal_states[t] = db.states[i]

# --- Compute distances ---
state_array = np.array(db.states[:db.length])
state_dim = state_array.shape[1]
mask = get_state_mask(state_dim) if use_state_mask else slice(None)

distances = []
for i in range(db.length):
    if db.traj_ids[i] == 1:
        t = db.traj_times[i]
        if t in nominal_states:
            s_pert = np.array(db.states[i])[mask]
            s_nom = np.array(nominal_states[t])[mask]
            dist = np.linalg.norm(s_nom - s_pert)
            distances.append(dist)

distances = np.array(distances)
print(f"Computed {len(distances)} perturbed vs nominal distances")
print(f"Mean: {np.mean(distances):.4f}, Std: {np.std(distances):.4f}")
print(f"90th percentile: {np.percentile(distances, 90):.4f}")
print(f"Max: {np.max(distances):.4f}")

# --- Plot ---
plt.figure(figsize=(8, 4))
plt.hist(distances, bins=50, alpha=0.8, color='steelblue', edgecolor='black')
plt.title("L2 Distance: Perturbed vs Nominal States")
plt.xlabel("L2 Distance")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.savefig(distance_plot_save_path)
plt.show()
