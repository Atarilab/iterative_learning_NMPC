import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# List of dataset directories to compare
data_dirs = [
    "/home/atari/workspace/Behavior_Cloning/examples/data/behavior_cloning/trot/Mar_11_2025_15_59_31/dataset/experiment",
    "/home/atari/workspace/Behavior_Cloning/examples/data/behavior_cloning/trot/Mar_13_2025_10_25_00/dataset/experiment",
    "/home/atari/workspace/Behavior_Cloning/examples/data/behavior_cloning/trot/Mar_16_2025_15_56_16/dataset/experiment"
]
k = None  # If None, use all trajectories

# Joint labels for visualization
joint_labels = [
    "FL_hip", "FL_thigh", "FL_calf",
    "FR_hip", "FR_thigh", "FR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
    "RR_hip", "RR_thigh", "RR_calf"
]

# Initialize dictionaries to store data for all datasets
dataset_position_dict = {}
dataset_velocity_dict = {}
dataset_PD_target_dict = {}

for dir_idx, data_dir in enumerate(data_dirs):
    # Read all `.npz` files in the directory
    trajectory_files = sorted(
        [f for f in os.listdir(data_dir) if f.endswith(".npz")],
        key=lambda f: os.path.getmtime(os.path.join(data_dir, f))
    )[:k]  # Use first k files if k is specified

    # Initialize lists for storing combined data
    combined_position = []
    combined_velocity = []
    combined_PD_target = []

    for traj_file in trajectory_files:
        data_path = os.path.join(data_dir, traj_file)
        data = np.load(data_path)

        # Extract joint positions, velocities, and PD targets
        joint_pos_his = data["q"][:, 7:]  # 12 joint positions
        joint_vel_his = data["v"][:, 6:]  # 12 joint velocities
        PD_target_his = data["action"]    # PD target values

        combined_position.append(joint_pos_his)
        combined_velocity.append(joint_vel_his)
        combined_PD_target.append(PD_target_his)

    # Convert lists to numpy arrays
    combined_position = np.vstack(combined_position) if combined_position else None
    combined_velocity = np.vstack(combined_velocity) if combined_velocity else None
    combined_PD_target = np.vstack(combined_PD_target) if combined_PD_target else None

    # Get dataset size (number of samples)
    dataset_size = combined_position.shape[0] if combined_position is not None else 0

    # Store combined data with dataset size in the label
    dataset_label = f"Dataset {dir_idx + 1} ({dataset_size} samples)"
    dataset_position_dict[dataset_label] = combined_position
    dataset_velocity_dict[dataset_label] = combined_velocity
    dataset_PD_target_dict[dataset_label] = combined_PD_target


# Function to plot KDE for multiple datasets
def plot_kde_multiple_datasets(dataset_dict, title, ylabel):
    """
    Plots KDE for multiple datasets on the same plot.

    Parameters:
        dataset_dict (dict): Dictionary where keys are dataset names (with sizes), and values are numpy arrays.
        title (str): The title for the plot.
        ylabel (str): Label for the y-axis.
    """
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))  # 4 rows, 3 columns for 12 joints

    for i, ax in enumerate(axes.flat):
        # Plot KDE for each dataset
        for dataset_name, dataset_data in dataset_dict.items():
            if dataset_data is not None:
                sns.kdeplot(dataset_data[:, i], label=dataset_name, ax=ax, fill=False, alpha=0.6)

        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} - {joint_labels[i]}")
        ax.grid()

        # Add legend inside the plot
        ax.legend(fontsize="small", loc="upper right", frameon=True)

    plt.tight_layout()
    plt.show()


# Plot KDE for Joint Position Distribution across datasets
plot_kde_multiple_datasets(dataset_position_dict, "Joint Position Distribution (KDE) Across Datasets", "Joint Position")

# Plot KDE for Joint Velocity Distribution across datasets
plot_kde_multiple_datasets(dataset_velocity_dict, "Joint Velocity Distribution (KDE) Across Datasets", "Joint Velocity")

# Plot KDE for PD Target Distribution across datasets
plot_kde_multiple_datasets(dataset_PD_target_dict, "Joint PD Target Distribution (KDE) Across Datasets", "PD Target")
