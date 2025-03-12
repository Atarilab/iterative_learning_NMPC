import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set directory path and number of trajectories to visualize
data_dir = "/home/atari/workspace/iterative_supervised_learning/examples/data/behavior_cloning/trot/Mar_11_2025_15_59_31/dataset/experiment"
k = 6  # Number of trajectories to visualize

# Joint labels for visualization
joint_labels = [
    "FL_hip", "FL_thigh", "FL_calf",
    "FR_hip", "FR_thigh", "FR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
    "RR_hip", "RR_thigh", "RR_calf"
]

# Read first k trajectories (sorted by modification time)
trajectory_files = sorted(
    [f for f in os.listdir(data_dir) if f.endswith(".npz")],
    key=lambda f: os.path.getmtime(os.path.join(data_dir, f))
)[:k]

# Initialize dictionaries for storing trajectory-wise data
position_dict = {}
velocity_dict = {}
PD_target_dict = {}

# Initialize combined data arrays
combined_position = []
combined_velocity = []
combined_PD_target = []

for idx, traj_file in enumerate(trajectory_files):
    data_path = os.path.join(data_dir, traj_file)
    data = np.load(data_path)

    # Extract joint positions, velocities, and PD target values
    joint_pos_his = data["q"][:, 7:]  # 12 joint positions
    joint_vel_his = data["v"][:, 6:]  # 12 joint velocities
    PD_target_his = data["action"]    # PD target values

    # Store trajectory-specific data
    traj_label = f"traj_{idx + 1}"
    position_dict[traj_label] = joint_pos_his
    velocity_dict[traj_label] = joint_vel_his
    PD_target_dict[traj_label] = PD_target_his

    # Append to combined dataset
    combined_position.append(joint_pos_his)
    combined_velocity.append(joint_vel_his)
    combined_PD_target.append(PD_target_his)

# Convert combined lists into numpy arrays
combined_position = np.vstack(combined_position) if combined_position else None
combined_velocity = np.vstack(combined_velocity) if combined_velocity else None
combined_PD_target = np.vstack(combined_PD_target) if combined_PD_target else None


# Function to plot KDE distribution
def plot_kde_distribution(data_dict, title, ylabel, combined_data=None):
    """
    Plots KDE of the state values to compare distributions across different trajectories.
    Also includes KDE for the combined dataset if provided.

    Parameters:
        data_dict (dict): Dictionary where keys are trajectory names (e.g., "traj_1"), 
                          and values are numpy arrays of shape (timesteps, joints).
        title (str): The title for the plot.
        ylabel (str): Label for the y-axis.
        combined_data (numpy array): Combined data across all trajectories (timesteps, joints).
    """
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))  # 4 rows, 3 columns for 12 joints

    for i, ax in enumerate(axes.flat):
        # Plot individual trajectories
        for traj_name, traj_data in data_dict.items():
            sns.kdeplot(traj_data[:, i], label=traj_name, ax=ax, fill=False, alpha=0.5)

        # Plot combined trajectory KDE
        if combined_data is not None:
            sns.kdeplot(combined_data[:, i], label="Combined", ax=ax, fill=True, color="black", alpha=0.3)

        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} - {joint_labels[i]}")
        ax.legend()
        ax.grid()

    plt.tight_layout()
    plt.show()


# Plot KDE for joint PD target distribution
plot_kde_distribution(PD_target_dict, "Joint PD target Distribution (KDE)", "Joint PD target", combined_PD_target)

# Plot KDE for joint position distribution
plot_kde_distribution(position_dict, "Joint Position Distribution (KDE)", "Joint Position", combined_position)

# Plot KDE for joint velocity distribution
plot_kde_distribution(velocity_dict, "Joint Velocity Distribution (KDE)", "Joint Velocity", combined_velocity)
