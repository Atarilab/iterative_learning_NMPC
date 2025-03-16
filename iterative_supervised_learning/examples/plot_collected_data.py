import os
import numpy as np
import matplotlib.pyplot as plt

# Function to load .npz files from a given directory
def load_data_from_directory(directory, k=5):
    """Loads the first k .npz files from the specified directory, sorted by creation time."""
    all_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".npz")]
    
    # Sort files by creation time
    all_files.sort(key=os.path.getctime)
    
    # Select the first k files
    selected_files = all_files[:k]
    
    return selected_files

# Define joint labels in the specified order
joint_labels = [
    "FL_hip", "FL_thigh", "FL_calf",
    "FR_hip", "FR_thigh", "FR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
    "RR_hip", "RR_thigh", "RR_calf"
]

# Global variables
visualize_length = 1500

# Directory containing data
directory_path = "/home/atari/workspace/iterative_supervised_learning/examples/data/behavior_cloning/trot/Mar_16_2025_15_56_16/dataset/experiment"

# Load the first k files from the directory
k = 1  # Number of files to visualize
data_paths = load_data_from_directory(directory_path, k)

# Initialize lists to store multiple trajectories
time_his_list = []
joint_pos_his_list = []
joint_vel_his_list = []
realized_PD_target_list = []
start_times = []  # Store start times for labels

# Load data from multiple files
for data_path in data_paths:
    data = np.load(data_path)
    
    time_his = data["time"][:visualize_length]
    time_his_list.append(time_his)
    joint_pos_his_list.append(data["q"][:, 7:][:visualize_length])
    joint_vel_his_list.append(data["v"][:, 6:][:visualize_length])
    realized_PD_target_list.append(data["action"][:visualize_length])

    # Store the first timestamp as the start time label
    start_times.append(f"Start Time: {time_his[0]:.2f}s")

def plot_joint_data(time_list, joint_data_list, title):
    """
    Function to plot multiple joint-related data over time in a 4x3 grid layout.
    
    Parameters:
        time_list (list of numpy arrays): List of time history arrays for multiple trajectories.
        joint_data_list (list of numpy arrays): List of joint data (12 joint values) for multiple trajectories.
        title (str): Title for the figure.
    """
    fig, axes = plt.subplots(4, 3, figsize=(12, 8), sharex=True)
    fig.suptitle(title, fontsize=14)

    for i, ax in enumerate(axes.flat):
        for j in range(len(joint_data_list)):  # Iterate over trajectories
            ax.plot(time_list[j], joint_data_list[j][:, i], label=start_times[j], alpha=0.7)
        ax.set_title(joint_labels[i])
        ax.set_ylabel("Value")
        ax.grid(True)
    
    plt.xlabel("Time (s)")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Plot multiple trajectories for joint position, velocity, and PD target
plot_joint_data(time_his_list, realized_PD_target_list, "Realized PD Target Over Time")
plot_joint_data(time_his_list, joint_pos_his_list, "Joint Positions Over Time")
plot_joint_data(time_his_list, joint_vel_his_list, "Joint Velocities Over Time")
