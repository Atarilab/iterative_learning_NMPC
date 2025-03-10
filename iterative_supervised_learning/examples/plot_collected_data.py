import numpy as np
import matplotlib.pyplot as plt

# Update this path to your correct file location
# Mar_07_2025_15_50_55
# data_paths = [
#     # benchmark traj
#     "/home/atari/workspace/iterative_supervised_learning/examples/data/behavior_cloning/trot/Mar_07_2025_15_50_55/dataset/experiment/simulation_data_03_07_2025_15_51_14.npz",
#     # starting from 0.05s
#     "/home/atari/workspace/iterative_supervised_learning/examples/data/behavior_cloning/trot/Mar_07_2025_15_50_55/dataset/experiment/simulation_data_03_07_2025_15_52_37.npz",
#     "/home/atari/workspace/iterative_supervised_learning/examples/data/behavior_cloning/trot/Mar_07_2025_15_50_55/dataset/experiment/simulation_data_03_07_2025_15_53_07.npz",
#     "/home/atari/workspace/iterative_supervised_learning/examples/data/behavior_cloning/trot/Mar_07_2025_15_50_55/dataset/experiment/simulation_data_03_07_2025_15_53_24.npz",
#     # starting from 0.1s
#     "/home/atari/workspace/iterative_supervised_learning/examples/data/behavior_cloning/trot/Mar_07_2025_15_50_55/dataset/experiment/simulation_data_03_07_2025_15_53_41.npz",
#     "/home/atari/workspace/iterative_supervised_learning/examples/data/behavior_cloning/trot/Mar_07_2025_15_50_55/dataset/experiment/simulation_data_03_07_2025_15_53_57.npz",
#     "/home/atari/workspace/iterative_supervised_learning/examples/data/behavior_cloning/trot/Mar_07_2025_15_50_55/dataset/experiment/simulation_data_03_07_2025_15_54_14.npz"
# ]

# Mar_05_2025_15_10_53
data_paths = [
    "/home/atari/workspace/iterative_supervised_learning/examples/data/behavior_cloning/trot/Mar_05_2025_15_10_53/dataset/experiment/simulation_data_03_05_2025_15_11_11.npz",
    # "/home/atari/workspace/iterative_supervised_learning/examples/data/behavior_cloning/trot/Mar_05_2025_15_10_53/dataset/experiment/simulation_data_03_05_2025_15_11_42.npz",
    # "/home/atari/workspace/iterative_supervised_learning/examples/data/behavior_cloning/trot/Mar_05_2025_15_10_53/dataset/experiment/simulation_data_03_05_2025_15_11_59.npz",
    # "/home/atari/workspace/iterative_supervised_learning/examples/data/behavior_cloning/trot/Mar_05_2025_15_10_53/dataset/experiment/simulation_data_03_05_2025_15_12_31.npz",
    # "/home/atari/workspace/iterative_supervised_learning/examples/data/behavior_cloning/trot/Mar_05_2025_15_10_53/dataset/experiment/simulation_data_03_05_2025_15_13_02.npz",
    # "/home/atari/workspace/iterative_supervised_learning/examples/data/behavior_cloning/trot/Mar_05_2025_15_10_53/dataset/experiment/simulation_data_03_05_2025_15_13_20.npz",
    # "/home/atari/workspace/iterative_supervised_learning/examples/data/behavior_cloning/trot/Mar_05_2025_15_10_53/dataset/experiment/simulation_data_03_05_2025_15_13_37.npz",
    
    # "/home/atari/workspace/iterative_supervised_learning/examples/data/behavior_cloning/trot/Mar_05_2025_15_10_53/dataset/experiment/simulation_data_03_05_2025_15_16_48.npz",
    # "/home/atari/workspace/iterative_supervised_learning/examples/data/behavior_cloning/trot/Mar_05_2025_15_10_53/dataset/experiment/simulation_data_03_05_2025_15_18_38.npz"
]

# Define joint labels in the specified order
joint_labels = [
    "FL_hip", "FL_thigh", "FL_calf",
    "FR_hip", "FR_thigh", "FR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
    "RR_hip", "RR_thigh", "RR_calf"
]

# Global variables
visualize_length = 2000

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
