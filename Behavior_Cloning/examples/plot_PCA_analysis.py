from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_pca_trajectories(data_dict, title):
    """
    Applies PCA to each trajectory and plots them in a shared 2D PCA space.

    Parameters:
        data_dict (dict): Dictionary where keys are trajectory names, 
                          and values are numpy arrays of shape (timesteps, features).
        title (str): Plot title (e.g., "PCA of Joint Positions").
    """
    all_data = []
    labels = []

    # Flatten and collect data across all trajectories
    for traj_name, traj_data in data_dict.items():
        flattened = traj_data.reshape(-1, traj_data.shape[-1])  # [timesteps, features]
        all_data.append(flattened)
        labels.extend([traj_name] * len(flattened))

    all_data = np.vstack(all_data)

    # Standardize
    scaler = StandardScaler()
    all_data_std = scaler.fit_transform(all_data)

    # PCA to 2D
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(all_data_std)

    # Plot
    plt.figure(figsize=(10, 8))
    unique_labels = sorted(set(labels))
    
    for label in unique_labels:
        idx = [i for i, l in enumerate(labels) if l == label]
        plt.scatter(pcs[idx, 0], pcs[idx, 1], label=label, alpha=0.5, s=10)

    # # Inside your plot loop
    # for i, label in enumerate(unique_labels):
    #     idx = [j for j, l in enumerate(labels) if l == label]
    #     pc_traj = pcs[idx]
    #     plt.scatter(pc_traj[:, 0], pc_traj[:, 1], c=np.linspace(0, 1, len(pc_traj)), cmap='viridis', s=10)

    plt.title(f"{title} (PCA 2D Projection)")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.legend(title="Trajectory", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def plot_pca_on_combined_array(all_data, title):
    # Standardize
    scaler = StandardScaler()
    all_data_std = scaler.fit_transform(all_data)

    # PCA to 2D
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(all_data_std)

    # Plot all data as one big cloud
    plt.figure(figsize=(8, 6))
    plt.scatter(pcs[:, 0], pcs[:, 1], alpha=0.5, s=5)
    plt.title(f"{title} (PCA 2D Projection)")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_pca_with_policy_overlay_all(train_data_dict, policy_npz_path, title_prefix="PCA Comparison"):
    """
    Applies PCA on combined training data and overlays learned policy trajectory.
    Generates three subplots: joint position, velocity, and PD target.

    Parameters:
        train_data_dict (dict): Dictionary with keys 'position', 'velocity', 'action', each a (N, D) ndarray.
        policy_npz_path (str): Path to the .npz file of the policy rollout.
        title_prefix (str): Title prefix for the plots.
    """
    policy_data = np.load(policy_npz_path)
    modalities = {
        "Position": {
            "train": train_data_dict['position'],
            "policy": policy_data["q"][:, 7:]
        },
        "Velocity": {
            "train": train_data_dict['velocity'],
            "policy": policy_data["v"][:, 6:]
        },
        "PD Target": {
            "train": train_data_dict['action'],
            "policy": policy_data["action"]
        }
    }

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, (label, data) in enumerate(modalities.items()):
        train_data = data["train"]
        policy_data = data["policy"]

        # Fit scaler and PCA on training data
        scaler = StandardScaler()
        train_std = scaler.fit_transform(train_data)
        pca = PCA(n_components=2)
        pcs_train = pca.fit_transform(train_std)

        # Project policy data
        policy_std = scaler.transform(policy_data)
        pcs_policy = pca.transform(policy_std)

        ax = axs[i]
        ax.scatter(pcs_train[:, 0], pcs_train[:, 1], alpha=0.2, s=5, label="Training Data")
        ax.plot(pcs_policy[:, 0], pcs_policy[:, 1], color='red', linewidth=2, label="Policy Trajectory")
        ax.scatter(pcs_policy[0, 0], pcs_policy[0, 1], color='green', label='Start', zorder=5)
        ax.scatter(pcs_policy[-1, 0], pcs_policy[-1, 1], color='black', label='End', zorder=5)
        
        ax.set_title(f"{title_prefix} - {label}")
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.grid(True)
        if i == 0:
            ax.legend(loc='best')

    plt.tight_layout()
    plt.show()

def plot_pca_state_action_overlay(train_state_action, policy_npz_path, title="PCA: State + Action"):
    policy_data = np.load(policy_npz_path)
    policy_state = policy_data["state"]  # shape: [T, 47]
    policy_action = policy_data["action"]  # shape: [T, 12]

    policy_state_action = np.hstack([policy_state, policy_action])

    # Standardize based on training data
    scaler = StandardScaler()
    train_std = scaler.fit_transform(train_state_action)
    policy_std = scaler.transform(policy_state_action)

    # PCA
    pca = PCA(n_components=2)
    pcs_train = pca.fit_transform(train_std)
    pcs_policy = pca.transform(policy_std)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(pcs_train[:, 0], pcs_train[:, 1], alpha=0.2, s=5, label="Training State+Action")
    plt.plot(pcs_policy[:, 0], pcs_policy[:, 1], color='red', linewidth=2, label="Policy Trajectory")
    plt.scatter(pcs_policy[0, 0], pcs_policy[0, 1], color='green', label='Start', zorder=5)
    plt.scatter(pcs_policy[-1, 0], pcs_policy[-1, 1], color='black', label='End', zorder=5)

    plt.title(title)
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_pca_state_action_with_pc3_overlay(train_state_action, policy_npz_path, title="PCA: State + Action"):
    policy_data = np.load(policy_npz_path)
    policy_state = policy_data["state"]  # shape: [T, 47]
    policy_action = policy_data["action"]  # shape: [T, 12]

    policy_state_action = np.hstack([policy_state, policy_action])

    # Standardize based on training data
    scaler = StandardScaler()
    train_std = scaler.fit_transform(train_state_action)
    policy_std = scaler.transform(policy_state_action)

    # PCA with at least 3 components
    pca = PCA(n_components=3)
    pcs_train = pca.fit_transform(train_std)
    pcs_policy = pca.transform(policy_std)

    # Plot PC1 vs PC2 and PC1 vs PC3
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Plot PC1 vs PC2
    axs[0].scatter(pcs_train[:, 0], pcs_train[:, 1], alpha=0.2, s=5, label="Training State+Action")
    axs[0].plot(pcs_policy[:, 0], pcs_policy[:, 1], color='red', linewidth=2, label="Policy Trajectory")
    axs[0].scatter(pcs_policy[0, 0], pcs_policy[0, 1], color='green', label='Start', zorder=5)
    axs[0].scatter(pcs_policy[-1, 0], pcs_policy[-1, 1], color='black', label='End', zorder=5)
    axs[0].set_title(f"{title} (PC1 vs PC2)")
    axs[0].set_xlabel("PC 1")
    axs[0].set_ylabel("PC 2")
    axs[0].legend()
    axs[0].grid(True)

    # Plot PC1 vs PC3
    axs[1].scatter(pcs_train[:, 0], pcs_train[:, 2], alpha=0.2, s=5, label="Training State+Action")
    axs[1].plot(pcs_policy[:, 0], pcs_policy[:, 2], color='red', linewidth=2, label="Policy Trajectory")
    axs[1].scatter(pcs_policy[0, 0], pcs_policy[0, 2], color='green', label="Start", zorder=5)
    axs[1].scatter(pcs_policy[-1, 0], pcs_policy[-1, 2], color='black', label='End', zorder=5)
    axs[1].set_title(f"{title} (PC1 vs PC3)")
    axs[1].set_xlabel("PC 1")
    axs[1].set_ylabel("PC 3")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()



## Example usage
# Set directory path and number of trajectories to visualize
# data_dir = "/home/atari/workspace/Behavior_Cloning/examples/data/behavior_cloning/trot/Apr_04_2025_09_04_52/dataset/experiment"
# data_dir = "/home/atari/workspace/Behavior_Cloning/examples/data/behavior_cloning/trot/Apr_03_2025_12_20_02/dataset/experiment"
# data_dir = "/home/atari/workspace/Behavior_Cloning/examples/data/behavior_cloning/trot/Apr_01_2025_15_09_58/dataset/experiment"

# data_dir = "/home/atari/workspace/Behavior_Cloning/examples/data/behavior_cloning/trot/Apr_10_2025_09_33_49/dataset/experiment"

# dataset that gives me the best policy
# data_dir = "/home/atari/workspace/Behavior_Cloning/examples/data/behavior_cloning/trot/Apr_16_2025_09_42_07/dataset/experiment"
data_dir = "/home/atari/workspace/Behavior_Cloning/examples/data/behavior_cloning/trot/Apr_16_2025_13_02_09/dataset/experiment"
# data_dir = "/home/atari/workspace/Behavior_Cloning/examples/data/behavior_cloning/trot/Apr_16_2025_15_57_03/dataset/experiment"

# data_dir = "/home/atari/workspace/Behavior_Cloning/examples/data/behavior_cloning/trot/Apr_23_2025_18_38_44/dataset/experiment"
k_start = 0  # Number of trajectories to visualize
k_end = None
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
)[k_start:k_end]

# Initialize dictionaries for storing trajectory-wise data
position_dict = {}
velocity_dict = {}
PD_target_dict = {}

# Initialize combined data arrays
combined_position = []
combined_velocity = []
combined_PD_target = []
combined_state_action = []

for idx, traj_file in enumerate(trajectory_files):
    data_path = os.path.join(data_dir, traj_file)
    data = np.load(data_path)

    # Extract joint positions, velocities, and PD target values
    joint_pos_his = data["q"][:, 7:]  # 12 joint positions
    joint_vel_his = data["v"][:, 6:]  # 12 joint velocities
    PD_target_his = data["action"]    # PD target values
    full_state = data["state"]
    
    # Store trajectory-specific data
    traj_label = f"traj_{idx + 1}"
    position_dict[traj_label] = joint_pos_his
    velocity_dict[traj_label] = joint_vel_his
    PD_target_dict[traj_label] = PD_target_his

    # Append to combined dataset
    combined_position.append(joint_pos_his)
    combined_velocity.append(joint_vel_his)
    combined_PD_target.append(PD_target_his)
    
    # Combine state and PD target (action)
    state_action = np.hstack([full_state, PD_target_his])  # [N, 59]
    combined_state_action.append(state_action)

# Convert combined lists into numpy arrays
combined_position = np.vstack(combined_position) if combined_position else None
combined_velocity = np.vstack(combined_velocity) if combined_velocity else None
combined_PD_target = np.vstack(combined_PD_target) if combined_PD_target else None
combined_state_action = np.vstack(combined_state_action)

# traj-wise pca
# plot_pca_trajectories(position_dict, "Joint Position Trajectories")
# plot_pca_trajectories(velocity_dict, "Joint Velocity Trajectories")
# plot_pca_trajectories(PD_target_dict, "PD Target Trajectories")

# combined data pca:
# plot_pca_on_combined_array(combined_position, "Combined Joint Positions")
# plot_pca_on_combined_array(combined_velocity, "Combined Joint Velocities")
# plot_pca_on_combined_array(combined_PD_target, "Combined PD Targets")

# policy data overlay
# policy_path = "/home/atari/workspace/Behavior_Cloning/examples/data/simulation_data_04_11_2025_12_48_57.npz"
policy_path = "/home/atari/workspace/Behavior_Cloning/examples/data/simulation_data_04_27_2025_20_22_15.npz"
train_data_dict = {
    'position': combined_position,
    'velocity': combined_velocity,
    'action': combined_PD_target
}

plot_pca_with_policy_overlay_all(train_data_dict, policy_path)

plot_pca_state_action_overlay(combined_state_action, policy_path)

# plot_pca_state_action_with_pc3_overlay(combined_state_action, policy_path)
