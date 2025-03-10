import numpy as np
import matplotlib.pyplot as plt

# global variables
visualize_length = 1000

# NOTE: read policy data from file
# kp = 40 kd = 5.0
# dummy (manual defined PD targets)
# data_path = "/home/atari/workspace/iterative_supervised_learning/utils/data/simulation_data_03_07_2025_10_21_10.npz"
# MPC (use mpc generated PD targets)
# data_path = "/home/atari/workspace/iterative_supervised_learning/utils/data/mpc_replay_kp40kd5_with_noise.npz"

# kp = 100 kd = 5.0
# data_path = "/home/atari/workspace/iterative_supervised_learning/utils/data/simulation_data_03_07_2025_10_22_54.npz"

#kp = 20 kd = 1.5
# MPC
# data_path = "/home/atari/workspace/iterative_supervised_learning/utils/data/mpc_replay_kp20kd1.5_with_noise.npz"
# policy
# data_path = "/home/atari/workspace/iterative_supervised_learning/utils/data/policy_rollout_kp20kd1.5.npz"
# data_path = "/home/atari/workspace/iterative_supervised_learning/utils/data/policy_rollout_kp20kd1.5_reduced_frequency.npz"

data_path = "/home/atari/workspace/iterative_supervised_learning/utils/data/policy_100_rollout_kp20_kd1.5.npz"

# policy rollout with realtime MPC data
#============================================
# kp = 2.0 kd = 0.1
# MPC
# data_path = "/home/atari/workspace/iterative_supervised_learning/utils/data/mpc_replay_kp2.0kd0.1_without_noise.npz"

# kp = 10.0 kd = 1.0
# MPC
# data_path = "/home/atari/workspace/iterative_supervised_learning/utils/data/mpc_replay_kp10kd1_without_noise.npz"
data = np.load(data_path)

# Define joint labels in the specified order
joint_labels = [
    "FL_hip", "FL_thigh", "FL_calf",
    "FR_hip", "FR_thigh", "FR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
    "RR_hip", "RR_thigh", "RR_calf"
]

# extract different variables from data
time_his = data["time"][:visualize_length]
joint_pos_his = data["q"][:,7:][:visualize_length]
joint_vel_his = data["v"][:,6:][:visualize_length]
realized_PD_target = data["action"][:visualize_length]

# extract reference variables
# dummy controller 
# action = [0,0.9,-1.8,
#           0,0.9,-1.8,
#           0,0.9,-1.8,
#           0,0.9,-1.8,]

# action = [0.3,1.0,-1.8,
#           -0.3,1.0,-1.8,
#           0.3,1.0,-1.8,
#           -0.3,1.0,-1.8,]
# reference_PD_target = np.tile(action, (len(time_his), 1))  # Repeating reference actions for all time steps
# reference_joint_pos = reference_PD_target  # Assuming reference positions are the same
# reference_joint_vel = np.zeros((len(time_his), 12))  # Zero reference velocities

# NOTE: read MPC PD target as reference
# data_MPC_path = "/home/atari/workspace/iterative_supervised_learning/examples/data/kp40_kd5.npz"
data_MPC_path = "/home/atari/workspace/iterative_supervised_learning/examples/data/kp20_kd1.5.npz"
# data_MPC_path = "/home/atari/workspace/iterative_supervised_learning/examples/data/kp2_kd0.1.npz"
# data_MPC_path = "/home/atari/workspace/iterative_supervised_learning/examples/data/kp10_kd1.npz"

data_MPC = np.load(data_MPC_path)
reference_PD_target = data_MPC["action"][:visualize_length,:]
reference_joint_pos = data_MPC["q"][:visualize_length,7:]
reference_joint_vel = data_MPC["v"][:visualize_length,6:]

# visualize
def plot_joint_tracking(data_real, data_ref, title, ylabel):
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))  # 4 rows, 3 columns for 12 joints
    for i, ax in enumerate(axes.flat):
        ax.plot(time_his, data_real[:, i], label="Realized", color="blue")
        ax.plot(time_his, data_ref[:, i], linestyle="--", label="Reference", color="red")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} - {joint_labels[i]}")
        ax.legend()
        ax.grid()
    plt.tight_layout()
    plt.show()

def plot_joint_tracking_with_policy(data_real, data_ref, data_policy, title, ylabel):
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))  # 4 rows, 3 columns for 12 joints
    for i, ax in enumerate(axes.flat):
        ax.plot(time_his, data_real[:, i], label="Realized", color="blue")
        ax.plot(time_his, data_ref[:, i], linestyle="--", label="Reference", color="red")
        ax.plot(time_his, data_policy[:, i], linestyle=":", label="Policy Output", color="green")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} - {joint_labels[i]}")
        ax.legend()
        ax.grid()
    plt.tight_layout()
    plt.show()
    
# 1- PD target tracking
# Load Policy-generated actions (`action_policy_his`)
policy_action_path = "/home/atari/workspace/iterative_supervised_learning/utils/data/action_policy_history.npz"
policy_data = np.load(policy_action_path)
action_policy_his = policy_data["action_policy_his"][:visualize_length, :]  # Ensure correct length
plot_joint_tracking_with_policy(realized_PD_target, reference_PD_target, action_policy_his, "PD Target Tracking", "PD Target")

# 2- joint position tracking
plot_joint_tracking(joint_pos_his, reference_joint_pos, "Joint Position Tracking", "Joint Position")

# 3- joint velocity tracking
plot_joint_tracking(joint_vel_his, reference_joint_vel, "Joint Velocity Tracking", "Joint Velocity")
