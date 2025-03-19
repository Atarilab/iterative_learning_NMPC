import numpy as np
import matplotlib.pyplot as plt

# global variables
visualize_length = 1000

# NOTE: read realized data from file
# data_path = "/home/atari/workspace/iterative_supervised_learning/utils/data/simulation_data_03_19_2025_10_04_21.npz"
# data_path = "/home/atari/workspace/iterative_supervised_learning/utils/data/simulation_data_03_19_2025_10_40_13.npz"
data_path = "/home/atari/workspace/iterative_supervised_learning/utils/data/simulation_data_03_19_2025_13_30_47.npz"
data = np.load(data_path)

data_path1 = "/home/atari/workspace/iterative_supervised_learning/utils/data/simulation_data_03_19_2025_13_00_29.npz"
data1 = np.load(data_path1) 

# Define joint labels in the specified order
joint_labels = [
    "FL_hip", "FL_thigh", "FL_calf",
    "FR_hip", "FR_thigh", "FR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
    "RR_hip", "RR_thigh", "RR_calf"
]

# extract different variables from data
time_his_policy = data["time"][:visualize_length]
joint_pos_his = data["q"][:,7:][:visualize_length]
joint_vel_his = data["v"][:,6:][:visualize_length]
realized_PD_target = data["action"][:visualize_length]
phase_percentage_his = data["state"][:,0][:visualize_length]

time_his_policy1 = data1["time"][:visualize_length]
joint_pos_his1 = data1["q"][:,7:][:visualize_length]
joint_vel_his1 = data1["v"][:,6:][:visualize_length]
realized_PD_target1 = data1["action"][:visualize_length]
phase_percentage_his1 = data1["state"][:,0][:visualize_length]


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
# without phase-percentage shift

# with phase-percentage shift
data_MPC_path = "/home/atari/workspace/iterative_supervised_learning/examples/data/behavior_cloning/trot/Mar_18_2025_11_07_47/dataset/experiment/simulation_data_03_18_2025_11_08_03.npz" # kp = 20 kd = 1.5

data_MPC = np.load(data_MPC_path)
time_his_MPC = data_MPC["time"][:visualize_length]
reference_PD_target = data_MPC["action"][:visualize_length,:]
reference_joint_pos = data_MPC["q"][:visualize_length,7:]
reference_joint_vel = data_MPC["v"][:visualize_length,6:]
# phase_percentage_his = data_MPC["state"][:,0][:visualize_length]

# visualize
def plot_joint_tracking(data_real, data_ref, title, ylabel):
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))  # 4 rows, 3 columns for 12 joints
    for i, ax in enumerate(axes.flat):
        ax.plot(time_his_policy, data_real[:, i], label="Realized", color="blue")
        ax.plot(time_his_MPC, data_ref[:, i], linestyle="--", label="Reference", color="red")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} - {joint_labels[i]}")
        ax.legend()
        ax.grid()
    plt.tight_layout()
    plt.show()

def plot_joint_tracking_with_policy(data_real, data_ref, data_policy, phase_percentage, title, ylabel):
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))  # 4 rows, 3 columns for 12 joints
    for i, ax in enumerate(axes.flat):
        ax.plot(time_his_policy, data_real[:, i], label="Realized", color="blue")
        ax.plot(time_his_MPC, data_ref[:, i], linestyle="--", label="Reference", color="red")
        ax.plot(time_his_MPC, data_policy[:, i], linestyle=":", label="Policy Output", color="green")
        ax.plot(time_his_policy, phase_percentage, label='Phase Percentage', color='black')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} - {joint_labels[i]}")
        ax.legend()
        ax.grid()
    plt.tight_layout()
    plt.show()

def plot_joint_tracking_multiple(data_real, data_real1, data_ref, title, ylabel):
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))  # 4 rows, 3 columns for 12 joints
    for i, ax in enumerate(axes.flat):
        ax.plot(time_his_policy, data_real[:, i], label="Realized", color="blue")
        # ax.plot(time_his_policy1, data_real1[:, i], label="Realized", color="black")
        ax.plot(time_his_MPC, data_ref[:, i], linestyle="--", label="Reference", color="red")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} - {joint_labels[i]}")
        ax.legend()
        ax.grid()
    plt.tight_layout()
    plt.show()

def plot_joint_tracking_with_policy_multiple(data_real, data_real1, data_ref, data_policy, phase_percentage, title, ylabel):
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))  # 4 rows, 3 columns for 12 joints
    for i, ax in enumerate(axes.flat):
        ax.plot(time_his_policy, data_real[:, i], label="Realized", color="blue")
        # ax.plot(time_his_policy1, data_real1[:, i], label="Realized", color="black")
        ax.plot(time_his_MPC, data_ref[:, i], linestyle="--", label="Reference", color="red")
        ax.plot(time_his_MPC, data_policy[:, i], linestyle=":", label="Policy Output", color="green")
        ax.plot(time_his_policy, phase_percentage, label='Phase Percentage', color='black')
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
# plot_joint_tracking_with_policy(realized_PD_target, reference_PD_target, action_policy_his, "PD Target Tracking", "PD Target")
# plot_joint_tracking_with_policy(realized_PD_target, reference_PD_target, action_policy_his, phase_percentage_his, "PD Target Tracking", "PD Target")

plot_joint_tracking_with_policy_multiple(realized_PD_target,realized_PD_target1,reference_PD_target, action_policy_his, phase_percentage_his, "PD Target Tracking", "PD Target")

# # Plot phase percentage vs time
# plt.figure(figsize=(10, 5))
# plt.plot(time_his, phase_percentage_his, label='Phase Percentage', color='b')
# plt.xlabel('Time')
# plt.ylabel('Phase Percentage')
# plt.title('Phase Percentage vs Time')
# plt.legend()
# plt.grid()
# plt.show()


# 2- joint position tracking
# plot_joint_tracking(joint_pos_his, reference_joint_pos, "Joint Position Tracking", "Joint Position")
plot_joint_tracking_multiple(joint_pos_his, joint_pos_his1, reference_joint_pos, "Joint Position Tracking", "Joint Position")

# 3- joint velocity tracking
# plot_joint_tracking(joint_vel_his, reference_joint_vel, "Joint Velocity Tracking", "Joint Velocity")
plot_joint_tracking_multiple(joint_vel_his, joint_vel_his1, reference_joint_vel, "Joint Velocity Tracking", "Joint Velocity")
