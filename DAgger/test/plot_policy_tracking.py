import numpy as np
import matplotlib.pyplot as plt

# Global variables
visualize_length = 20000

realized_traj_files = [
#    "/home/atari/workspace/DAgger/test/data/traj_nominal_05_08_2025_12_31_08.npz"
   "/home/atari/workspace/DAgger/example/data/multigoal-locosafedagger/May_08_2025_12_06_04/iter_2/dataset/experiment/traj_nominal_05_08_2025_12_10_14.npz" 
    ]
data_MPC_path = "/home/atari/workspace/Behavior_Cloning/examples/data/behavior_cloning/trot/Apr_16_2025_13_02_09/dataset/experiment/traj_nominal_04_16_2025_13_02_15.npz"

# Load realized data from selected files (choose range a to b)
a, b = 0, 3  # Define your range here
realized_data = [np.load(file) for file in realized_traj_files[a:b]]

# Define joint labels
joint_labels = [
    "FL_hip", "FL_thigh", "FL_calf",
    "FR_hip", "FR_thigh", "FR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
    "RR_hip", "RR_thigh", "RR_calf"
]

# Extract realized data variables
time_his_policies = [data["time"][:visualize_length] for data in realized_data]
joint_pos_his_list = [data["q"][:,7:][:visualize_length] for data in realized_data]
joint_vel_his_list = [data["v"][:,6:][:visualize_length] for data in realized_data]
realized_PD_targets = [data["action"][:visualize_length] for data in realized_data]
phase_percentage_his_list = [data["state"][:,0][:visualize_length] for data in realized_data]
base_pos_his_list = [data["q"][:,:3][:visualize_length] for data in realized_data]
base_vel_his_list = [data["v"][:,:3][:visualize_length] for data in realized_data]

# Load MPC reference data
data_MPC = np.load(data_MPC_path)

time_his_MPC = data_MPC["time"][:visualize_length]
reference_PD_target = data_MPC["action"][:visualize_length, :]
reference_joint_pos = data_MPC["q"][:visualize_length, 7:]
reference_joint_vel = data_MPC["v"][:visualize_length, 6:]
reference_base_pos = data_MPC["q"][:visualize_length, :3]
reference_base_vel = data_MPC["v"][:visualize_length, :3]

# Load policy-generated actions
# policy_action_path = "/home/atari/workspace/Behavior_Cloning/utils/data/action_policy_history.npz"
# policy_data = np.load(policy_action_path)
# action_policy_his = policy_data["action_policy_his"][:visualize_length, :]

# Plot functions
def plot_joint_tracking_multiple(realized_data_list, data_ref, phase_percentage_list=None, title="Joint Tracking", ylabel="Value"):
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    for i, ax in enumerate(axes.flat):
        for idx, data_real in enumerate(realized_data_list):
            ax.plot(time_his_policies[idx], data_real[:, i], label=f"Realized {idx+1}", color=["blue", "black","green"][idx])
        ax.plot(time_his_MPC, data_ref[:, i], linestyle="--", label="Reference", color="red")
        
        if phase_percentage_list is not None:
            for idx, phase in enumerate(phase_percentage_list):
                ax.plot(time_his_policies[idx], phase, linestyle="-", label=f'Phase {idx+1}', color="gray")
        
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} - {joint_labels[i]}")
        ax.legend()
        ax.grid()
    plt.tight_layout()
    plt.show()

def plot_base_tracking_multiple(realized_data_list, data_ref, time_his_policies, time_his_ref, title="Base Tracking", ylabel="Value", dim_labels=["x", "y", "z"]):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, ax in enumerate(axes):
        for idx, data_real in enumerate(realized_data_list):
            ax.plot(time_his_policies[idx], data_real[:, i], label=f"Realized {idx+1}", color=["blue", "black", "green"][idx])
        ax.plot(time_his_ref, data_ref[:, i], linestyle="--", label="Reference", color="red")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} - {dim_labels[i]}")
        ax.legend()
        ax.grid()

    plt.tight_layout()
    plt.show()


# Plot
plot_joint_tracking_multiple(realized_PD_targets, reference_PD_target, phase_percentage_his_list, "PD Target Tracking", "PD Target")
plot_joint_tracking_multiple(joint_pos_his_list, reference_joint_pos, None,"Joint Position Tracking", "Joint Position")
plot_joint_tracking_multiple(joint_vel_his_list, reference_joint_vel, None,"Joint Velocity Tracking", "Joint Velocity")

plot_base_tracking_multiple(
    base_pos_his_list,
    reference_base_pos,
    time_his_policies,
    time_his_MPC,
    title="Base Position Tracking",
    ylabel="Position (m)"
)

plot_base_tracking_multiple(
    base_vel_his_list,
    reference_base_vel,
    time_his_policies,
    time_his_MPC,
    title="Base Velocity Tracking",
    ylabel="Velocity (m/s)"
)

