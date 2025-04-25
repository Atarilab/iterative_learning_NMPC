## NOTE: This script is for plotting the velocity-conditioned goal-reaching error.
import numpy as np
import matplotlib.pyplot as plt

def plot_velocity_tracking(timesteps, vx_mpc, vy_mpc, vx_ref, vy_ref, 
                           vx_policy=None, vy_policy=None, title="Velocity-conditioned Goal Reaching Error"):
    """
    Plots velocity tracking (vx, vy) for MPC and optionally for a policy.

    Args:
        timesteps (np.ndarray): 1D array of timesteps
        vx_mpc (np.ndarray): vx from MPC
        vy_mpc (np.ndarray): vy from MPC
        vx_ref (float): reference vx
        vy_ref (float): reference vy
        vx_policy (np.ndarray, optional): vx from policy
        vy_policy (np.ndarray, optional): vy from policy
        title (str): title of the plot
    """
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # vx subplot
    axs[0].plot(timesteps, vx_mpc, label='MPC', linestyle='--')
    axs[0].plot(timesteps, np.ones_like(timesteps) * vx_ref, label='Reference', linestyle='-', color='red', linewidth=2)
    if vx_policy is not None:
        axs[0].plot(timesteps, vx_policy, label='Policy')
    axs[0].set_ylabel('vx [m/s]')
    axs[0].legend()
    axs[0].grid(True)

    # vy subplot
    axs[1].plot(timesteps, vy_mpc, label='MPC', linestyle='--')
    axs[1].plot(timesteps, np.ones_like(timesteps) * vy_ref, label='Reference', linestyle='-', color='red', linewidth=2)
    if vy_policy is not None:
        axs[1].plot(timesteps, vy_policy, label='Policy')
    axs[1].set_ylabel('vy [m/s]')
    axs[1].set_xlabel('Timestep')
    axs[1].grid(True)

    plt.tight_layout()
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.show()


# Constants
vx_ref = 0.15
vy_ref = 0.0
visualization_length = 3000
timesteps = np.arange(visualization_length)

# Load MPC data
mpc_path = "/home/atari/workspace/iterative_supervised_learning/examples/data/simulation_data_03_26_2025_14_10_11.npz"
data_mpc = np.load(mpc_path)
vx_mpc = data_mpc["v"][:visualization_length, 0]
vy_mpc = data_mpc["v"][:visualization_length, 1]

# load policy data
policy_path = "/home/atari/workspace/iterative_supervised_learning/utils/data/working_policy_traj.npz"
data_policy = np.load(policy_path)
vx_policy = data_policy["v"][:visualization_length, 0]
vy_policy = data_policy["v"][:visualization_length, 1]

# Call plotting function
plot_velocity_tracking(timesteps, vx_mpc, vy_mpc, vx_ref, vy_ref, vx_policy, vy_policy)

