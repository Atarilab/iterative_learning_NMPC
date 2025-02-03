## NOTE: This script is for data collection for behavior cloning.
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../..')

import argparse
from typing import Tuple, List
import os
import numpy as np
from datetime import datetime
from iterative_supervised_learning.utils.RolloutMPC import RolloutMPC
import random

def rollout_mpc(mode: str = "close_loop",
                       sim_time: float = 5,
                       robot_name: str = "go2",
                       record_dir: str = "./data/",
                       v_des: List[float] = [0.5, 0.0, 0.0],
                       save_data: bool = True,
                       interactive: bool = False,
                       record_video: bool = False,
                       visualize: bool = False) -> Tuple[str, List[float], List[List[float]], List[List[float]], List[List[float]]]:
    """
    Function to run MPC simulation with specified parameters.

    Args:
        mode (str): Mode of simulation ('traj_opt', 'open_loop', 'close_loop').
        sim_time (float): Total simulation time.
        robot_name (str): Name of the robot.
        record_dir (str): Directory to save recorded data.
        v_des (List[float]): Desired velocity (x, y, yaw).
        save_data (bool): Whether to save recorded data.
        interactive (bool): Use interactive mode for setting velocity goals.
        record_video (bool): Record a video of the simulation.
        visualize (bool): Enable or disable visualization.

    Returns:
        Tuple[str, List[float], List[List[float]], List[List[float]], List[List[float]]]:
            - Path to the recorded data directory.
            - Time array of the simulation.
            - Recorded positions (q).
            - Recorded velocities (v).
            - Recorded control inputs (ctrl).
    """
    # Create argparse-like structure
    class Args:
        def __init__(self):
            self.mode = mode
            self.sim_time = sim_time
            self.robot_name = robot_name
            self.record_dir = record_dir
            self.v_des = v_des
            self.save_data = save_data
            self.interactive = interactive
            self.record_video = record_video
            self.visualize = visualize

    args = Args()

    # Ensure the record directory exists
    if save_data:
        os.makedirs(record_dir, exist_ok=True)

    # Instantiate the RolloutMPC class
    rollout_mpc = RolloutMPC(args)

    # Run the appropriate simulation
    if mode == 'traj_opt':
        rollout_mpc.run_traj_opt()
    elif mode == 'open_loop':
        rollout_mpc.run_open_loop()
    elif mode == 'close_loop':
        rollout_mpc.run_mpc()
    else:
        raise ValueError("Invalid mode. Choose from 'traj_opt', 'open_loop', or 'close_loop'.")

    # Collect and return recorded data
    if save_data:
        data_file = None
        for file in os.listdir(record_dir):
            if file.startswith("simulation_data_") and file.endswith(".npz"):
                data_file = os.path.join(record_dir, file)
                break
        
        if data_file:
            data = np.load(data_file)
            print("data loaded from", data_file)
            # here time is the time stamp
            # q is the [x,y,z] of the base point
            # v is the [vx,vy.vz] of the base point
            time_array = data["time"].tolist()
            q_array = data["q"].tolist()
            v_array = data["v"].tolist()
            ctrl_array = data["ctrl"].tolist()
            
            # Extract only the first three entries
            q_first_three = [entry[:3] for entry in q_array]
            v_first_three = [entry[:3] for entry in v_array]
            
            return record_dir, time_array, q_first_three, v_first_three, ctrl_array
    return record_dir, [], [], [], []

# Example usage
if __name__ == "__main__":
    # Define goal space
    vx_des_min, vx_des_max = 0.0, 0.5
    vy_des_min, vy_des_max = -0.1, 0.1
    w_des_min, w_des_max = 0.0, 0.0
    data_save_path = "./data"
    num_goals_each_dim = 4

    # Generate grid sampled goals
    vx_values = np.linspace(vx_des_min, vx_des_max, num_goals_each_dim)
    vy_values = np.linspace(vy_des_min, vy_des_max, num_goals_each_dim)
    w_values = np.linspace(w_des_min, w_des_max, 1)  # Single value for w since min == max

    # Initialize dataset storage
    collected_data = {
        "time": [],
        "q": [],
        "v": [],
        "ctrl": []
    }

    # Rollout MPC on the selected goals
    for i, vx in enumerate(vx_values):
        for j, vy in enumerate(vy_values):
            for k, w in enumerate(w_values):
                v_des = [vx, vy, w]
                
                # Rollout with MPC
                record_dir, time, q, v, ctrl = rollout_mpc(
                    mode="close_loop",
                    sim_time=5,
                    robot_name="go2",
                    record_dir=f"{data_save_path}/iteration_{i}_{j}_{k}/",
                    v_des=v_des,
                    save_data=True,
                    interactive=False,
                    record_video=False,
                    visualize=False
                )
                
                # Append collected data
                collected_data["time"].extend(time)
                collected_data["q"].extend(q)
                collected_data["v"].extend(v)
                collected_data["ctrl"].extend(ctrl)

    # Save combined dataset
    np.savez(os.path.join(data_save_path, "collected_data.npz"), 
             time=collected_data["time"],
             q=collected_data["q"],
             v=collected_data["v"],
             ctrl=collected_data["ctrl"])
    
    print(f"Collected dataset saved at {data_save_path}/collected_data.npz")
        
