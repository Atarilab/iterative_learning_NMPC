## NOTE： This script is for testing MPC rollout, a nominal trajectory should be collected, along with state, action history, realized velocity-conditioned goal and contact-conditioned goal, etc.
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../..')


from typing import List
import pinocchio as pin
import numpy as np
from mj_pin.abstract import VisualCallback, DataRecorder # type: ignore
from mj_pin.simulator import Simulator # type: ignore
from mj_pin.utils import load_mj_pin   # type: ignore

from mpc_controller.mpc import LocomotionMPC

from iterative_supervised_learning.utils.RolloutMPC import RolloutMPC

def test_rollout_mpc(sim_time = 5,  # Simulation time in seconds
    sim_dt = 1.0e-3,  # Simulation time step
    robot_name = "go2",  # Robot name
    record_dir = "./data/",  # Directory to save recorded data
    v_des = [0.5, 0.0, 0.0]  # Desired velocity command
    ):
    """
    Test the RolloutMPC functionality and return recorded data.
    """
    # sim_time = 5  # Simulation time in seconds
    # sim_dt = 1.0e-3  # Simulation time step
    # robot_name = "go2"  # Robot name
    # record_dir = "./data/"  # Directory to save recorded data
    # v_des = [0.5, 0.0, 0.0]  # Desired velocity command

    # Create and execute RolloutMPC
    rollout_mpc = RolloutMPC(sim_time, sim_dt, robot_name, record_dir, v_des)
    rollout_mpc.execute()

    # Return recorded data as a dictionary
    recorded_data = {
        "time": rollout_mpc.recorded_time,
        "q": rollout_mpc.recorded_q,
        "v": rollout_mpc.recorded_v,
        "ctrl": rollout_mpc.recorded_ctrl,
    }

    return recorded_data

if __name__ == "__main__":
    recorded_data = test_rollout_mpc()

    # Print recorded data for verification
    # print("Recorded time:", recorded_data["time"])
    # print("Recorded positions (q):", recorded_data["q"])
    # print("Recorded velocities (v):", recorded_data["v"])
    # print("Recorded controls (ctrl):", recorded_data["ctrl"])
