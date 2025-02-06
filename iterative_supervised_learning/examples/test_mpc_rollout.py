## NOTE： This script is for testing MPC rollout, a nominal trajectory should be collected, along with state, action history, realized velocity-conditioned goal and contact-conditioned goal, etc.
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

# NOTE: from Xun's code base-relative foot position is included in the state variable, but how to get
# the foot(end-effector) position?
def base_wrt_foot(q):
    """Calculate relative x, y distance of robot base frame from end effector

    Args:
        q (_type_): current robot configuration

    Returns:
        out: [x, y] * number of end effectors
    """    
    # initilize output array    
    out = np.zeros(2*len(f_arr))
    
    # loop for each end effector
    for i in range(len(f_arr)):
        # get translation of end effector from origin frame
        # TODO: how to get the end-effector position
        foot = self.pin_robot.data.oMf[self.pin_robot.model.getFrameId(self.f_arr[i])].translation
        # get relative distance of robot base frame from end effector
        out[2*i:2*(i+1)] = q[0:2] - foot[0:2]
        
    return out


def rollout_mpc(mode: str = "close_loop",
                sim_time: float = 5,
                sim_dt: float = 0.001,
                start_time: float = 0.0,
                robot_name: str = "go2",
                record_dir: str = "./data/",
                v_des: List[float] = [0.5, 0.0, 0.0],
                save_data: bool = True,
                interactive: bool = False,
                record_video: bool = False,
                visualize: bool = False) -> Tuple[str, List[float], List[List[float]], List[List[float]], List[List[float]]]:

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

    # NOTE: Why is phase percentage a part of vc_goals?
    def phase_percentage(t:int):
        """get current gait phase percentage based on gait period

        Args:
            t (int): current sim step (NOT sim time!)

        Returns:
            phi: current gait phase. between 0 - 1
        """        
        phi = ((t*sim_dt) % self.gait_params.gait_period)/self.gait_params.gait_period
        return phi
    
    # define some global variables
    n_state = 36
    n_action = 12
    nv = 18
    f_arr = ["FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT"]
    kp = 2.0
    kd = 0.1

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
        
        # TODO: return as a compact state and action:
        """
        state: 
            1- v (robot velocity): 6 base velocities(3 linear, 3 angular) + 12 joint velocities(4 legs * 3 joints/leg) = 18
            2- base_wrt_foot(q): relative x,y distances from the robot's base to each foot (4 feet * 2 values(x,y)) = 8  -- this thing is now not implemented
            3- q[2:] :q is full configuration vector: base position(x,y,z), base orientation(quaternion: x,y,z,w), 12 joint angles, total 19
                we exclude first 2 elements of q which is (x,y), and have (z,quaternion(4),12 joint angles) -> 17
            
            Finally: n_state = 18+8+17 = 43
        
        action: 4 legs * 3 joints/leg
        base: q[0:3]
        """
        # define return variables
        num_time_steps = int(sim_time / sim_dt) - int(start_time / sim_dt)
        state_history = np.zeros((num_time_steps, n_state))
        base_history = np.zeros((num_time_steps, 3))
        vc_goal_history = np.zeros((num_time_steps, 3))
        cc_goal_history = np.zeros((num_time_steps, 3))  # Assuming it should be 3D
        action_history = np.zeros((num_time_steps, n_action)) # define action space
        
    
        for file in os.listdir(record_dir):
            if file.startswith("simulation_data_") and file.endswith(".npz"):
                data_file = os.path.join(record_dir, file)
                break
        
        if data_file:
            data = np.load(data_file)
            print("data loaded from", data_file)
            
            time_array = np.array(data["time"])
            q_array = np.array(data["q"])
            v_array = np.array(data["v"])
            ctrl_array = np.array(data["ctrl"])
            
            # Extract base position (x, y, z)
            base_history = q_array[:, :3]
            
            # form state and action history
            for i in range(num_time_steps):
                current_time = time_array[i]  # Get current simulation time
                q = q_array[i]
                v = v_array[i]

                # Store simulation time in first column
                state_history[i, 0] = current_time

                # Store velocity in state_history (starting from column 1)
                state_history[i, 1:nv + 1] = v

                # Store base-relative foot positions (shifted accordingly)
                # state_history[i, nv + 1:nv + 1 + 2 * len(f_arr)] = base_wrt_foot(q)

                # Store configuration (excluding first two elements)
                state_history[i, nv + 1:] = q[2:]
                
                # Store vc_goal_history
                vc_goal_history[i,:] = v_des
                
                # Store cc_goal history
                cc_goal_history = np.zeros((num_time_steps, 1))  # Prevent empty entries
                
                # construct action history
                tau = ctrl_array[i,:]
                action_history[i,:] = (tau + kd * v[6:])/kp + q[7:]
                
            return record_dir, state_history, base_history, vc_goal_history, cc_goal_history, action_history
    return record_dir, [], [], [], [], [], []



# Example usage
if __name__ == "__main__":
    record_dir, state_history, base_history,vc_goal_history,cc_goal_history, action_history = rollout_mpc(mode="close_loop", sim_time=5, robot_name="go2",
                                                      record_dir="./data/", v_des=[0.5, 0.1, 0.0],
                                                      save_data=True, interactive=False, record_video=False, visualize=True)
    print(f"Recorded data path: {record_dir}")

    # if time or q or v or ctrl:
    #     print("Recorded data:")
    #     print(f"Time: {time}")
    #     print(f"Q: {q}")
    #     print(f"V: {v}")
    #     print(f"Ctrl: {ctrl}")
    # vx_des_min,vx_des_max = 0.0,0.5
    # vy_des_min,vy_des_max = -0.1,0.1
    # w_des_min,w_des_max = 0.0,0.0
    # data_save_path = "./data"
    # for i in range(5):
    #     vx = random.uniform(vx_des_min, vx_des_max)
    #     vy = random.uniform(vy_des_min, vy_des_max)
    #     w = random.uniform(w_des_min, w_des_max)
    #     v_des = [vx, vy, w]
    
    #     # Rollout with MPC
    #     record_dir, time, q, v, ctrl = rollout_mpc(
    #         mode="close_loop",
    #         sim_time=5,
    #         robot_name="go2",
    #         record_dir=data_save_path + f"/iteration_{i+1}/",
    #         v_des=v_des,
    #         save_data=True,
    #         interactive=False,
    #         record_video=False,
    #         visualize=False
    #     )
