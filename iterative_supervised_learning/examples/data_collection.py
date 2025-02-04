## NOTE: This script is for data collection for behavior cloning.
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../..')

from omegaconf import OmegaConf
import argparse
from typing import Tuple, List
import os
import numpy as np
from datetime import datetime
from iterative_supervised_learning.utils.RolloutMPC import RolloutMPC
from iterative_supervised_learning.utils.database import Database
import random
import hydra
import h5py
import pickle

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
            
            # form state history
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

                
            return record_dir, state_history, base_history, vc_goal_history, cc_goal_history, ctrl_array
    return record_dir, [], [], [], [], [], []

class DataCollection:
    def __init__(self, cfg):
        self.cfg = cfg
        self.episode_length = cfg.episode_length
        self.sim_dt = cfg.sim_dt
        self.n_iteration = cfg.n_iteration
        # self.num_perturbations = cfg.num_perturbations_per_replanning
        self.gaits = cfg.gaits
        self.vx_range = (cfg.vx_des_min, cfg.vx_des_max)
        self.vy_range = (cfg.vy_des_min, cfg.vy_des_max)
        self.w_range = (cfg.w_des_min, cfg.w_des_max)
        self.database = Database(limit=cfg.database_size)
        self.data_save_path = self._prepare_save_path()
    
    def _prepare_save_path(self):
        current_time = datetime.now().strftime("%b_%d_%Y_%H_%M_%S")
        base_path = f"{self.cfg.data_save_path}/behavior_cloning/{'_'.join(self.gaits)}/"
        if self.cfg.suffix:
            base_path += f"_{self.cfg.suffix}/"
        return os.path.join(base_path, current_time, "dataset")
    
    def save_dataset(self, iteration):
        os.makedirs(self.data_save_path, exist_ok=True)
        
        data_len = len(self.database)
        with h5py.File(f"{self.data_save_path}/database_{iteration}.hdf5", 'w') as hf:
            # print(f"Final states shape: {self.database.states.shape}, Type: {type(self.database.states)}")
            hf.create_dataset('states', data=self.database.states[:data_len])
            hf.create_dataset('vc_goals', data=self.database.vc_goals[:data_len])
            hf.create_dataset('cc_goals', data=self.database.cc_goals[:data_len])
            hf.create_dataset('actions', data=self.database.actions[:data_len]) 
            
        config_path = f"{self.data_save_path}/config.pkl"
        if not os.path.exists(config_path):
            with open(config_path, "wb") as f:
                pickle.dump(self.cfg, f)
        print(f"Dataset saved at iteration {iteration}")
    
    def run(self):
        for i in range(self.n_iteration):
            gait = random.choice(self.gaits)
            # TODO: grid sample velocity goal
            v_des = [random.uniform(*self.vx_range), random.uniform(*self.vy_range), random.uniform(*self.w_range)]
            
            record_dir = f"{self.data_save_path}/iteration_{i}/"
            os.makedirs(record_dir, exist_ok=True)
            
            _, state_history, base_history, vc_goal_history, cc_goal_history, ctrl = rollout_mpc(
                mode="close_loop",
                sim_time=self.episode_length * self.sim_dt,
                robot_name=self.cfg.robot_name,
                record_dir=record_dir,
                v_des=v_des,
                save_data=True,
                visualize=False
            )
            print(f"Shape of states: {state_history.shape}, Type: {type(state_history)}")
            if len(state_history) != 0:
                self.database.append(states=state_history, vc_goals=vc_goal_history, cc_goals=cc_goal_history, actions=ctrl)
                print("MPC data saved into database")
                print('database size: ' + str(len(self.database)))  
            else:
                print('mpc rollout failed')
            
            self.save_dataset(i)

# Example usage without database
# if __name__ == "__main__":
#     # Define goal space
#     vx_des_min, vx_des_max = 0.0, 0.5
#     vy_des_min, vy_des_max = -0.1, 0.1
#     w_des_min, w_des_max = 0.0, 0.0
#     data_save_path = "./data"
#     num_goals_each_dim = 4

#     # Generate grid sampled goals
#     vx_values = np.linspace(vx_des_min, vx_des_max, num_goals_each_dim)
#     vy_values = np.linspace(vy_des_min, vy_des_max, num_goals_each_dim)
#     w_values = np.linspace(w_des_min, w_des_max, 1)  # Single value for w since min == max

#     # Initialize dataset storage
#     collected_data = {
#         "time": [],
#         "q": [],
#         "v": [],
#         "ctrl": []
#     }

#     # Rollout MPC on the selected goals
#     for i, vx in enumerate(vx_values):
#         for j, vy in enumerate(vy_values):
#             for k, w in enumerate(w_values):
#                 v_des = [vx, vy, w]
                
#                 # Rollout with MPC
#                 record_dir, time, q, v, ctrl = rollout_mpc(
#                     mode="close_loop",
#                     sim_time=5,
#                     robot_name="go2",
#                     record_dir=f"{data_save_path}/iteration_{i}_{j}_{k}/",
#                     v_des=v_des,
#                     save_data=True,
#                     interactive=False,
#                     record_video=False,
#                     visualize=False
#                 )
                
#                 # Append collected data
#                 collected_data["time"].extend(time)
#                 collected_data["q"].extend(q)
#                 collected_data["v"].extend(v)
#                 collected_data["ctrl"].extend(ctrl)

#     # Save combined dataset
#     np.savez(os.path.join(data_save_path, "collected_data.npz"), 
#              time=collected_data["time"],
#              q=collected_data["q"],
#              v=collected_data["v"],
#              ctrl=collected_data["ctrl"])
    
#     print(f"Collected dataset saved at {data_save_path}/collected_data.npz")


# Example usage with database
@hydra.main(config_path='cfgs', config_name='data_collection_config.yaml')
def main(cfg):
    dc = DataCollection(cfg)
    dc.run()

if __name__ == '__main__':
    main()

