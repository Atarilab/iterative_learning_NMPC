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
from iterative_supervised_learning.utils.RolloutMPC import rollout_mpc
from iterative_supervised_learning.utils.database import Database
import random
import hydra
import h5py
import pickle


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
        print(f"Dataset saved at iteration {iteration+1}")
    
    def run(self):
        for i in range(self.n_iteration):
            gait = random.choice(self.gaits)
            
            # v_des = [random.uniform(*self.vx_range), random.uniform(*self.vy_range), random.uniform(*self.w_range)]
            
            # TODO: grid sample velocity goal
            
            # TODO: sample disturbed goals around default goals
            # Default goal velocity
            v_des_default = np.array([0.5, 0.1, 0])

            # Define disturbance levels for each entry
            noise_std = np.array([0.02, 0.01, 0.0])  # No noise for the third entry

            # Sample disturbed goals
            v_des = v_des_default + np.random.normal(0, noise_std, size=3)

            print("Sampled goal is:", v_des)

            # input()

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
@hydra.main(config_path='cfgs', config_name='data_collection_config.yaml',version_base="1.1")
def main(cfg):
    dc = DataCollection(cfg)
    dc.run()

if __name__ == '__main__':
    main()

