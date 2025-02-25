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

# from iterative_supervised_learning.utils.RolloutMPC import RolloutMPC
from iterative_supervised_learning.utils.RolloutMPC_rewrite import rollout_mpc
from iterative_supervised_learning.utils.database import Database
import random
import hydra
import h5py
import pickle
import scipy.spatial.transform as st
import pinocchio as pin
from mj_pin.utils import get_robot_description

class DataCollection():
    def __init__(self, cfg):
        self.cfg = cfg
        self.episode_length = cfg.episode_length
        self.sim_dt = cfg.sim_dt
        self.n_iteration = cfg.n_iteration
        self.num_pertubations_per_replanning = cfg.num_pertubations_per_replanning
        
        self.gaits = cfg.gaits
        self.vx_range = (cfg.vx_des_min, cfg.vx_des_max)
        self.vy_range = (cfg.vy_des_min, cfg.vy_des_max)
        self.w_range = (cfg.w_des_min, cfg.w_des_max)
        self.database = Database(limit=cfg.database_size,norm_input=False)
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
        nq = 19
        nv = 17
        replan_freq = 2000
        n_state = 44
        
        # rollout nominal trajectory
        _, record_path_nominal = rollout_mpc(show_plot=False,
                                         v_des = [0.3,0.0,0.0],
                                         sim_time=4.0)
        
        # calculate replanning points
        replanning_points = np.arange(0, self.episode_length, replan_freq)
        print("Replanning points:", replanning_points)
        
        # extract nominal state on replanning points
        data = np.load(record_path_nominal)
        state = data["state"]
        # print(state[:2])
        # input()
        
        phase_percentage = state[:,0]
        nominal_v = data["v"]
        nominal_q = data["q"]
        # print("shape of v is = ",np.shape(nominal_v))
        # print("shape of q is = ",np.shape(nominal_q))
        # print("shape of phase_percentage is = ", np.shape(phase_percentage) )
        # input()
        vc_goals = data["vc_goals"][0]
        cc_goals = None
        actions = data["ctrl"]
        
        # rollout MPC at each replanning point
        for i_replanning in replanning_points:
            print(f"Replanning at step {i_replanning}")
            q0 = nominal_q[i_replanning]
            v0 = nominal_v[i_replanning]
            for j in range(self.num_pertubations_per_replanning):
                # randomize on given state and pass to mpc simulator
                randomize_on_given_state = np.concatenate((q0, v0, np.array([phase_percentage[i_replanning]])))
                
                early_termination = False
                # run MPC from replanning state until the simulation finishes
                while True:
                    early_termination, record_path_replanning = rollout_mpc(randomize_on_given_state=randomize_on_given_state, 
                                                                            v_des=[0.3,0.0,0.0],
                                                                            sim_time=4.0,
                                                                            show_plot=False)
                    if not early_termination:
                        break
                    
                
# Example usage with database
@hydra.main(config_path='cfgs', config_name='data_collection_config.yaml',version_base="1.1")
def main(cfg):
    dc = DataCollection(cfg)
    dc.run()

if __name__ == '__main__':
    main()

