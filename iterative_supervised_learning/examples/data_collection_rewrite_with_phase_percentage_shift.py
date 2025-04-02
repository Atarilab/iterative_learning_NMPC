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

from iterative_supervised_learning.utils.RolloutMPC_shift_phase_percentage import rollout_mpc_phase_percentage_shift
from iterative_supervised_learning.utils.database import Database
import random
import hydra
import h5py
import pickle
import scipy.spatial.transform as st
import pinocchio as pin
from mj_pin.utils import get_robot_description

SIM_DT = 0.001
nq = 19
nv = 17
replan_freq = 50
t0 = 0.0
v_des = [0.15,0.0,0.0]

# with base_wrt_feet
n_state = 44

def contact_vec_to_frame_names(contact_vec: np.ndarray) -> List[str]:
    frame_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    return [frame_names[i] for i in range(len(frame_names)) if contact_vec[i] == 1]

class DataCollection():
    def __init__(self, cfg):
        # initialize parameters from configuration
        self.cfg = cfg
        self.episode_length = cfg.episode_length
        self.sim_dt = cfg.sim_dt
        self.n_iteration = cfg.n_iteration
        self.num_pertubations_per_replanning = cfg.num_pertubations_per_replanning
        
        self.gaits = cfg.gaits
        self.feet_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        
        # currently v_des = [vx,vy,w] is a given number [0.3,0.0,0.0]
        self.vx_range = (cfg.vx_des_min, cfg.vx_des_max)
        self.vy_range = (cfg.vy_des_min, cfg.vy_des_max)
        self.w_range = (cfg.w_des_min, cfg.w_des_max)
        
        # initialize database
        self.database = Database(limit=cfg.database_size,norm_input=True)
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
        # Use the prepared dataset path for saving experiments
        experiment_dir = os.path.join(self.data_save_path, "experiment")
        os.makedirs(experiment_dir, exist_ok=True)
        
        # rollout nominal trajectory
        _, record_path_nominal = rollout_mpc_phase_percentage_shift(show_plot=False,
                                        visualize= True,
                                        record_video = False,
                                        v_des = v_des,
                                        sim_time=5.0,
                                        save_data=True,
                                        record_dir=experiment_dir,
                                        nominal_flag = True)
        
        # calculate replanning points
        # sample replanning points in one gait cycle
        replanning_points = []
        gait_period = 0.5
        num_replanning = int(gait_period*1000/replan_freq)
        start_timestep = t0*1000
        for i in range(num_replanning):
            next_replanning_point = int(i*replan_freq + start_timestep)
            replanning_points.append(next_replanning_point)
        print("Replanning points:", replanning_points)
        # input()
        
        # # sample replanning points in n gait cycles
        # replanning_points = []
        # gait_period = 0.5
        # n_gait_cycles = 4
        # num_replanning = int(n_gait_cycles * gait_period * 1000 / replan_freq)
        # start_timestep = int(t0 * 1000)
        # for i in range(num_replanning):
        #     next_replanning_point = int(i * replan_freq + start_timestep)
        #     replanning_points.append(next_replanning_point)
        # print("Replanning points:", replanning_points)
        # # input() 
        
        # extract nominal state on replanning points
        print("loading nominal traj data from path = ")
        print(record_path_nominal)
        data = np.load(record_path_nominal)
        state = data["state"]
        feet_pos = data["feet_pos_w"]
        
        nominal_v = data["v"]
        nominal_q = data["q"]
        
        vc_goals = data["vc_goals"][0]
        cc_goals = None
        actions = data["ctrl"]
        contact_vec = data["contact_vec"]
        # input()
        
        # rollout MPC at each replanning point
        for i_replanning in replanning_points:
            print(f"Replanning at step {i_replanning}")
            
            q0 = nominal_q[i_replanning]
            v0 = nominal_v[i_replanning]
            
            # # find ee_in_contact with feet position threshold
            # feet_pos_all = feet_pos[i_replanning]
            # print("four feet positions = ")
            # print(feet_pos_all)
            # ee_in_contact = []
            # for i,f_name in enumerate(self.feet_names):
            #     feet_pos_current = feet_pos_all[3*i:3*i+3]
            #     print("current feet name is = ", f_name)
            #     print(feet_pos_current)
            #     if feet_pos_current[-1] <= 0.005:
            #         ee_in_contact.append(f_name)
            
            # find ee_in_contact from recorded contact vec
            current_contact_vec = contact_vec[i_replanning]
            ee_in_contact = contact_vec_to_frame_names(current_contact_vec)
            
            print("print out replanning points")
            print("nominal q0 is = ", q0)
            print("nominal v0 is = ", v0)
            print("current ee in contact FROM SIMULATOR is = ", ee_in_contact)
            # input()
            for j in range(self.num_pertubations_per_replanning):
                # continue
                # NOTE: use nullspace randomization
                phase_percentage = state[:,0]
                print("current replanning phase percentage is = ", phase_percentage[i_replanning])
                # print(phase_percentage[i_replanning])
                # input()
            
                # randomize on given state and pass to mpc simulator
                randomize_on_given_state = np.concatenate((q0, v0, np.array([phase_percentage[i_replanning]])))
                current_time = np.round(i_replanning * SIM_DT,4)
                print("current time is  = ", current_time)
                # input()
                
                early_termination = False
                # run MPC from replanning state until the simulation finishes
                while True:
                    early_termination, record_path_replanning = rollout_mpc_phase_percentage_shift(randomize_on_given_state=randomize_on_given_state, 
                                                                            v_des=v_des,
                                                                            sim_time=5.0,
                                                                            current_time = current_time,
                                                                            show_plot = False,
                                                                            visualize = False,
                                                                            record_video = True,
                                                                            save_data = True,
                                                                            record_dir = experiment_dir,
                                                                            ee_in_contact = ee_in_contact,
                                                                            nominal_flag=False,
                                                                            replanning_point=i_replanning,
                                                                            nth_traj_per_replanning=j+1)
                    if not early_termination:
                        break
                    
        # SAVE DATABASE
        for file_name in os.listdir(experiment_dir):
            file_path = os.path.join(experiment_dir, file_name)
            if file_name.endswith(".npz") and os.path.isfile(file_path):
                print(f"Loading data from: {file_path}")
            data = np.load(file_path)
            self.database.append(
                states = data["state"],
                vc_goals = data["vc_goals"],
                cc_goals = data["cc_goals"],
                actions = data["action"]
            )
        
        self.save_dataset(iteration=0)
                
# Example usage with database
@hydra.main(config_path='cfgs', config_name='data_collection_config.yaml',version_base="1.1")
def main(cfg):
    dc = DataCollection(cfg)
    dc.run()

if __name__ == '__main__':
    main()

