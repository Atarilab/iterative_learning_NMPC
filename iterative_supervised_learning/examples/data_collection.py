## NOTE: This script is for data collection for behavior cloning.
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../..')

import pinocchio as pin
import argparse
from typing import List, Tuple
import numpy as np

from mj_pin.abstract import VisualCallback, DataRecorder  # type: ignore
from mj_pin.simulator import Simulator  # type: ignore
from mj_pin.utils import get_robot_description  # type: ignore
from mpc_controller.mpc import LocomotionMPC

from datetime import datetime
from iterative_supervised_learning.utils.RolloutMPC import RolloutMPC
from iterative_supervised_learning.utils.database import Database

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

import random
import hydra
import pickle
import h5py

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
            return record_dir, data["time"].tolist(), data["q"].tolist(), data["v"].tolist(), data["ctrl"].tolist()

    return record_dir, [], [], [], []

class DataCollection():

    def __init__(self, cfg):        
        # configuration file (containing the hyper/parameters)
        self.cfg = cfg
            
        # Simulation rollout properties
        self.episode_length = cfg.episode_length
        self.sim_dt = cfg.sim_dt
        
        # MPC rollout pertubations
        self.mu_base_pos, self.sigma_base_pos = cfg.mu_base_pos, cfg.sigma_base_pos # base position
        self.mu_joint_pos, self.sigma_joint_pos = cfg.mu_joint_pos, cfg.sigma_joint_pos # joint position
        self.mu_base_ori, self.sigma_base_ori = cfg.mu_base_ori, cfg.sigma_base_ori # base orientation
        self.mu_vel, self.sigma_vel = cfg.mu_vel, cfg.sigma_vel # joint velocity
        
        # Model Parameters
        self.action_type = cfg.action_type
        self.normalize_policy_input = cfg.normalize_policy_input
        
        # Iterations
        self.n_iteration = cfg.n_iteration
        self.num_pertubations_per_replanning = cfg.num_pertubations_per_replanning
        
        print('number of iterations: ' + str(self.n_iteration))
        print('number of pertubations per positon: ' + str(self.num_pertubations_per_replanning))
        max_dataset_size = self.n_iteration * 10 * self.num_pertubations_per_replanning * self.episode_length
        print('estimated dataset size: ' + str(max_dataset_size))
        
        # Desired Motion Parameters
        self.gaits = cfg.gaits
        self.vx_des_min, self.vx_des_max = cfg.vx_des_min, cfg.vx_des_max
        self.vy_des_min, self.vy_des_max = cfg.vy_des_min, cfg.vy_des_max
        self.w_des_min, self.w_des_max = cfg.w_des_min, cfg.w_des_max
        
        # define log file name
        str_gaits = ''
        for gait in self.gaits:
            str_gaits = str_gaits + gait
        self.str_gaits = str_gaits
        
        current_date = datetime.today().strftime("%b_%d_%Y_")
        current_time = datetime.now().strftime("%H_%M_%S")

        save_path_base = "/behavior_cloning/" + str_gaits
        if cfg.suffix != '':
            save_path_base += "_"+cfg.suffix
        save_path_base += "/" +  current_date + current_time
        
        self.data_save_path = self.cfg.data_save_path + save_path_base
        self.dataset_savepath = self.data_save_path + '/dataset'
        
        # Declare Database
        self.database = Database(limit=cfg.database_size)
    
    def save_dataset(self, iter):
        """save the current database as h5py file and the config as pkl

        Args:
            iter (int): the current iteration
        """
        print("saving dataset for iteration " + str(iter))
        
        # make directory
        os.makedirs(self.dataset_savepath, exist_ok=True)
        
        # Get data len from dataset class
        data_len = len(self.database)
        
        # save numpy datasets
        with h5py.File(self.dataset_savepath + "/database_" + str(iter) + ".hdf5", 'w') as hf:
            hf.create_dataset('states', data=self.database.states[:data_len])
            hf.create_dataset('vc_goals', data=self.database.vc_goals[:data_len])
            hf.create_dataset('cc_goals', data=self.database.cc_goals[:data_len])
            hf.create_dataset('actions', data=self.database.actions[:data_len]) 
        
        # save config as pickle only once
        if os.path.exists(self.dataset_savepath + "/config.pkl") == False:
            # convert hydra cfg to config dict
            config_dict = OmegaConf.to_container(self.cfg, resolve=True)
            
            f = open(self.dataset_savepath + "/config.pkl", "wb")
            pickle.dump(config_dict, f)
            f.close()
        
        print("saved dataset at iteration " + str(iter))
        
    def run(self):
        for iteration in range(self.n_iteration):
            print(f"Starting iteration {iteration+1}/{self.n_iteration}")
            for _ in range(self.num_pertubations_per_replanning):
                # Randomly sample a velocity goal
                vx = random.uniform(self.vx_des_min, self.vx_des_max)
                vy = random.uniform(self.vy_des_min, self.vy_des_max)
                w = random.uniform(self.w_des_min, self.w_des_max)
                v_des = [vx, vy, w]

                # Rollout with MPC
                record_dir, time, q, v, ctrl = rollout_mpc(
                    mode="close_loop",
                    sim_time=self.episode_length,
                    robot_name=self.cfg.robot_name,
                    record_dir=self.data_save_path + f"/iteration_{iteration+1}/",
                    v_des=v_des,
                    save_data=True,
                    interactive=False,
                    record_video=False,
                    visualize=False
                )

                # Add data to the database
                for t, state, vel, ctrl_input in zip(time, q, v, ctrl):
                    self.database.add(state=state, velocity=vel, action=ctrl_input, goal=v_des)

            # Save dataset at the end of each iteration
            self.save_dataset(iteration+1)
            print(f"Completed iteration {iteration+1}/{self.n_iteration}")

@hydra.main(config_path='cfgs', config_name='data_collection_config')
def main(cfg):
    dc = DataCollection(cfg)
    dc.run()

if __name__ == '__main__':
    main()