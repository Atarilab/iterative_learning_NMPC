## NOTE: This script is for data collection for safedagger.
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../..')

from omegaconf import OmegaConf
import argparse
from typing import Tuple, List
import numpy as np
from datetime import datetime

from DAgger.utils.Rollout_combined_controller import rollout_combined_controller
from DAgger.utils.database import Database
import random
import hydra
import h5py
import pickle
import scipy.spatial.transform as st
import pinocchio as pin
from mj_pin.utils import get_robot_description

class DataCollection():
    def __init__(self, 
                 cfg,
                 iter_index: int = 0,
                 goal_index: int = 0,
                 previous_dataset_path: str = None,
                 previous_policy_path: str = None,
                 goal: List[float] = [0.0, 0.0, 0.0]):
        # fixed parameters
        self.cfg = cfg
        self.episode_length = cfg.episode_length
        self.sim_dt = cfg.sim_dt
        self.sim_time = cfg.sim_time
        self.start_time = cfg.start_time
        self.visualize = cfg.visualize
        self.save_data = cfg.save_data
        self.record_video = cfg.record_video
        self.interactive = cfg.interactive
        self.v_des = goal
        self.initial_control_mode = cfg.initial_control_mode
        
        self.robot_name = cfg.robot_name
        self.gaits = cfg.gaits
        self.feet_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        
        self.vx_range = (cfg.vx_des_min, cfg.vx_des_max)
        self.vy_range = (cfg.vy_des_min, cfg.vy_des_max)
        self.w_range = (cfg.w_des_min, cfg.w_des_max)
        
        # parameters from higher layer 
        self.iter_index = iter_index
        self.goal_index = goal_index
        self.previous_dataset_path = previous_dataset_path
        self.previous_policy_path = previous_policy_path
        
        # internal parameters
        self.database = Database(limit=cfg.database_size, norm_input=True)
        self.data_save_path = self._prepare_save_path()
        self.ood_database = Database(limit=cfg.database_size, norm_input=True)

    def _prepare_save_path(self):
        """
        Constructs the dataset save path dynamically using `run_dir` and `iter_index`.
        """
        base_dir = self.cfg.run_dir
        return os.path.join(base_dir, f"goal_{self.goal_index}",f"iter_{self.iter_index}", "dataset")

    def save_dataset(self, iteration):
        os.makedirs(self.data_save_path, exist_ok=True)
        data_len = len(self.database)
        with h5py.File(f"{self.data_save_path}/database_{iteration}.hdf5", 'w') as hf:
            hf.create_dataset('states', data=self.database.states[:data_len])
            hf.create_dataset('vc_goals', data=self.database.vc_goals[:data_len])
            hf.create_dataset('cc_goals', data=self.database.cc_goals[:data_len])
            hf.create_dataset('actions', data=self.database.actions[:data_len]) 
            
            if self.database.traj_ids[0] is not None:
                traj_ids_array = np.array(self.database.traj_ids[:data_len], dtype='S')
                hf.create_dataset('traj_ids', data=traj_ids_array)

            if self.database.traj_times[0] is not None:
                traj_times_array = np.array(self.database.traj_times[:data_len])
                hf.create_dataset('traj_times', data=traj_times_array)

        config_path = f"{self.data_save_path}/config.pkl"
        if not os.path.exists(config_path):
            with open(config_path, "wb") as f:
                pickle.dump(self.cfg, f)
        print(f"Dataset saved at iteration {iteration+1}")
    
    def append_to_dataset(self, base_dataset_path: str, output_path: str = None):
        print("running append_to_dataset")
        # input()
        assert os.path.exists(base_dataset_path), f"Base dataset file not found: {base_dataset_path}"
        
        with h5py.File(base_dataset_path, 'r') as f:
            base_states = f['states'][:]
            base_vc_goals = f['vc_goals'][:]
            base_cc_goals = f['cc_goals'][:]
            base_actions = f['actions'][:]
            base_traj_ids = f['traj_ids'][:] if 'traj_ids' in f else None
            base_traj_times = f['traj_times'][:] if 'traj_times' in f else None

        new_len = len(self.database)
        new_states = self.database.states[:new_len]
        new_vc_goals = self.database.vc_goals[:new_len]
        new_cc_goals = self.database.cc_goals[:new_len]
        new_actions = self.database.actions[:new_len]
        new_traj_ids = np.array(self.database.traj_ids[:new_len], dtype='S') if self.database.traj_ids[0] is not None else None
        new_traj_times = np.array(self.database.traj_times[:new_len]) if self.database.traj_times[0] is not None else None

        # Concatenate
        agg_states = np.concatenate([base_states, new_states], axis=0)
        agg_vc_goals = np.concatenate([base_vc_goals, new_vc_goals], axis=0)
        agg_cc_goals = np.concatenate([base_cc_goals, new_cc_goals], axis=0)
        agg_actions = np.concatenate([base_actions, new_actions], axis=0)
        agg_traj_ids = np.concatenate([base_traj_ids, new_traj_ids], axis=0) if base_traj_ids is not None and new_traj_ids is not None else None
        agg_traj_times = np.concatenate([base_traj_times, new_traj_times], axis=0) if base_traj_times is not None and new_traj_times is not None else None

        # Save result
        output_path = output_path or base_dataset_path  # overwrite by default
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('states', data=agg_states)
            f.create_dataset('vc_goals', data=agg_vc_goals)
            f.create_dataset('cc_goals', data=agg_cc_goals)
            f.create_dataset('actions', data=agg_actions)
            if agg_traj_ids is not None:
                f.create_dataset('traj_ids', data=agg_traj_ids)
            if agg_traj_times is not None:
                f.create_dataset('traj_times', data=agg_traj_times)

    def run(self):
        experiment_dir = os.path.join(self.data_save_path, "experiment")
        os.makedirs(experiment_dir, exist_ok=True)
        # for calculation of expert influence
        total_timesteps = 0
        expert_timesteps = 0
        
        # policy initialization
        policy_path = self.previous_policy_path
        reference_mpc_path = self.cfg.reference_mpc_path
        data = np.load(reference_mpc_path)
        initial_state = []
        q0 = data["q"][int(self.start_time * 1000)]
        v0 = data["v"][int(self.start_time * 1000)]
        initial_state = [q0, v0]
        
        # call rollout function
        rollout_combined_controller(
            # initialization
            robot_name=self.robot_name,
            sim_time=self.sim_time,
            start_time=self.start_time,
            policy_path=policy_path,
            reference_mpc_path=reference_mpc_path,
            initial_state=initial_state,
            control_mode=self.initial_control_mode,
            
            # goal
            v_des=self.v_des,
            
            # simulation control
            record_video=self.record_video,
            visualize=self.visualize,
            save_data=self.save_data,
            record_dir=experiment_dir,
            interactive=self.interactive,    
        )

        for file_name in os.listdir(experiment_dir):
            file_path = os.path.join(experiment_dir, file_name)
            if file_name.endswith(".npz") and os.path.isfile(file_path):
                print(f"Loading data from: {file_path}")
                data = np.load(file_path)

                # get expert mask
                if "is_expert" not in data:
                    print(f"Skipping {file_path}: 'is_expert' key missing.")
                    continue

                is_expert_mask = data["is_expert"].astype(bool)
                total_timesteps += len(is_expert_mask)
                expert_timesteps += np.sum(is_expert_mask)
                
                if not np.any(is_expert_mask):
                    print(f"No expert data in {file_path}, skipping.")
                    continue

                # filter data per timestep
                states = data["state"][is_expert_mask]
                actions = data["action"][is_expert_mask]
                vc_goals = data["vc_goals"][is_expert_mask]
                cc_goals = data["cc_goals"][is_expert_mask]
                times = np.arange(len(data["state"]))[is_expert_mask]

                traj_ids = [1] * len(states)  # expert-labeled

                self.database.append(
                    states=states,
                    actions=actions,
                    vc_goals=vc_goals,
                    cc_goals=cc_goals,
                    traj_id=traj_ids,
                    times=times
                )
        # save expert dataset as database_0.hdf5      
        self.save_dataset(iteration=0)
        
        ## Perform aggregation
        # Output path for the aggregated dataset
        agg_dataset_path = os.path.join(self.data_save_path, "agg_dataset.hdf5")
        
        self.append_to_dataset(
            base_dataset_path=self.previous_dataset_path,
            output_path=agg_dataset_path
        )
        
        if total_timesteps > 0:
            ratio = expert_timesteps / total_timesteps
            print(f"📊 Expert Influence Ratio: {expert_timesteps}/{total_timesteps} = {ratio:.3f}")
        else:
            print("No timesteps processed; cannot compute expert influence.")
        # input()

@hydra.main(config_path='../cfgs', config_name='iter_locosafedagger.yaml', version_base="1.1")
def main(cfg):
    if cfg.policy_path is None:
        cfg.policy_path = cfg.initial_policy_path
    dc = DataCollection(cfg)
    dc.run()

if __name__ == "__main__":
    main()