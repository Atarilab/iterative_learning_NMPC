## NOTE: This script is for data collection for behavior cloning.
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

from omegaconf import OmegaConf
import argparse
from typing import Tuple, List
import numpy as np
from datetime import datetime

from Behavior_Cloning.utils.Rollout_MPC import Rollout_MPC
from Behavior_Cloning.utils.database import Database
import random
import hydra
import h5py
import pickle
import scipy.spatial.transform as st
import pinocchio as pin
from mj_pin.utils import get_robot_description

def contact_vec_to_frame_names(contact_vec: np.ndarray) -> List[str]:
    frame_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    return [frame_names[i] for i in range(len(frame_names)) if contact_vec[i] == 1]

class DataCollection():
    def __init__(self, cfg):
        self.cfg = cfg
        
        # import global parameters
        self.sim_dt = cfg.SIM_DT
        self.n_state = cfg.n_state
        self.gaits = cfg.gaits
        self.feet_names = cfg.feet_names
        
        # for simulation
        self.episode_length = cfg.episode_length
        self.sim_time = cfg.sim_time
        self.sim_time_nominal = cfg.sim_time_nominal
        self.sim_time_perturbation = cfg.sim_time_perturbation
        
        # for perturbation
        self.replan_freq = cfg.replan_freq
        self.t0 = cfg.t0
        self.n_iteration = cfg.n_iteration
        self.num_pertubations_per_replanning = cfg.num_pertubations_per_replanning
        self.v_des = cfg.v_des
        
        self.force_start_offset = cfg.force_start_offset
        self.force_duration_range= cfg.force_duration_range
        self.force_magnitude_range= cfg.force_magnitude_range
        self.force_direction_range= cfg.force_direction_range
        
        # setup Rollout MPC to do both unperturbed and perturbed rollout job
        self.rollout_mpc_unperturbed = Rollout_MPC(self.cfg)
        self.rollout_mpc_force_perturbation = Rollout_MPC(self.cfg)
        
        # setup database for data collection
        self.save_data = cfg.save_data
        self.database = Database(limit=cfg.database_size, norm_input=True)
        # self.ood_database = Database(limit=cfg.database_size, norm_input=True)
        
        self.data_save_path = self._prepare_save_path()
        self.experiment_data_save_path = self._prepare_experiment_dir()
        # self.ood_data_save_path = self._prepare_ood_save_path()
        
        print("Data_collection config import successful!")
        print("Experiment data will be saved at path = ", self.experiment_data_save_path)
        print("Combined dataset will be saved at path = ", self.data_save_path)
        # print("OOD dataset will be saved at path = ",self.ood_data_save_path)

    def _prepare_save_path(self):
        current_time = datetime.now().strftime("%b_%d_%Y_%H_%M_%S")
        base_path = f"{self.cfg.data_save_path}/behavior_cloning/{'_'.join(self.gaits)}/"
        if self.cfg.suffix:
            base_path += f"_{self.cfg.suffix}/"
        save_path = os.path.join(base_path, current_time, "dataset")
        # print("Collected data will be saved in path = ", save_path)
        return save_path
    
    def _prepare_experiment_dir(self):
        # prepare saving path for inidividual rollouts        
        experiment_dir = os.path.join(self.data_save_path, "experiment")
        os.makedirs(experiment_dir, exist_ok=True)
        return experiment_dir
    
    def _prepare_ood_save_path(self):
        return None
    
    def dump_data_to_hdf5(self, iteration):
        # this function takes care of dumping data in a hdf5 dataset
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
    
    def save_ood_val_set_dummy(self, experiment_dir, states, vc_goals, cc_goals, actions,file_name):
        if "nominal" not in file_name.lower():
            self.ood_database.append(
                states=states[:1000],
                vc_goals=vc_goals[:1000],
                cc_goals=cc_goals[:1000],
                actions=actions[:1000]
            )

    def save_ood_val_set_l2_distance(self, experiment_dir, states, vc_goals, cc_goals, actions, times, file_name, distance_threshold=4.0):
        if "nominal" in file_name.lower():
            return

        # Load nominal trajectory
        nominal_file = [f for f in os.listdir(experiment_dir) if "nominal" in f.lower()][0]
        nominal_path = os.path.join(experiment_dir, nominal_file)
        nominal_data = np.load(nominal_path)

        # Build time-indexed nominal dictionary
        nominal_states_by_time = {
            round(t, 4): s for t, s in zip(nominal_data["time"], nominal_data["state"])
        }

        num_added = 0
        for idx in range(len(states)):
            t = round(times[idx], 4)
            s_pert = states[idx]

            if t in nominal_states_by_time:
                s_nom = nominal_states_by_time[t]

                s_pert_vec = s_pert[1:]
                s_nom_vec = s_nom[1:]
                dist = np.linalg.norm(s_pert_vec - s_nom_vec)

                if dist > distance_threshold:
                    self.ood_database.append(
                        states=[s_pert],
                        vc_goals=[vc_goals[idx]],
                        cc_goals=[cc_goals[idx]],
                        actions=[actions[idx]],
                    )
                    num_added += 1

        print(f"Added {num_added} OOD samples from {file_name}")

    def run_unperturbed(self,record_dir):
        # customize rollout parameters using reset()
        self.rollout_mpc_unperturbed.setup_nominal_rollout(
            sim_time = self.sim_time_nominal,
            record_dir = record_dir
        )
        # run
        early_termination, record_path = self.rollout_mpc_unperturbed.run()
        return early_termination, record_path
    
    def run_force_perturbed(self,
                            record_dir,
                            replan_instructions,
                            perturbation):    
        # reset parameter for a new rollout based on replanning and perturbation
        self.rollout_mpc_force_perturbation.setup_force_perturbation(
                                                record_dir,
                                                replan_instructions,
                                                perturbation,
                                                sim_time = self.sim_time_perturbation,
                                                )
        # run
        early_termination, record_path = self.rollout_mpc_force_perturbation.run()
        return early_termination, record_path
        
    def get_reference_state(self, record_path_nominal):
        # load nominal trajectory data and get replanning points
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
        phase_percentage = state[:,0]
        return phase_percentage,nominal_q,nominal_v

    def get_replanning_points(self):
        # sample replanning points
        replanning_points = []
        gait_period = 0.5
        num_replanning = int(gait_period * 1000 / self.replan_freq)
        start_timestep = self.t0 * 1000
        for i in range(num_replanning):
            next_replanning_point = int(i * self.replan_freq + start_timestep)
            replanning_points.append(next_replanning_point)
        print("Replanning points:", replanning_points)
        return replanning_points
    
    def get_force_struct(self):
        # Sample force start time
        force_start_time = self.force_start_offset

        # Sample force duration
        force_duration = np.random.uniform(
            self.force_duration_range[0],
            self.force_duration_range[1]
        )

        # Sample force direction
        force_direction = np.random.uniform(
            self.force_direction_range[0],
            self.force_direction_range[1],
            size=3
        )
        force_direction /= np.linalg.norm(force_direction) + 1e-6

        # Sample force magnitude
        magnitude = np.random.uniform(
            self.force_magnitude_range[0],
            self.force_magnitude_range[1]
        )

        # Construct full 6D force vector: (Fx, Fy, Fz, Tx, Ty, Tz)
        force_vec = np.concatenate([magnitude * force_direction, np.zeros(3)])

        print(f"🔴 Random push: start={force_start_time:.2f}s, duration={force_duration:.2f}s, vec={force_vec[:3]}")

        force_struct = {
            "start_time": force_start_time,
            "duration": force_duration,
            "force_vec": force_vec,
        }

        return force_struct

    def save_ood_dataset(self):
        pass
    
    def save_training_dataset(self,
                              experiment_dir):
        for file_name in os.listdir(experiment_dir):
            file_path = os.path.join(experiment_dir, file_name)
            if file_name.endswith(".npz") and os.path.isfile(file_path):
                print(f"Loading data from: {file_path}")
                data = np.load(file_path)
                states = data["state"]
                actions = data["action"]
                vc_goals = data["vc_goals"]
                cc_goals = data["cc_goals"]
                times = np.arange(len(states))
                is_nominal = "nominal" in file_name.lower()
                traj_id = 0 if is_nominal else 1
                traj_ids = [traj_id] * len(states)

                self.database.append(
                    states=states,
                    actions=actions,
                    vc_goals=vc_goals,
                    cc_goals=cc_goals,
                    traj_id=traj_ids,
                    times=times
                )
        self.dump_data_to_hdf5(iteration = 0)
        
    def run(self):
        # rollout reference trajectory
        _, record_dir_reference_traj = self.run_unperturbed(record_dir=self.experiment_data_save_path)
        
        # get initial condition from reference traj
        reference_phase_percentage, nominal_q, nominal_v= self.get_reference_state(record_dir_reference_traj)
        
        # sample replanning points
        replanning_points = self.get_replanning_points()
        
        # loop over replanning points and do perturbations
        for i_replanning in replanning_points:
            print(f"Replanning at step {i_replanning}")
            current_phase_percentage = reference_phase_percentage[i_replanning]
            q0 = nominal_q[i_replanning]
            # manually put the robot in the initial position, otherwise it is going to be a MPC tracking reference problem
            q0[0] = 0
            v0 = nominal_v[i_replanning]
            # current_contact_vec = contact_vec[i_replanning]
            # ee_in_contact = contact_vec_to_frame_names(current_contact_vec)
            
            for j in range(self.num_pertubations_per_replanning):
                # pack replan_instructions
                current_time = np.round(i_replanning * self.sim_dt, 4)
                replan_instructions = {
                        "current_time":current_time,
                        "current_phase_percentage":current_phase_percentage,
                        "q0":q0,
                        "v0":v0,
                        "replanning_point":i_replanning,
                        "nth_traj_per_replanning":j+1,
                        "nominal_flag": False,                        
                }
                
                early_termination = False
                # main perturbation loop: loop until there is no early_termination
                while True:
                    # pack force struct
                    force_struct = self.get_force_struct()
                    early_termination, record_path = self.run_force_perturbed(
                        record_dir = self.experiment_data_save_path,
                        replan_instructions = replan_instructions,
                        perturbation = force_struct
                    )
                    if not early_termination:
                        break

        self.save_training_dataset(self.experiment_data_save_path)

@hydra.main(config_path='../examples/cfgs/', config_name='bc_experimental.yaml', version_base="1.1")
def main(cfg):
    dc = DataCollection(cfg)
    # replanning_points = dc.get_replanning_points()
    # force_struct = dc.get_force_struct()
    # print("replanning points = ", replanning_points)
    # print("force_struct = ", force_struct)
    
    # _,dir = dc.run_unperturbed(dc.experiment_data_save_path)
    # print(dir)

    dc.run()
    
if __name__ == '__main__':
    main()