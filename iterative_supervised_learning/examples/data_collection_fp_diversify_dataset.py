## NOTE: This script is for data collection for behavior cloning.
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../..')

from omegaconf import OmegaConf
import argparse
from typing import Tuple, List
import numpy as np
from datetime import datetime

from iterative_supervised_learning.utils.RolloutMPC_force_at_interval import rollout_mpc_phase_percentage_shift
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
# v_des = [0.15, 0.0, 0.0]
n_state = 44

def contact_vec_to_frame_names(contact_vec: np.ndarray) -> List[str]:
    frame_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    return [frame_names[i] for i in range(len(frame_names)) if contact_vec[i] == 1]

def generate_force_perturbation_schedule_every_n_steps(
    start_step: int,
    end_step: int,
    every_n_timesteps: int,
    sim_dt: float
):
    """
    Generate force perturbation schedule based on simulation steps.

    Args:
        start_step (int): Step at which perturbations begin.
        end_step (int): Step at which perturbations stop.
        every_n_timesteps (int): Apply a perturbation every N timesteps.
        sim_dt (float): Simulator timestep (used to convert steps to time).

    Returns:
        force_start_times (List[float])
        force_durations (List[float])
        force_vecs (List[np.ndarray])
    """
    force_start_times = []
    force_durations = []
    force_vecs = []

    for step in range(start_step, end_step, every_n_timesteps):
        direction = np.random.choice(["x+", "x-","y+","y-"])
        sign = np.random.choice([-1.0, 1.0])

        if direction == "x+":
            magnitude = np.random.uniform(45.0, 55.0)
            force = np.array([
                magnitude, 0.0, 0.0,
                0.0, 0.0, 0.0
            ])
        if direction == "x-":
            magnitude = np.random.uniform(20.0, 35.0)
            force = np.array([
                -magnitude, 0.0, 0.0,
                0.0, 0.0, 0.0
            ])
        if direction == "y-" or direction == "y+":
            magnitude = np.random.uniform(25.0, 35.0) * sign
            force = np.array([
                0.0, magnitude, 0.0,
                0.0, 0.0, 0.0
            ])

        duration = np.random.uniform(0.2, 0.4)  # seconds
        start_time = step * sim_dt

        force_start_times.append(start_time)
        force_durations.append(duration)
        force_vecs.append(force)

    return force_start_times, force_durations, force_vecs


class DataCollection():
    def __init__(self, cfg):
        self.cfg = cfg
        self.episode_length = cfg.episode_length
        self.sim_dt = cfg.sim_dt
        self.n_iteration = cfg.n_iteration
        self.num_pertubations_per_replanning = cfg.num_pertubations_per_replanning
        
        self.gaits = cfg.gaits
        self.feet_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        
        self.vx_range = (cfg.vx_des_min, cfg.vx_des_max)
        self.vy_range = (cfg.vy_des_min, cfg.vy_des_max)
        self.w_range = (cfg.w_des_min, cfg.w_des_max)
        
        self.database = Database(limit=cfg.database_size, norm_input=True)
        self.data_save_path = self._prepare_save_path()
        self.ood_database = Database(limit=cfg.database_size, norm_input=True)

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
        ood_range = 500
        if "nominal" not in file_name.lower():
            self.ood_database.append(
                states=states[:ood_range],
                vc_goals=vc_goals[:ood_range],
                cc_goals=cc_goals[:ood_range],
                actions=actions[:ood_range]
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

    

    def run(self):        
        experiment_dir = os.path.join(self.data_save_path, "experiment")
        os.makedirs(experiment_dir, exist_ok=True)

        goal_list = [0.0, 0.15, 0.3]
        # for i in range(3):
        #     v_des = [goal_list[i], 0.0, 0.0]
        #     _, record_path_nominal = rollout_mpc_phase_percentage_shift(
        #         show_plot=False,
        #         visualize=True,
        #         record_video=False,
        #         v_des=v_des,
        #         sim_time=10.0,
        #         save_data=True,
        #         record_dir=experiment_dir,
        #         nominal_flag=True
        #     )
            
        # force_start_times = [3.0, 6.0, 9.0]
        # force_durations = [0.3, 0.3, 0.3]
        # force_vecs = [
        #     np.array([50.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        #     np.array([0.0, 30.0, 0.0, 0.0, 0.0, 0.0]),
        #     np.array([30.0, 40.0, 0.0, 0.0, 0.0, 0.0]),
        # ]
        
        sim_time = 120
        start_step = int(1.0 / SIM_DT)
        end_step = int((sim_time-1) / SIM_DT)
        every_n_timesteps = 2000  # every 3 seconds at 1 kHz

        force_start_times, force_durations, force_vecs = generate_force_perturbation_schedule_every_n_steps(
            start_step=start_step,
            end_step=end_step,
            every_n_timesteps=every_n_timesteps,
            sim_dt=SIM_DT
        )

 
        for i in range(3):
            v_des = [goal_list[i], 0.0, 0.0]
            _, record_path = rollout_mpc_phase_percentage_shift(
                show_plot=False,
                visualize=True,
                record_video=False,
                v_des=v_des,
                sim_time=sim_time,
                save_data=True,
                record_dir=experiment_dir,
                nominal_flag=False,
                force_start_times=force_start_times,
                force_durations=force_durations,
                force_vecs=force_vecs,
            )

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

                if "nominal" not in file_name.lower():
                    self.save_ood_val_set_dummy(experiment_dir, states, vc_goals, cc_goals, actions, file_name)
                    # self.save_ood_val_set_l2_distance(experiment_dir, states, vc_goals, cc_goals, actions, times, file_name, distance_threshold=4.0)

        self.save_dataset(iteration=0)
        ood_save_path = os.path.join(self.data_save_path, "ood_val_data.npz")
        self.ood_database.save_as_npz(ood_save_path)
        print(f"OOD validation dataset saved to {ood_save_path}")

@hydra.main(config_path='cfgs', config_name='data_collection_config.yaml', version_base="1.1")
def main(cfg):
    dc = DataCollection(cfg)
    dc.run()

if __name__ == '__main__':
    main()