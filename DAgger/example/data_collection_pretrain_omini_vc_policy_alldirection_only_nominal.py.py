# This script does behavior cloning on omini-vc goals and returns a suboptimal polivy## NOTE: This script is for data collection for behavior cloning.
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../..')

from omegaconf import OmegaConf
import argparse
from typing import Tuple, List
import numpy as np
from datetime import datetime

from DAgger.utils.RolloutMPC import rollout_mpc_phase_percentage_shift
from DAgger.utils.database import Database
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
    
    def merge_all_datasets(self):
        import glob

        # Gather all per-iteration dataset files
        h5_files = sorted(glob.glob(os.path.join(self.data_save_path, "database_*.hdf5")))
        print(f"Found {len(h5_files)} datasets to merge.")

        merged_data = {
            "states": [],
            "vc_goals": [],
            "cc_goals": [],
            "actions": [],
            "traj_ids": [],
            "traj_times": [],
        }

        for fpath in h5_files:
            print(f"Reading {fpath}")
            with h5py.File(fpath, 'r') as hf:
                for key in merged_data:
                    if key in hf:
                        merged_data[key].append(hf[key][:])

        # Concatenate arrays
        for key in merged_data:
            if merged_data[key]:  # Non-empty
                merged_data[key] = np.concatenate(merged_data[key], axis=0)
            else:
                merged_data[key] = None  # Allow optional keys

        # Save final merged dataset
        final_path = os.path.join(self.data_save_path, "database_final.hdf5")
        with h5py.File(final_path, 'w') as hf:
            for key, data in merged_data.items():
                if data is not None:
                    hf.create_dataset(key, data=data)

        print(f"✅ Merged final dataset saved to {final_path}")
    
    def sample_all_goal_velocities(self) -> List[np.ndarray]:
        vx_max = self.vx_range[1]
        vy_max = self.vy_range[1]
        w_fixed = 0

        angle_interval_deg = 30
        num_speeds = 5

        angles_deg = np.arange(0, 360, angle_interval_deg)
        goal_velocities = [np.array([0.0, 0.0, w_fixed])]  # Include [0, 0, 0] once

        for angle_deg in angles_deg:
            theta = np.deg2rad(angle_deg)
            dx = np.cos(theta)
            dy = np.sin(theta)

            for s in np.linspace(0, 1.0, num_speeds + 1)[1:]:  # Skip s=0 to avoid duplicate [0,0,0]
                vx = s * vx_max * dx
                vy = s * vy_max * dy
                goal = np.round(np.array([vx, vy, w_fixed]), decimals=6)
                goal_velocities.append(goal)

        return goal_velocities



    def rollout_nominal_trajectory(self, v_des, experiment_dir):
        _, record_path_nominal = rollout_mpc_phase_percentage_shift(
            show_plot=False,
            visualize=True,
            record_video=True,
            v_des=v_des,
            sim_time=2.0,
            save_data=True,
            record_dir=experiment_dir,
            nominal_flag=True
        )
        print("loading nominal traj data from path = ")
        print(record_path_nominal)
        return np.load(record_path_nominal)

    def sample_replanning_points(self, traj_length: int) -> List[int]:
        replanning_points = []
        gait_period = 0.5
        num_replanning = int(gait_period * 1000 / replan_freq)
        start_timestep = t0 * 1000
        for i in range(num_replanning):
            next_replanning_point = int(i * replan_freq + start_timestep)
            replanning_points.append(next_replanning_point)
        print("Replanning points:", replanning_points)
        return replanning_points

    def rollout_perturbed_trajectories(self, state, nominal_q, nominal_v, contact_vec,
                                        replanning_points, experiment_dir, v_des):
        for i_replanning in replanning_points:
            print(f"Replanning at step {i_replanning}")
            q0 = nominal_q[i_replanning]
            q0[0] = 0  # fix base x offset
            v0 = nominal_v[i_replanning]
            current_contact_vec = contact_vec[i_replanning]
            ee_in_contact = contact_vec_to_frame_names(current_contact_vec)

            for j in range(self.num_pertubations_per_replanning):
                phase_percentage = state[:, 0]
                randomize_on_given_state = np.concatenate((q0, v0, np.array([phase_percentage[i_replanning]])))
                current_time = np.round(i_replanning * SIM_DT, 4)

                early_termination = False
                while True:
                    force_direction = np.random.uniform(-1.0, 1.0, size=3)
                    force_direction /= np.linalg.norm(force_direction) + 1e-6
                    magnitude = np.random.uniform(50.0, 70.0)
                    force_vec = np.concatenate([magnitude * force_direction, np.zeros(3)])
                    force_start_time = 0.0
                    force_duration = np.random.uniform(0.2, 0.4)

                    print(f"Random push: start={force_start_time:.2f}s, duration={force_duration:.2f}s, vec={force_vec[:3]}")

                    early_termination, _ = rollout_mpc_phase_percentage_shift(
                        randomize_on_given_state=randomize_on_given_state,
                        v_des=v_des,
                        sim_time=1.5,
                        current_time=current_time,
                        show_plot=False,
                        visualize=False,
                        record_video=True,
                        save_data=True,
                        record_dir=experiment_dir,
                        ee_in_contact=ee_in_contact,
                        nominal_flag=False,
                        replanning_point=i_replanning,
                        nth_traj_per_replanning=j+1,
                        force_start_time=force_start_time,
                        force_duration=force_duration,
                        force_vec=force_vec,
                    )
                    if not early_termination:
                        break

    def merge_trajs_from_dir(self, experiment_dir):
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

                if not is_nominal:
                    self.save_ood_val_set_l2_distance(
                        experiment_dir, states, vc_goals, cc_goals, actions, times, file_name, distance_threshold=4.0
                    )

    def run(self):
        goal_velocities = self.sample_all_goal_velocities()

        print("\n📌 Sampled Goal Velocities:")
        for i, v in enumerate(goal_velocities):
            print(f"  {i:2d}: {v}")
        input()
        
        for i_iter, v_des in enumerate(goal_velocities):
            print(f"\n🔹 Iteration {i_iter + 1}/{len(goal_velocities)} — Executing goal velocity: {v_des}")

            experiment_dir = os.path.join(self.data_save_path, f"experiment/iter_{i_iter}")
            os.makedirs(experiment_dir, exist_ok=True)

            data = self.rollout_nominal_trajectory(v_des, experiment_dir)

            self.merge_trajs_from_dir(experiment_dir)
            self.save_dataset(iteration=i_iter)
            self.database = Database(limit=self.cfg.database_size, norm_input=True)

            ood_save_path = os.path.join(self.data_save_path, f"ood_val_data_iter_{i_iter}.npz")
            self.ood_database.save_as_npz(ood_save_path)
            print(f"✅ OOD validation dataset saved to {ood_save_path}")

        # After all iterations, save the combined training dataset
        self.merge_all_datasets()


@hydra.main(config_path='../cfgs', config_name='data_collection_safedagger_config.yaml', version_base="1.1")
def main(cfg):
    dc = DataCollection(cfg)
    dc.run()

if __name__ == '__main__':
    main()