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
# from iterative_supervised_learning.utils.RolloutMPC import rollout_mpc

from iterative_supervised_learning.utils.RolloutMPC_test import RolloutMPC
from iterative_supervised_learning.utils.RolloutMPC_test import rollout_mpc
from iterative_supervised_learning.utils.database import Database
import random
import hydra
import h5py
import pickle
import scipy.spatial.transform as st
import pinocchio as pin
from mj_pin.utils import get_robot_description

def random_quaternion_perturbation(sigma):
    """
    Generate a small random quaternion perturbation.
    The perturbation is sampled from a normal distribution with standard deviation sigma.
    """
    random_axis = np.random.normal(0, 1, 3)  # Random rotation axis
    random_axis /= np.linalg.norm(random_axis)  # Normalize to unit vector
    angle = np.random.normal(0, sigma)  # Small random rotation angle
    perturb_quat = st.Rotation.from_rotvec(angle * random_axis).as_quat()  # Convert to quaternion
    return perturb_quat

def apply_quaternion_perturbation(nominal_quat, sigma_base_ori):
    """
    Apply a small random rotation perturbation to a given quaternion.
    """
    perturb_quat = random_quaternion_perturbation(sigma_base_ori)
    perturbed_quat = st.Rotation.from_quat(nominal_quat) * st.Rotation.from_quat(perturb_quat)
    return perturbed_quat.as_quat()  # Convert back to quaternion

class DataCollection:
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
        # very basic running script
        for i in range(self.n_iteration):
            gait = random.choice(self.gaits)
            
            # Sample disturbed goals around default goals
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
    
    def run_rand_initial_condition(self):
        for i in range(self.n_iteration):
            gait = random.choice(self.gaits)
            
            # Sample disturbed goals around default goals
            # Default goal velocity
            # v_des_default = np.array([0.5, 0.1, 0])

            # Define disturbance levels for each entry
            # noise_std = np.array([0.02, 0.01, 0.0])  # No noise for the third entry

            # Sample disturbed goals
            # v_des = v_des_default + np.random.normal(0, noise_std, size=3)
            
            v_des = [0.5,0.1,0.0]
            print("Sampled goal is:", v_des)

            # TODO: set customized initial condition
            

            record_dir = f"{self.data_save_path}/iteration_{i}/"
            os.makedirs(record_dir, exist_ok=True)
            
            _, state_history, base_history, vc_goal_history, cc_goal_history, ctrl = rollout_mpc(
                mode="close_loop",
                sim_time=self.episode_length * self.sim_dt,
                robot_name=self.cfg.robot_name,
                record_dir=record_dir,
                v_des=v_des,
                save_data=True,
                visualize=False,
                randomize_initial_state=True
            )
            print(f"Shape of states: {state_history.shape}, Type: {type(state_history)}")
            if len(state_history) != 0:
                self.database.append(states=state_history, vc_goals=vc_goal_history, cc_goals=cc_goal_history, actions=ctrl)
                print("MPC data saved into database")
                print('database size: ' + str(len(self.database)))  
            else:
                print('mpc rollout failed')
            
            self.save_dataset(i)
    
    def run_perturbed_mpc_when_replanning(self):
        nv = 18
        nq = 17
        plan_freq = 1000 #replan every 1000 steps
        n_state = 35
        for i in range(self.n_iteration):
            print(f"============ Iteration {i+1}  ==============")
            v_des = [0.5,0.1,0.0]
            print("Sampled goal is:", v_des)
            
            record_dir = f"{self.data_save_path}/iteration_{i}/"
            os.makedirs(record_dir, exist_ok=True)
            # rollout nominal trajectory
            _, nominal_state_history, nominal_base_history, nominal_vc_goal_history, nominal_cc_goal_history, nominal_ctrl = rollout_mpc(
                mode="close_loop",
                sim_time=self.episode_length * self.sim_dt,
                robot_name=self.cfg.robot_name,
                record_dir=record_dir,
                v_des=v_des,
                save_data=True,
                visualize=True,
                randomize_initial_state=True,
            )
            # print("first line of nominal_base_history = ",nominal_base_history[0])
            # print("shape of nominal_base_history = ",np.shape(nominal_base_history))
            
            nominal_pos = np.concatenate((nominal_base_history[:,:2], nominal_state_history[:, nv:]), axis=1)
            nominal_vel = nominal_state_history[:, :nv]  # This line is fine

            # print("shape of nominal_pos = ",np.shape(nominal_pos))
            # print("shape of nominal_vel = ",np.shape(nominal_vel))
            
            # Calculate replanning points
            replanning_points = np.arange(0, self.episode_length, plan_freq)
            print("Replanning points:", replanning_points)
            # input()

            # Rollout MPC from each replanning point
            for i_replanning in replanning_points:
                print(f"Replanning at step {i_replanning}")

                # Extract nominal state at replanning point
                initial_q = nominal_pos[i_replanning]  # Position (full state q)
                initial_v = nominal_vel[i_replanning]  # Velocity
                # print(initial_q)
                # print(np.shape(initial_q))
                # print(initial_v)
                # print(np.shape(initial_v))
                # input()
                
                initial_state = np.concatenate((initial_q, initial_v), axis=0)

                # Run MPC from this replanning state
                __, replanned_state_history, replanned_base_history, replanned_vc_goal_history, replanned_cc_goal_history, nominal_ctrl = rollout_mpc(
                    mode="close_loop",
                    sim_time=(self.episode_length - i_replanning) * self.sim_dt,  # Remaining time
                    robot_name=self.cfg.robot_name,
                    record_dir=record_dir + f"/replanning_{i_replanning}/",
                    v_des=v_des,
                    save_data=True,
                    visualize=True,
                    randomize_initial_state=True,
                    set_initial_state=initial_state,  # Set initial state for MPC
                )
                

    def run_perturbed_mpc_when_replanning_test(self):
        nv = 18
        nq = 17 # 19 - 2(two absolute horizontal coordinate of base point)
        plan_freq = 100  # Replan every 1000 steps
        n_state = 36 # nv + nq + phase_percentage
        
        # pertubation variables
        mu_base_pos = 0.0
        sigma_base_pos = 0.1
        mu_joint_pos = 0.0
        sigma_joint_pos = 0.2
        mu_base_ori = 0.0
        sigma_base_ori = 0.1
        mu_vel = 0.0
        sigma_vel = 0.1

        for i in range(self.n_iteration):
            print(f"============ Iteration {i+1}  ==============")
            v_des = np.array([0.3, 0.0, 0.0])  # Sampled goal
            print("Sampled goal is:", v_des)

            record_dir = f"{self.data_save_path}/iteration_{i}/"
            os.makedirs(record_dir, exist_ok=True)

            # Rollout nominal trajectory
            _, nominal_state_history, nominal_base_history, nominal_vc_goal_history, nominal_cc_goal_history, nominal_ctrl = rollout_mpc(
                mode="close_loop",
                sim_time=self.episode_length * self.sim_dt,
                robot_name=self.cfg.robot_name,
                record_dir=record_dir,
                v_des=v_des,
                save_data=True,
                visualize=False,
                randomize_initial_state=False,
                show_plot=False
            )

            if len(nominal_state_history) == 0:
                print(f"Nominal MPC rollout failed at iteration {i}")
                continue  # Skip to next iteration if nominal rollout fails

            # Compute nominal position (q) and velocity (v)
            nominal_pos = np.concatenate((nominal_base_history[:, :2], nominal_state_history[:, nv+1:]), axis=1)
            nominal_vel = nominal_state_history[:, 1:nv+1]

            # Store nominal trajectory in database
            self.database.append(
                states=nominal_state_history,
                vc_goals=nominal_vc_goal_history,
                cc_goals=nominal_cc_goal_history,
                actions=nominal_ctrl,
            )
            print("Nominal trajectory saved in database.")

            # Calculate replanning points
            replanning_points = np.arange(0, self.episode_length, plan_freq)
            print("Replanning points:", replanning_points)

            # Rollout MPC from each replanning point
            for i_replanning in replanning_points:
                print(f"Replanning at step {i_replanning}")

                for j in range(self.num_pertubations_per_replanning):
                    print(f"executing the {j+1}th pertubation at replanning point {i_replanning}")
                    # Extract nominal state at replanning point
                    nominal_q = nominal_pos[i_replanning]  # Position (full state q)
                    nominal_v = nominal_vel[i_replanning]  # Velocity
                    
                    # very important: pass current time(i_replanning) to rollout_mpc in order to calculate current phase percentage
                    nominal_state = np.concatenate((nominal_q, nominal_v, np.array([i_replanning])))
                    
                    # print("shape of initial_q is = ",np.shape(initial_q))
                    # print("shape of initial_v is = ",np.shape(initial_v))
                    
                    # print(nominal_state)
                    # print("shape of nominal_state is  = ", np.shape(nominal_state))
                    # input()
                    
                    # NOTE: randomize quatenion is tricky
                    # nominal_quat = initial_q[3:7]
                    # perturbed_quat = apply_quaternion_perturbation(nominal_quat, sigma_base_ori)
                    # print("nominal_quat = ",nominal_quat)
                    # print("perturbed_quat = ", perturbed_quat)
                    # input()
                    
                    # base orientation is in quatenion
                    # perturbation_q = np.concatenate((np.random.normal(mu_base_pos, sigma_base_pos, 3),\
                    #                                 perturbed_quat ,\
                    #                                 np.random.normal(mu_joint_pos,sigma_joint_pos,len(initial_q)-7)))
                    # perturbation_v = np.random.normal(mu_vel,sigma_vel,len(initial_v))
                    
                    # print("perturbation_q = ",perturbation_q)
                    # print("perturbation_v = ",perturbation_v)
                    # input()
                    # initial_state = np.concatenate((initial_q+perturbation_q, initial_v+perturbation_v), axis=0)  # Full initial state
                    # print(initial_state)
                    # print(np.shape(initial_state))
                    # input()
                    
                    # Run MPC from this replanning state
                    while True:
                        __, replanned_state_history, replanned_base_history, replanned_vc_goal_history, replanned_cc_goal_history, replanned_ctrl = rollout_mpc(
                            mode="close_loop",
                            sim_time=4.0,
                            robot_name=self.cfg.robot_name,
                            record_dir=record_dir + f"/replanning_{i_replanning}/",
                            v_des=v_des,
                            save_data=True,
                            visualize=False,
                            randomize_initial_state=False,  # Use predefined state
                            randomize_on_given_state=nominal_state,
                            show_plot=False
                        )
                        
                        # NOTE: break loop if simulation 
                        if len(replanned_state_history) != 0:
                            break
                        else:
                            print(f"Replanned MPC rollout failed at step {i_replanning}")

                    # Store replanned trajectory in database
                    self.database.append(
                        states=replanned_state_history,
                        vc_goals=replanned_vc_goal_history,
                        cc_goals=replanned_cc_goal_history,
                        actions=replanned_ctrl,
                    )
                print(f"Replanned trajectory at step {i_replanning} saved in database.")
                print("current database length is = ", self.database.length)

            # Save dataset after each iteration
            self.save_dataset(i)

    def run_perturbed_mpc_when_replanning_with_new_state_space(self):
        nv = 18
        nq = 17 # 19 - 2(two absolute horizontal coordinate of base point)
        plan_freq = 500  # Replan every 1000 steps
        n_state = 44 # phase_percentage + q[2:] + v + base_wrt_feet
        
        for i in range(self.n_iteration):
            print(f"============ Iteration {i+1}  ==============")
            v_des = np.array([0.3, 0.0, 0.0])  # Sampled goal
            print("Sampled goal is:", v_des)

            record_dir = f"{self.data_save_path}/iteration_{i}/"
            os.makedirs(record_dir, exist_ok=True)

            # Rollout nominal trajectory
            _, nominal_state_history, nominal_base_history, nominal_vc_goal_history, nominal_cc_goal_history, nominal_ctrl = rollout_mpc(
                mode="close_loop",
                sim_time=self.episode_length * self.sim_dt,
                robot_name=self.cfg.robot_name,
                record_dir=record_dir,
                v_des=v_des,
                save_data=True,
                visualize=False,
                randomize_initial_state=False,
                show_plot=False
            )

            if len(nominal_state_history) == 0:
                print(f"Nominal MPC rollout failed at iteration {i}")
                continue  # Skip to next iteration if nominal rollout fails

            # Compute nominal position (q) and velocity (v)
            nominal_pos = np.concatenate((nominal_base_history[:, :2], nominal_state_history[:, nv+1:-8]), axis=1)
            nominal_vel = nominal_state_history[:, 1:nv+1]

            # Store nominal trajectory in database
            self.database.append(
                states=nominal_state_history,
                vc_goals=nominal_vc_goal_history,
                cc_goals=nominal_cc_goal_history,
                actions=nominal_ctrl,
            )
            print("Nominal trajectory saved in database.")

            # Calculate replanning points
            replanning_points = np.arange(0, self.episode_length, plan_freq)
            print("Replanning points:", replanning_points)

            # Rollout MPC from each replanning point
            for i_replanning in replanning_points:
                print(f"Replanning at step {i_replanning}")

                for j in range(self.num_pertubations_per_replanning):
                    print(f"executing the {j+1}th pertubation at replanning point {i_replanning}")
                    # Extract nominal state at replanning point
                    nominal_q = nominal_pos[i_replanning]  # Position (full state q)
                    nominal_v = nominal_vel[i_replanning]  # Velocity
                    
                    # very important: pass current time(i_replanning) to rollout_mpc in order to calculate current phase percentage
                    nominal_state = np.concatenate((nominal_q, nominal_v, np.array([i_replanning])))
                    
                    # print("shape of initial_q is = ",np.shape(initial_q))
                    # print("shape of initial_v is = ",np.shape(initial_v))
                    
                    # print(nominal_state)
                    # print("shape of nominal_state is  = ", np.shape(nominal_state))
                    # input()
                                    
                    # Run MPC from this replanning state
                    while True:
                        __, replanned_state_history, replanned_base_history, replanned_vc_goal_history, replanned_cc_goal_history, replanned_ctrl = rollout_mpc(
                            mode="close_loop",
                            sim_time=5.0,
                            robot_name=self.cfg.robot_name,
                            record_dir=record_dir + f"/replanning_{i_replanning}/",
                            v_des=v_des,
                            save_data=True,
                            visualize=False,
                            randomize_initial_state=False,  # Use predefined state
                            randomize_on_given_state=nominal_state,
                            show_plot=False
                        )
                        
                        # NOTE: break loop if simulation 
                        if len(replanned_state_history) != 0:
                            break
                        else:
                            print(f"Replanned MPC rollout failed at step {i_replanning}")

                    # Store replanned trajectory in database
                    self.database.append(
                        states=replanned_state_history,
                        vc_goals=replanned_vc_goal_history,
                        cc_goals=replanned_cc_goal_history,
                        actions=replanned_ctrl,
                    )
                print(f"Replanned trajectory at step {i_replanning} saved in database.")
                print("current database length is = ", self.database.length)

            # Save dataset after each iteration
            self.save_dataset(i)
            

# Example usage with database
@hydra.main(config_path='cfgs', config_name='data_collection_config.yaml',version_base="1.1")
def main(cfg):
    dc = DataCollection(cfg)
    # default run is gather n_iteration nominal mpc trajectories
    # dc.run()
    
    # since default run gather too much irrelevant data, I try to randomize initial state for different mpc rollouts
    # dc.run_rand_initial_condition()
    
    # try rollout with replanning
    # dc.run_perturbed_mpc_when_replanning_test()
    
    # try data collection with new state space
    dc.run_perturbed_mpc_when_replanning_with_new_state_space()

if __name__ == '__main__':
    main()

