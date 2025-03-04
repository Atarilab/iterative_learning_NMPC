import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../..')

import argparse
import time
from typing import List, Dict
import pinocchio as pin
import numpy as np

from mj_pin.abstract import VisualCallback, DataRecorder, Controller # type: ignore
from mj_pin.simulator import Simulator # type: ignore
from mj_pin.utils import get_robot_description, mj_frame_pos, mj_joint_name2act_id, mj_joint_name2dof   # type: ignore

from iterative_supervised_learning.utils.network import GoalConditionedPolicyNet
from iterative_supervised_learning.utils.database import Database

import torch
from tqdm import tqdm
import threading
import mujoco
import mujoco.viewer

# define global variables

SIM_DT = 1.0e-3
VIEWER_DT = 1/30.
# with base_wrt_feet
n_state = 44 # state:44 + vc_goal:3

# without base_wrt_feet
# n_state = 36

n_state += 3
print("n_state = ",n_state)
n_action = 12
kp = 40.0
kd = 5.0 

def get_phase_percentage(t:int):
    """get current gait phase percentage based on gait period

    Args:
        t (int): current sim step (NOT sim time!)

    Returns:
        phi: current gait phase. between 0 - 1
    """ 
       
    # for trot
    gait_period = 0.5
    phi = (t % gait_period)/gait_period
    return phi

class PolicyController(Controller):
    def __init__(self, policy_path: str, 
                 n_state: int, 
                 n_action: int, 
                 joint_name2act_id: Dict[str, int],
                 v_des: np.ndarray = np.array([0.3, 0.0, 0.0]), 
                 norm_policy_input: bool = False,
                 database_path: str = "", 
                 device: str = "cpu",
                 mj_model = None):
        super().__init__()
        
        # initialize policy network
        self.device = device
        self.policy_net = GoalConditionedPolicyNet(input_size=n_state, output_size=n_action, num_hidden_layer=3,
                                                   hidden_dim=512, batch_norm=True)
        self.policy_net.load_state_dict(torch.load(policy_path, map_location=device)['network'])
        self.policy_net.to(device)
        self.policy_net.eval()
        print("policy network is = ", self.policy_net)
        
        # initialization 
        self.joint_name2act_id = joint_name2act_id
        self.v_des = v_des
        self.n_state = n_state
        self.nu = 18 # base joint: 6 + 4* each leg:3
        
        # initialize robot description
        self.robot_name = "go2"
        # xml_path = get_robot_description(self.robot_name).xml_path
        self.mj_model = mj_model

        # load data base and get mean & std
        self.norm_policy_input = norm_policy_input
        self.mean_std = None
        if self.norm_policy_input and database_path:
            db = Database(limit=10000000, norm_input=True)
            db.load_saved_database(database_path)
            # db.calc_input_mean_std()
            self.mean_std = db.get_database_mean_std()
            # self.mean_std = np.array(self.mean_std, dtype=np.float64)
            print("self.mean_std = ", self.mean_std)
            # print("shape of self.mean_std = ", np.shape(self.mean_std))
            input()
        
        # # for debugging purpose: load PD target from file and see if it replays
        # data_path = "/home/atari/workspace/iterative_supervised_learning/examples/data/simulation_data_03_04_2025_14_35_43.npz"
        # data = np.load(data_path)
        # self.action_history = data["action"]

    def compute_torques_dof(self, mj_data) -> Dict[str, float]:
        # extract q and v from mujoco
        q = mj_data.qpos.copy()
        v = mj_data.qvel.copy()
        print("current q is = ", q)
        print("current v is = ", v)
        
        # calculate phase_percentage
        current_time = np.round(mj_data.time,4)
        print("current simulation time = ", current_time)
        phase_percentage = get_phase_percentage(current_time)
        print("current phase_percentage is = ", phase_percentage)
        # input()
        
        # robot_state
        robot_state = q[2:]
        
        # base position
        base_position = q[:3]
        print("current base_position is = ",base_position)
        
        # base with right to feet
        # base_wrt_feet = self._get_base_wrt_feet(mj_data)
        
        feet_names = ["FL", "FR", "RL", "RR"]
        base_wrt_feet = np.zeros(2 * len(feet_names))
        feet_pos_all = []
        
        for i, f_name in enumerate(feet_names):
            feet_pos = mj_frame_pos(self.mj_model, mj_data, f_name)  # Add self.mj_model as the first argument
            feet_pos_all.extend(feet_pos)
            base_wrt_feet[2 * i:2 * i + 2] = (q[:3] - feet_pos)[:2]
            
        print("feet positions are = ",feet_pos_all)
        print("shape of feet_pos_all is  = ", np.shape(feet_pos_all))
        
        print("base_wrt_feet = ", base_wrt_feet)
        
        
        # combine state variable
        # state with base_wrt_feet
        state = np.concatenate(([phase_percentage], v, robot_state, base_wrt_feet))[:self.n_state-3]
        
        # state without base_wrt_feet
        # state = state = np.concatenate(([phase_percentage], v, robot_state))[:self.n_state-3]
        
        # normalize state without phase percentage
        if self.norm_policy_input and self.mean_std is not None:
            state_mean, state_std = self.mean_std[0],self.mean_std[1]
            # print("state_mean = ", state_mean)
            # print("state_std = ",state_std)
            # input()
            state[1:] = (state[1:] - state_mean[1:]) / state_std[1:]

        print("current state is  = ", state)
        # input()
        
        #==============================================================================================
        # form policy input
        x = np.concatenate([np.array(state), np.array(self.v_des)])[:self.n_state]
        print("current policy input is = ", x)
        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # get policy output
        y_tensor = self.policy_net(x_tensor)
        action = y_tensor.detach().cpu().numpy().reshape(-1)
        print("PD target is = ", action)
        #===================================================================================================
        
        # for debugging purposes
        # hold original position
        # action = [0,0.9,-1.8,
        #           0,0.9,-1.8,
        #           0,0.9,-1.8,
        #           0,0.9,-1.8,]

        # # hold given position
        # action = [0.3,1.0,-1.8,
        #           -0.3,1.0,-1.8,
        #           0.3,1.0,-1.8,
        #           -0.3,1.0,-1.8,]
        
        # # read from file to see if PD controller works
        # action = self.action_history[int(current_time/SIM_DT)] # action is in the order of [FL,FR,RL,RR]
        # print("current action index = ", int(current_time/SIM_DT))
        # print("current PD target = ",action)
        
        # calculate torque based on PD target
        tau_flfrrlrr = kp * (action - q[7:]) - kd * v[6:]
        FL_torque = tau_flfrrlrr[0:3]
        FR_torque = tau_flfrrlrr[3:6]
        RL_torque = tau_flfrrlrr[6:9]
        RR_torque = tau_flfrrlrr[9:]
        tau_frflrrrl = np.concatenate([FR_torque,FL_torque,RR_torque,RL_torque])
        
        print("joint position is = ", q[7:])
        print("joint velocity is = ", v[6:])
        
        print("tau_flfrrlrr is = ")
        print(tau_flfrrlrr)
        print("FR torque is ")
        print(FR_torque)
        print("FL torque is ")
        print(FL_torque)
        print("RR torque is ")
        print(RR_torque)
        print("RL torque is ")
        print(RL_torque)
        print("tau_frflrrrl is = ")
        print(tau_frflrrrl)
        # input()
        
        # store torque to self.torques_dof
        self.torques_dof = np.zeros(self.nu)
        self.torques_dof[-12:] = tau_flfrrlrr
        print(f"current time {current_time}: Applied control torques (high precision): {self.torques_dof}")
        # input()

    def get_torque_map(self) -> Dict[str, float]:
        # print(self.joint_name2act_id)
        torque_map = {
            j_name: self.torques_dof[dof_id]
            for j_name, dof_id in self.joint_name2act_id.items()
        }
        print("current torque map is = ", torque_map)
        return torque_map

def rollout_policy(
    policy_path: str,
    robot_name: str = "go2",
    sim_time: float = 2.0,
    v_des: np.ndarray = np.array([0.3, 0.0, 0.0]),
    record_video: bool = False,
    save_data: bool = True,
    record_dir: str = "./data/",
    visualize: bool = True,
    norm_policy_input: bool = True,
    database_path: str = ""
):  
    # set up robot description
    robot_desc = get_robot_description(robot_name)
    
    # set up simulator
    sim = Simulator(robot_desc.xml_scene_path, sim_dt=SIM_DT, viewer_dt=VIEWER_DT)
    sim.vs.track_obj = "base"
    sim.vs.set_front_view()
    sim.setup()
    # joint_name2act_id= mj_joint_name2act_id(sim.mj_model)
    joint_name2act_id= mj_joint_name2dof(sim.mj_model)
    print("Joint to Actuator ID Mapping:", joint_name2act_id)
    # input()

    controller = PolicyController(
        policy_path=policy_path,
        n_state=n_state,
        n_action=n_action,
        joint_name2act_id=joint_name2act_id,
        v_des=v_des,
        norm_policy_input=norm_policy_input,
        database_path=database_path,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        mj_model=sim.mj_model
    )

    sim.run(
        sim_time=sim_time,
        use_viewer=visualize,
        controller=controller,
        record_video=record_video,
    )
    print("🎉 Policy rollout finished successfully.")
    
    
    

if __name__ == '__main__':
    policy_path = '/home/atari/workspace/iterative_supervised_learning/examples/data/behavior_cloning/trot/Mar_04_2025_15_40_12/network/policy_final.pth'
    database_path = "/home/atari/workspace/iterative_supervised_learning/examples/data/behavior_cloning/trot/Mar_04_2025_15_40_12/dataset/database_0.hdf5"
    rollout_policy(policy_path, 
                   sim_time=2.0, 
                   v_des=[0.3, 0.0, 0.0], 
                   record_video=False,
                   database_path=database_path,
                   norm_policy_input=True)
    

    