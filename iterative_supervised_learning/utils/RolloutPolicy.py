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
import random
from tqdm import tqdm
import threading
import mujoco
import mujoco.viewer

import matplotlib.pyplot as plt

# Set random seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# define global variables
SIM_DT = 1.0e-3
VIEWER_DT = 1/30.
HIDDEN_DIM = 512
t0 = 0.028

# with base_wrt_feet
n_state = 44 # state:44 + vc_goal:3

n_state += 3
print("n_state = ",n_state)
n_action = 12

kp = 20.0
kd = 1.5

def get_phase_percentage(t:int):
    """get current gait phase percentage based on gait period

    Args:
        t (int): current sim step (NOT sim time in seconds!)

    Returns:
        phi: current gait phase. between 0 - 1
    """ 
       
    # for trot
    gait_period = 0.5
    if t < t0:
        return 0
    else:
        phi = ((t-t0) % gait_period)/gait_period
        return phi

# Data recorder
class StateDataRecorder(DataRecorder):
    def __init__(
        self,
        record_dir: str = "",
        record_step: int = 1,
        v_des: np.ndarray = np.array([0,0,0]),
        current_time: float = 0.0) -> None:
        """
        A simple data recorder that saves simulation data to a .npz file.
        """
        super().__init__(record_dir, record_step)
        self.data = {}
        self.vc_goals = v_des
        self.cc_goals = np.random.normal(loc=0.0, scale=0.1, size=(8,))
        self.current_time = current_time
        
        # initialization of robot model
        self.feet_names = ["FL", "FR", "RL", "RR"]
        self.robot_name = "go2"
        xml_path = get_robot_description(self.robot_name).xml_path
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        
        # reset
        self.reset()

    def reset(self) -> None:
        self.data = {"time": [], 
                     "q": [], 
                     "v": [], 
                     "ctrl": [],
                     "feet_pos_w":[],
                     "base_wrt_feet":[],
                     "state":[],
                     "action":[],
                     "vc_goals":[],
                     "cc_goals":[]}

    def save(self) -> None:
        if not self.record_dir:
            self.record_dir = os.getcwd()
        os.makedirs(self.record_dir, exist_ok=True)

        timestamp = self.get_date_time_str()
        file_path = os.path.join(self.record_dir, f"simulation_data_{timestamp}.npz")

        try:
            # Uncomment to save data
            np.savez(file_path, **self.data)
            print(f"Data successfully saved to {file_path}")
            return file_path
        except Exception as e:
            print(f"Error saving data: {e}")
            return ""
    
    def record(self, mj_data) -> None:
        """
        Record simulation data at the current simulation step.
        """
        # Record time and state
        q = mj_data.qpos.copy()
        v = mj_data.qvel.copy()
        self.data["time"].append(round(mj_data.time + self.current_time, 4))
        self.data["q"].append(q) # in the order of [FL,FR,RL,RR]
        self.data["v"].append(v) # in the order of [FL,FR,RL,RR]
        self.data["ctrl"].append(mj_data.ctrl.copy()) # in the order of [FR,FL,RR,RL]
        
        # Record feet position in the world (x,y,z)
        feet_pos_all = []
        ee_in_contact = []
        base_wrt_feet = np.zeros(2*len(self.feet_names))
        
        for i, f_name in enumerate(self.feet_names):
            feet_pos = mj_frame_pos(self.mj_model, mj_data, f_name)
            # print(f"{f_name} feet_pos = {feet_pos}")
            if feet_pos[-1] <= 0.005:
                ee_in_contact.append(f_name)
            feet_pos_all.extend(feet_pos)
            base_wrt_feet[2*i:2*i+2] = (q[:3] - feet_pos)[:2]
        
        self.data["feet_pos_w"].append(np.array(feet_pos_all))
        
        # base with right to feet in world frame
        self.data["base_wrt_feet"].append(np.array(base_wrt_feet))
        
        ## form state variable
        # the format of state = [[phase_percentage],v,q[2:],base_wrt_feet]
        # if in replanning step, phase percentage is not starting from 0
        phase_percentage = np.round([get_phase_percentage(mj_data.time + self.current_time)], 4)
        
        #==========================================================================================
        # state with base_wrt_feet
        state = np.concatenate([phase_percentage, v, q[2:], base_wrt_feet])
        self.data["state"].append(np.array(state)) # here is unnormalized state
        #=========================================================================================
        # transform action from torque to PD target and store
        tau_frflrrrl = mj_data.ctrl.copy() # in the order of [FR,FL,RR,RL]
        FR_torque = tau_frflrrrl[0:3]
        FL_torque = tau_frflrrrl[3:6]
        RR_torque = tau_frflrrrl[6:9]
        RL_torque = tau_frflrrrl[9:]
        tau_flfrrlrr = np.concatenate([FL_torque,FR_torque,RL_torque,RR_torque])
        
        # calculate realized PD target and store
        action = (tau_flfrrlrr + kd * v[6:])/kp + q[7:] # in the order of [FL,FR,RL,RR]
        self.data["action"].append(np.array(action))
        
        # record the velocity conditioned goals
        self.data["vc_goals"].append(self.vc_goals)
        
        # record contact conditioned goals(currently just a random noise)
        self.cc_goals = np.random.normal(loc=0.0, scale=0.1, size=(8,))
        self.data["cc_goals"].append(self.cc_goals)
        
class PolicyController(Controller):
    def __init__(self, policy_path: str, 
                 n_state: int, 
                 n_action: int, 
                 joint_name2act_id: Dict[str, int],
                 v_des: np.ndarray = np.array([0.3, 0.0, 0.0]), 
                 norm_policy_input: bool = False,
                 device: str = "cpu",
                 mj_model = None,
                 start_time: float = 0.0,
                 reference_path: str = ""):
        super().__init__()
        
        # initialize policy network
        self.device = device
        self.policy_net = GoalConditionedPolicyNet(input_size=n_state, output_size=n_action, num_hidden_layer=3,
                                                   hidden_dim=HIDDEN_DIM, batch_norm=True)
        
        # Load saved state dict and normalization info
        payload = torch.load(policy_path, map_location=device, weights_only = False)

        # Reconstruct network architecture
        self.policy_net = GoalConditionedPolicyNet(input_size=n_state, output_size=n_action,
                                                num_hidden_layer=3, hidden_dim=HIDDEN_DIM,
                                                batch_norm=True)

        self.policy_net.load_state_dict(payload['network_state_dict'])
        self.policy_net.to(device)
        self.policy_net.eval()

        # Load normalization info
        self.norm_policy_input = norm_policy_input
        self.mean_std = None
        if self.norm_policy_input:
            self.mean_std = payload["norm_policy_input"]

        # initialization global variables
        self.joint_name2act_id = joint_name2act_id
        self.v_des = v_des
        self.n_state = n_state
        self.nu = 18 # base joint: 6 + 4* each leg:3
        self.start_time = start_time
        self.ctrl_index = int(self.start_time * 1000)
        self.reference_path = reference_path
        
        # initialize robot description
        self.robot_name = "go2"
        self.mj_model = mj_model
            
        # load reference data for comparing
        self.data_mpc = np.load(self.reference_path)
        self.reference_PD = self.data_mpc["action"]

    def compute_torques_dof(self, mj_data) -> Dict[str, float]:
        # extract q and v from mujoco
        q = mj_data.qpos.copy()
        v = mj_data.qvel.copy()
        
        # calculate phase_percentage
        current_time = np.round(mj_data.time + self.start_time,4)
        phase_percentage = get_phase_percentage(current_time)
        
        print("current q from simulator is = ", q)
        print("current v from simulator is = ", v)
        print("current simulator time = ", mj_data.time)
        print("current time for phase_percentage calculation = ", current_time)
        print("current phase_percentage is = ", phase_percentage)
        # input()
        
        # robot_state
        robot_state = q[2:]
        
        # base position
        base_position = q[:3]
        # print("current base_position is = ",base_position)
        
        # calculate base with right to feet        
        feet_names = ["FL", "FR", "RL", "RR"]
        
        base_wrt_feet = np.zeros(2 * len(feet_names))
        feet_pos_all = []
        
        for i, f_name in enumerate(feet_names):
            feet_pos = mj_frame_pos(self.mj_model, mj_data, f_name)  # Add self.mj_model as the first argument
            feet_pos_all.extend(feet_pos)
            base_wrt_feet[2 * i:2 * i + 2] = (q[:3] - feet_pos)[:2]
        
        # NOTE: form state variable for policy inference
        # state with base_wrt_feet
        state = np.concatenate(([phase_percentage], v, robot_state, base_wrt_feet))[:self.n_state-3]
        
        # normalize state without phase percentage
        if self.norm_policy_input and self.mean_std is not None:
            state_mean, state_std = self.mean_std[0],self.mean_std[1]
            state[1:] = (state[1:] - state_mean[1:]) / state_std[1:]
        #==============================================================================================
        # NOTE: policy network inference
        # form policy input
        x = np.concatenate([np.array(state), np.array(self.v_des)])[:self.n_state]
        print("current policy input is = ", x)
        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # get policy output
        with torch.no_grad():  # Disable gradient tracking
            y_tensor = self.policy_net(x_tensor)

        action_policy = y_tensor.detach().cpu().numpy().reshape(-1)

        print()
        print("Policy generated PD target is = ", action_policy)
        print()
        # print("Reference PD target is = ", self.reference_PD[self.ctrl_index])
        # self.ctrl_index += 1
        
        # calculate torque based on PD target
        # use PD targets generated by the policy
        tau_flfrrlrr_policy = kp * (action_policy - q[7:]) - kd * v[6:]
        
        #==============================================================================================
        # print("joint position from simulator is = ", q[7:])
        # print("mpc joint position is = ",self.q_his[int(current_time/SIM_DT)][7:])
        # print()
        # print("joint velocity from simulator is = ", v[6:])
        # print("mpc joint velocity is = ",self.v_his[int(current_time/SIM_DT)][6:])
        # print()
        
        # print("tau_flfrrlrr is = ")
        # print(tau_flfrrlrr)
        # print("FR torque is ")
        # print(FR_torque)
        # print("FL torque is ")
        # print(FL_torque)
        # print("RR torque is ")
        # print(RR_torque)
        # print("RL torque is ")
        # print(RL_torque)
        # print("tau_frflrrrl is = ")
        # print(tau_frflrrrl)
        # input()
        
        # apply torque calculated from policy output to self.torques_dof
        self.torques_dof = np.zeros(self.nu)
        self.torques_dof[-12:] = tau_flfrrlrr_policy
        
        # print(f"current time {current_time}: Applied control torques (high precision): {self.torques_dof}")
        # print("torque calculated from MPC PD targets is = ", tau_flfrrlrr_MPC)
        # print()
        # input()

    def get_torque_map(self) -> Dict[str, float]:
        # print(self.joint_name2act_id)
        torque_map = {
            j_name: self.torques_dof[dof_id]
            for j_name, dof_id in self.joint_name2act_id.items()
        }
        # print("current torque map is = ", torque_map)
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
    initial_state = [],
    start_time: float = 0.0,
    data_MPC_path : str = ""
):  
    # set up robot description
    robot_desc = get_robot_description(robot_name)
    
    # set up simulator from arbitrary initial condition
    q0 = initial_state[0]
    v0 = initial_state[1]
    sim = Simulator(robot_desc.xml_scene_path, sim_dt=SIM_DT, viewer_dt=VIEWER_DT)
    sim.vs.track_obj = "base"
    sim.setup()
    sim.set_initial_state(q0,v0)
    
    # get action order to joint name mapping
    joint_name2act_id= mj_joint_name2dof(sim.mj_model)
    print("Joint to Actuator ID Mapping:", joint_name2act_id)

    # setup controller
    controller = PolicyController(
        policy_path=policy_path,
        n_state=n_state,
        n_action=n_action,
        joint_name2act_id=joint_name2act_id,
        v_des=v_des,
        norm_policy_input=norm_policy_input,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        mj_model=sim.mj_model,
        start_time = start_time,
        reference_path= data_MPC_path
    )
    
    # initialize data recorder
    if save_data:
        data_recorder = StateDataRecorder(record_dir,v_des=v_des,current_time=start_time)
    else:
        data_recorder = None

    sim.run(
        sim_time=sim_time,
        use_viewer=visualize,
        controller=controller,
        record_video=record_video,
        data_recorder=data_recorder
    )
    print("🎉 Policy rollout finished successfully.")
    
    
    

if __name__ == '__main__':
    # define policy, database path for policy rollout, and data_MPC_path for setting initial conditions
    # v_des = [0.15,0,0]
    # TODO: maybe I can store the path in a config file so that I don't need to change everytime I want to do a test
    # policy_path = "/home/atari/workspace/iterative_supervised_learning/examples/data/behavior_cloning/trot/working_policy/policy_200.pth"
    
    policy_path = "/home/atari/workspace/iterative_supervised_learning/examples/data/behavior_cloning/trot/Mar_27_2025_14_12_19/network/policy_final.pth"
    data_MPC_path = "/home/atari/workspace/iterative_supervised_learning/examples/data/behavior_cloning/trot/Mar_27_2025_14_12_19/dataset/experiment/traj_nominal_03_27_2025_14_12_25.npz"
    v_des = [0.3,0.0,0.0]
    
    # extract initial states from start time
    data_MPC = np.load(data_MPC_path)
    # start_time = 0.16
    start_time = 0.028
    q_MPC = data_MPC["q"]
    v_MPC = data_MPC["v"]
    
    q0 = q_MPC[int(start_time * 1000)]
    v0 = v_MPC[int(start_time * 1000)]
    print("current q0 from MPC recording is = ",q0)
    print("current v0 from MPC recording is = ",v0)
    initial_state = [q0,v0]
    
    # rollout policy
    rollout_policy(policy_path, 
                   sim_time=5.0, 
                   v_des = v_des, 
                   record_video=False,
                   norm_policy_input=True,
                   save_data=False,
                   initial_state = initial_state,
                   start_time = start_time,
                   data_MPC_path=data_MPC_path)
    

    
    

    