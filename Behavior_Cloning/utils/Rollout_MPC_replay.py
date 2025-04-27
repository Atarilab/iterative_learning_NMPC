# This script is to check if recorded MPC data replays in Mujoco

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

from Behavior_Cloning.utils.network import GoalConditionedPolicyNet
from Behavior_Cloning.utils.database import Database

import torch
from tqdm import tqdm
import threading
import mujoco
import mujoco.viewer

import matplotlib.pyplot as plt

# define global variables
SIM_DT = 1.0e-3
VIEWER_DT = 1/30.
t0 = 0.028
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
        
        # print("current action is = ",action)
        self.data["action"].append(np.array(action))
        
        # record the velocity conditioned goals
        self.data["vc_goals"].append(self.vc_goals)
        
        # record contact conditioned goals(currently just a random noise)
        self.cc_goals = np.random.normal(loc=0.0, scale=0.1, size=(8,))
        self.data["cc_goals"].append(self.cc_goals)

class ReplayController(Controller):
    def __init__(self,
                 data_path: str,
                 joint_name2act_id: Dict[str, int],
                 v_des: np.ndarray = np.array([0.3, 0.0, 0.0]),
                 mj_model = None,
                 start_time = 0.0
                 ):
        super().__init__()
        
        # initialization 
        self.joint_name2act_id = joint_name2act_id
        self.v_des = v_des
        self.data_path = data_path
        self.nu = 18
        self.start_time = start_time
        self.ctrl_index = 0
        
        # initialize robot description
        self.robot_name = "go2"
        self.mj_model = mj_model
        
        # extract necessary data from data_path
        data = np.load(self.data_path)
        self.action_history = data["action"]
        self.ctrl_history = data["ctrl"] # in the order of [FR,FL,RR,RL]
        self.q_his = data["q"]
        self.v_his = data["v"]
        self.feet_pos_his = data["feet_pos_w"]
        self.base_wrt_feet_his = data["base_wrt_feet"]
    
    def compute_torques_dof(self, mj_data):
        # get current time
        # current_time = np.round(mj_data.time + self.start_time,4)
        
        # direct apply torque from file
        tau_frflrrrl_MPC = self.ctrl_history[self.ctrl_index]
        FR_torque = tau_frflrrrl_MPC[0:3]
        FL_torque = tau_frflrrrl_MPC[3:6]
        RR_torque = tau_frflrrrl_MPC[6:9]
        RL_torque = tau_frflrrrl_MPC[9:]
        tau_flfrrlrr_direct = np.concatenate([FL_torque,FR_torque,RL_torque,RR_torque])
        
        # apply PD target from file and calculate torque to replay
        pd_target = self.action_history[self.ctrl_index]
        q = mj_data.qpos.copy()
        v = mj_data.qvel.copy()
        tau_from_PD_target = kp * (pd_target - q[7:]) - kd * v[6:]  # PD control
        self.ctrl_index += 1
        # copy torque to mj_data
        self.torques_dof = np.zeros(self.nu)
        # self.torques_dof[-12:] = tau_flfrrlrr_direct
        self.torques_dof[-12:] = tau_from_PD_target
        
        # for debugging
        print(f"current time {self.ctrl_index * SIM_DT}: Applied control torques (high precision): {self.torques_dof}")
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

def rollout_replay(
    data_path : str,
    robot_name: str = "go2",
    sim_time: float = 2.0,
    v_des: np.ndarray = np.array([0.3, 0.0, 0.0]),
    record_video: bool = False,
    save_data: bool = True,
    record_dir: str = "./data/",
    visualize: bool = True,
    start_time: float = 0.0
):
    # setup robot description
    robot_desc = get_robot_description(robot_name)
    
    # setup simulator
    sim = Simulator(robot_desc.xml_scene_path, sim_dt=SIM_DT, viewer_dt=VIEWER_DT)
    sim.vs.track_obj = "base"
    sim.setup()
    # sim.set_initial_state(q0,v0)
    
    # setup joint_name2act
    joint_name2act_id= mj_joint_name2dof(sim.mj_model)
    print("Joint to Actuator ID Mapping:", joint_name2act_id)
    
    # setup controller
    controller = ReplayController(
        data_path=data_path,
        joint_name2act_id = joint_name2act_id,
        v_des = v_des,
        mj_model = sim.mj_model
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
    # define the MPC recording you want to replay
    data_path = "/home/atari/workspace/Behavior_Cloning/examples/data/example_traj_smoothed.npz"
    rollout_replay(
        data_path = data_path,
        sim_time = 4.0,
        v_des = [0.15,0.0,0.0],
        save_data=True,
        start_time=0.0)