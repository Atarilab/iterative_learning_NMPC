import sys
import os

from typing import Tuple, List, Dict
import pinocchio as pin
import numpy as np
from mj_pin.abstract import VisualCallback, DataRecorder, Controller
from mj_pin.simulator import Simulator  # type: ignore
from mj_pin.utils import get_robot_description, mj_frame_pos, mj_joint_name2act_id, mj_joint_name2dof 

from mpc_controller.mpc import LocomotionMPC
from DAgger.utils.RolloutPolicy import PolicyController, rollout_policy
from DAgger.utils.RolloutMPC_shift_phase_percentage import rollout_mpc_phase_percentage_shift
import scipy.spatial.transform as st
import mujoco 
import matplotlib.pyplot as plt
import time
import torch

# global variables
SIM_DT = 1.0e-3
VIEWER_DT = 1/30.
gait_period = 0.5 # trotting

# VisualCallback

# State Data Recorder

# Customized functions
def get_phase_percentage(t:int):
    """get current gait phase percentage based on gait period

    Args:
        t (int): current sim step (NOT sim time in seconds!)

    Returns:
        phi: current gait phase. between 0 - 1
    """ 
    # get rid of phase percentage
    return 0

    # # for trot
    # gait_period = 0.5
    # if t < t0:
    #     return 0
    # else:
    #     phi = ((t-t0) % gait_period)/gait_period
    #     return phi

# define combined controller
class CombinedController(Controller):
    def __init__(self, 
                 mpc_controller: None,
                 policy_controller: None, 
                 joint_name2act_id: Dict[str, int],
                 v_des: np.ndarray = np.array([0.3, 0.0, 0.0]),
                 mj_model = None,
                 control_mode: str = "policy",
                 nu: int = 12):
        super().__init__()
        
        # initialization
        self.joint_name2act_id = joint_name2act_id
        self.v_des = v_des
        self.mpc_controller = mpc_controller
        self.policy_controller = policy_controller
        self.mpc_active = False
        self.policy_active = True
        self.control_mode = control_mode
        
        # initialize robot description
        self.robot_name = "go2"
        self.mj_model = mj_model
        self.robot_desc = get_robot_description(self.robot_name)
        self.feet_frame_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        
        self.nu = nu
    
    def check_unsafe_state(self, mj_data):
        # check if the robot is in an unsafe state to use mpc
        pass
    
    def set_current_control_mode(self, control_mode: str):
        if self.check_unsafe_state():
            # mpc takes over
            pass
        else:
            # policy takes over
            pass
        pass
    
    def compute_torques_dof(self, mj_data):
        if self.control_mode == "policy":
            # use policy
            self.mpc_active = False
            self.policy_active = True
            print("Using Policy controller")
            self.policy_controller.compute_torques_dof(mj_data)
            self.torques_dof = self.policy_controller.torques_dof.copy()

        else:
            # use mpc
            self.mpc_active = True
            self.policy_active = False
            print("Using MPC controller")
            self.mpc_controller.compute_torques_dof(mj_data)
            self.torques_dof = self.mpc_controller.torques_dof.copy()

        
    def get_torque_map(self) -> Dict[str, float]:
        # print(self.joint_name2act_id)
        torque_map = {
            j_name: self.torques_dof[dof_id]
            for j_name, dof_id in self.joint_name2act_id.items()
        }
        # print("current torque map is = ", torque_map)
        return torque_map
        
# define rollout function

def rollout_combined_controller(
    # robot related
    robot_name: str = "go2",
    
    # simulator related
    sim_time: float = 2.0,
    start_time: float = 0.0,
    initial_state = [],
    v_des: np.ndarray = np.array([0.3, 0.0, 0.0]),
    record_video: bool = False,
    visualize: bool = True,
    save_data: bool = True,
    record_dir: str = "./data/",
    interactive: bool = False,
    # mpc related
    
    # policy related
    policy_path: str = "",
    reference_mpc_path: str = "",
    norm_policy_input: bool = True,
    n_state = 47, # state:44 + vc_goal:3
    n_action = 12, # action:4*3  
    
    control_mode: str = "policy",
):
    
    # define robot related parameters
    robot_desc = get_robot_description(robot_name)
    feet_frame_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    feet_name = ["FL", "FR", "RL", "RR"]
    
    # define mpc related parameters
    interactive = False
    
    # define policy related parameters
    
    # setup simulator
    sim = Simulator(robot_desc.xml_scene_path, 
                    sim_dt=SIM_DT, 
                    viewer_dt=VIEWER_DT)
    
    # define simulator related parameters
    sim.vs.track_obj = "base"
    sim.apply_force = False
    sim.force_schedules = []
    sim.setup()
    
    if initial_state:
        q0 = initial_state[0]
        v0 = initial_state[1]
        sim.set_initial_state(q0,v0)
        
    joint_name2act_id= mj_joint_name2dof(sim.mj_model)
    print("Joint to Actuator ID Mapping:", joint_name2act_id)
    
    
    # setup visual callback
    
    # setup data recorder
    
    # setup mpc controller
    mpc_controller = LocomotionMPC(
            path_urdf=robot_desc.urdf_path,
            feet_frame_names=feet_frame_names,
            robot_name = robot_name,
            joint_ref=robot_desc.q0,
            interactive_goal=interactive,
            sim_dt=SIM_DT,
            print_info=False,
            solve_async=True,
        )
    if not interactive:
        mpc_controller.set_command(v_des,0.0)
        print("MPC controller initialized")
    
    
    # setup policy controller
    policy_controller = PolicyController(
        policy_path=policy_path,
        n_state=n_state,
        n_action=n_action,
        joint_name2act_id=joint_name2act_id,
        v_des=v_des,
        norm_policy_input=norm_policy_input,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        mj_model=sim.mj_model,
        start_time = start_time,
        reference_path= reference_mpc_path
    )
    print("Policy controller initialized")
    
    # setup combined controller
    combined_controller = CombinedController(
        control_mode=control_mode,
        mpc_controller = mpc_controller,
        policy_controller = policy_controller,
        joint_name2act_id=joint_name2act_id,
        mj_model=sim.mj_model,
        nu = n_action,
    )
    print("Combined controller initialized")
    
    sim.run(
        sim_time=sim_time,
        use_viewer=visualize,
        controller=combined_controller,
        record_video=record_video,
    )
    print("🎉 Policy rollout finished successfully.")
    
    
if __name__ == "__main__":
    # define robot related parameters
    robot_name = "go2"
    control_mode = "mpc"
    
    # simulator related parameters
    sim_time = 10.0
    start_time = 1.0
    initial_state = []
    v_des = np.array([0.15, 0.0, 0.0])
    record_video = False
    visualize = True
    save_data = True
    record_dir = "./data/"
    interactive = False
    
    # mpc related parameters
    
    # policy related parameters
    policy_path = "/home/atari/workspace/Behavior_Cloning/examples/data/behavior_cloning/trot/Apr_16_2025_13_02_09/network/policy_400.pth"
    reference_mpc_path = "/home/atari/workspace/Behavior_Cloning/examples/data/behavior_cloning/trot/Apr_16_2025_13_02_09/dataset/experiment/traj_nominal_04_16_2025_13_02_15.npz"
    data = np.load(reference_mpc_path)
    q0 = data["q"][int(start_time * 1000)]
    v0 = data["v"][int(start_time * 1000)]
    initial_state = [q0, v0]
    
    # call rollout function
    rollout_combined_controller(
        control_mode=control_mode,
        robot_name=robot_name,
        sim_time=sim_time,
        start_time=start_time,
        initial_state=initial_state,
        v_des=v_des,
        record_video=record_video,
        visualize=visualize,
        save_data=save_data,
        record_dir=record_dir,
        interactive=interactive,
        policy_path=policy_path,
        reference_mpc_path=reference_mpc_path
    )
    
    
    
    
    