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
from mj_pin.utils import get_robot_description, mj_frame_pos   # type: ignore

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
n_state = 44 # state:44 + vc_goal:3
n_state += 3
print("n_state = ",n_state)
n_action = 12
kp = 40.0
kd = 5.0 

class ReferenceVisualCallback(VisualCallback):
    def __init__(self, mpc_controller, update_step = 1):
        super().__init__(update_step)
        self.mpc = mpc_controller
        self.radius = 0.01

    def add_visuals(self, mj_data):
        # Contact locations
        for i, foot_cnt in enumerate(self.mpc.solver.dyn.feet):
            cnt_pos = self.mpc.solver.params[foot_cnt.plane_point.name]
            cnt_pos_unique = np.unique(cnt_pos, axis=1).T
            for pos in cnt_pos_unique:
                if np.sum(pos) == 0.: continue
                self.add_sphere(pos, self.radius, self.colors.id(i))

        # Base reference
        BLACK = self.colors.name("black")
        base_ref = self.mpc.solver.cost_ref[self.mpc.solver.dyn.base_cost.name][:, 0]
        R_WB = pin.rpy.rpyToMatrix(base_ref[3:6][::-1]).flatten()
        self.add_box(base_ref[:3], rot=R_WB, size=[0.08, 0.04, 0.04], rgba=BLACK)
        
        # Base terminal reference
        base_ref = self.mpc.solver.cost_ref_terminal[self.mpc.solver.dyn.base_cost.name]
        R_WB = pin.rpy.rpyToMatrix(base_ref[3:6][::-1]).flatten()
        self.add_box(base_ref[:3], rot=R_WB, size=[0.08, 0.04, 0.04], rgba=BLACK)
        
class StateDataRecorder(DataRecorder):
    def __init__(self, record_dir: str = "", record_step: int = 1) -> None:
        """
        A simple data recorder that saves simulation data to a .npz file.
        """
        super().__init__(record_dir, record_step)
        self.data = {}
        
        # Initialization of feet names and robot model
        self.feet_names = ["FL", "FR", "RL", "RR"]
        self.robot_name = "go2"
        xml_path = get_robot_description(self.robot_name).xml_path
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        
        self.reset()

    def reset(self) -> None:
        """Reset the data dictionary to initialize recording."""
        self.data = {
            "time": [], 
            "q": [], 
            "v": [], 
            "ctrl": [],
            "feet_pos_w": [],
            "base_wrt_feet": []
        }

    def save(self) -> None:
        """Save the recorded simulation data to a .npz file."""
        if not self.record_dir:
            self.record_dir = os.getcwd()
        os.makedirs(self.record_dir, exist_ok=True)

        timestamp = self.get_date_time_str()
        file_path = os.path.join(self.record_dir, f"simulation_data_{timestamp}.npz")

        try:
            np.savez(file_path, **self.data)
            print(f"✅ Data successfully saved to {file_path}")
        except Exception as e:
            print(f"❌ Error saving data: {e}")

    def record(self, mj_data) -> None:
        """Record simulation data at the current simulation step."""
        q = mj_data.qpos.copy()
        v = mj_data.qvel.copy()
        
        self.data["time"].append(round(mj_data.time, 4))
        self.data["q"].append(q)
        self.data["v"].append(v)
        self.data["ctrl"].append(mj_data.ctrl.copy())
        
        # Record feet positions and base-to-feet positions
        feet_pos_all = []
        base_wrt_feet = np.zeros(2 * len(self.feet_names))
        
        for i, f_name in enumerate(self.feet_names):
            feet_pos = mj_frame_pos(self.mj_model, mj_data, f_name)
            feet_pos_all.extend(feet_pos)
            base_wrt_feet[2 * i:2 * i + 2] = (q[:3] - feet_pos)[:2]
        
        self.data["feet_pos_w"].append(np.array(feet_pos_all))
        self.data["base_wrt_feet"].append(np.array(base_wrt_feet))

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
    def __init__(self, policy_path: str, n_state: int, n_action: int, joint_name2act_id: Dict[str, int],
                 v_des: np.ndarray = np.array([0.3, 0.0, 0.0]), norm_policy_input: bool = True,
                 database_path: str = "", device: str = "cpu"):
        super().__init__()
        self.device = device
        self.policy_net = GoalConditionedPolicyNet(input_size=n_state, output_size=n_action, num_hidden_layer=3,
                                                   hidden_dim=512, batch_norm=True)
        self.policy_net.load_state_dict(torch.load(policy_path, map_location=device)['network'])
        self.policy_net.to(device)
        self.policy_net.eval()
        self.joint_name2act_id = joint_name2act_id
        self.v_des = v_des
        self.n_state = n_state
        self.norm_policy_input = norm_policy_input
        self.mean_std = None
        self.robot_name = "go2"
        xml_path = get_robot_description(self.robot_name).xml_path
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)

        if norm_policy_input and database_path:
            db = Database(limit=10000000, norm_input=True)
            db.load_saved_database(database_path)
            self.mean_std = db.get_database_mean_std()

    def get_torques(self, step: int, mj_data) -> Dict[str, float]:
        q = mj_data.qpos.copy()
        v = mj_data.qvel.copy()
        # phase_percentage
        phase_percentage = get_phase_percentage(step * SIM_DT)
        
        # robot_state
        robot_state = q[2:]
        
        # base with right to feet
        base_wrt_feet = self._get_base_wrt_feet(mj_data)
        
        # combine state variable
        state = np.concatenate(([phase_percentage], v, robot_state, base_wrt_feet))[:self.n_state-3]
        
        # normalize state without phase percentage
        if self.norm_policy_input and self.mean_std is not None:
            state_mean, state_std = self.mean_std
            state[1:] = (state[1:] - state_mean[1:]) / np.where(state_std[1:] == 0, 1.0, state_std[1:])

        # form policy input
        x = np.concatenate([np.array(state), np.array(self.v_des)])[:self.n_state]
        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # get policy output
        y_tensor = self.policy_net(x_tensor)
        action = y_tensor.cpu().detach().numpy().reshape(-1)

        # calculate torque based on PD target
        tau = kp * (action - q[7:]) - kd * v[6:]
        
        # distribute torque to corresponding joints
        return self.create_torque_map(tau)

    def _get_base_wrt_feet(self, mj_data) -> np.ndarray:
        # get base with right to feet position (x,y)*4
        feet_names = ["FL", "FR", "RL", "RR"]
        base_wrt_feet = np.zeros(2 * len(feet_names))
        
        for i, f_name in enumerate(feet_names):
            feet_pos = mj_frame_pos(self.mj_model, mj_data, f_name)  # Add self.mj_model as the first argument
            base_wrt_feet[2 * i:2 * i + 2] = (mj_data.qpos[:3] - feet_pos)[:2]
        return base_wrt_feet

    def create_torque_map(self, torques: np.ndarray) -> Dict[str, float]:
        return {
            j_name: float(torques[joint_id])
            for j_name, joint_id in self.joint_name2act_id.items()
            if joint_id < len(torques)
        }

def rollout_policy(
    policy_path: str,
    robot_name: str = "go2",
    sim_time: float = 5.0,
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
    sim.vs.set_high_quality()
    sim.setup()
    sim._init_model_data()

    controller = PolicyController(
        policy_path=policy_path,
        n_state=n_state,
        n_action=n_action,
        joint_name2act_id=sim.joint_name2act_id,
        v_des=v_des,
        norm_policy_input=norm_policy_input,
        database_path=database_path,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    sim.run(
        sim_time=sim_time,
        use_viewer=visualize,
        controller=controller,
        record_video=record_video,
        data_recorder=StateDataRecorder(record_dir) if save_data else None
    )
    print("🎉 Policy rollout finished successfully.")
    
    
    

if __name__ == '__main__':
    policy_path = '/home/atari/workspace/iterative_supervised_learning/examples/data/policy_final.pth'
    rollout_policy(policy_path, sim_time=5.0, v_des=[0.3, 0.0, 0.0], record_video=False)
    

    