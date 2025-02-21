import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../..')

import argparse
import time
from typing import List
import pinocchio as pin
import numpy as np

from mj_pin.abstract import VisualCallback, DataRecorder # type: ignore
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
    def __init__(
        self,
        record_dir: str = "",
        record_step: int = 1) -> None:
        """
        A simple data recorder that saves simulation data to a .npz file.
        """
        super().__init__(record_dir, record_step)
        self.data = {}
        
        # some initialization
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
                     "base_wrt_feet":[]}

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
        except Exception as e:
            print(f"Error saving data: {e}")

    def record(self, mj_data) -> None:
        """
        Record simulation data at the current simulation step.
        """
        # Record time and state
        q = mj_data.qpos.copy()
        v = mj_data.qvel.copy()
        self.data["time"].append(round(mj_data.time, 4))
        self.data["q"].append(q)
        self.data["v"].append(v)
        self.data["ctrl"].append(mj_data.ctrl.copy())
        
        # # Record feet position in the world
        feet_pos_all = []
        base_wrt_feet = np.zeros(2*len(self.feet_names))
        
        for i, f_name in enumerate(self.feet_names):
            feet_pos = mj_frame_pos(self.mj_model, mj_data, f_name)
            # print("feet_pos = ",feet_pos)
            # print("base_pos = ",q[:3])
            feet_pos_all.extend(feet_pos)
            base_wrt_feet[2*i:2*i+2] = (q[:3] - feet_pos)[:2]  # Correct indexing
        
        # print("base_wrt_feet = ", base_wrt_feet)
        # input()
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

def rollout_policy_multithread(policy_path:str, sim_time = 5.0, v_des = [0.3,0.0,0.0], gait ='trot',record_video = False):
    """
    Rollout a trained policy on the Go2 robot in MuJoCo.

    Args:
        policy_path (str): Path to trained policy file (.pth).
        sim_time (float): Duration of the rollout in seconds.
        gait (str): Gait pattern to follow.
        record_video (bool): Whether to record a video.
    """
    
    # load trained policy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = GoalConditionedPolicyNet(input_size=n_state, output_size=n_action, num_hidden_layer=3, hidden_dim=512, batch_norm=True)
    policy_net.load_state_dict(torch.load(policy_path, map_location=device)['network'])
    policy_net.to(device)
    policy_net.eval()
    
    print(policy_net)

    # load robot description
    robot_description = get_robot_description("go2")
    xml_path = robot_description.xml_path
    feet_names = ["FL", "FR", "RL", "RR"]
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    
    # load simulator
    sim = Simulator(robot_description.xml_scene_path, sim_dt=SIM_DT, viewer_dt=VIEWER_DT)
    sim.vs.track_obj = "base"
    sim.vs.set_side_view()
    sim.setup()
    input()
    
    # Start MuJoCo Viewer in a separate thread
    viewer_running = [True]
    
    def viewer_thread():
        with mujoco.viewer.launch_passive(sim.mj_model, sim.mj_data, show_left_ui=False, show_right_ui=False) as viewer:
            while viewer.is_running() and viewer_running[0]:
                viewer.sync()
                time.sleep(VIEWER_DT)
    
    viewer_t = threading.Thread(target=viewer_thread, daemon=True)
    viewer_t.start()
    
    # Recorder
    recorder = StateDataRecorder()
    
    # Simulation rollout
    num_steps = int(sim_time / SIM_DT)
    state_history = np.zeros((num_steps, n_state-3))
    action_history = np.zeros((num_steps, n_action))
    
    print(f"🚀 Starting rollout for {sim_time} seconds ({num_steps} steps)...")
    start_time = time.time()
    norm_policy_input = True
    if norm_policy_input:
            # load database
            database_path = "/home/atari/workspace/iterative_supervised_learning/examples/data/behavior_cloning/trot/Feb_19_2025_14_40_59/dataset/database_0.hdf5"
            # Initialize Database with a suitable limit
            db = Database(limit=10000000,norm_input=True)  # Ensure the limit is large enough to load the full dataset
            # Load the saved database
            print(f"Loading database from: {database_path}")
            db.load_saved_database(database_path)
            mean_std = db.get_database_mean_std()
    
    # main simulation loop
    for t in tqdm(range(num_steps)):
        # Get state
        q = sim.mj_data.qpos.copy()
        v = sim.mj_data.qvel.copy()
        
        # print("q_mj is = ",q)
        
        # Check for invalid states
        if np.isnan(q).any() or np.isnan(v).any() or np.isinf(q).any() or np.isinf(v).any():
            print(f"❌ Simulation failed at step {t}: NaN/Inf detected! Stopping.")
            break
        
        ## construct input state for inference
        # extract phase percentage
        print("current_time = ", t*SIM_DT)
        phase_percentage = get_phase_percentage(t*SIM_DT)
        print("phase_percentage = ", phase_percentage)
        
        # extract robot state
        robot_state = q[2:]
        # print("current robot state is = ",robot_state)
        
        # extract base state
        base_state = q[:3]
        # print("current base state is = ", base_state)
        
        # extract feet position and base_wrt_feet
        feet_pos_all = []
        base_wrt_feet = np.zeros(2*len(feet_names))
        for i, f_name in enumerate(feet_names):
            feet_pos = mj_frame_pos(mj_model, sim.mj_data, f_name)
            # print("feet_pos = ",feet_pos)
            # print("base_pos = ", q[:3])
            feet_pos_all.extend(feet_pos)
            base_wrt_feet[2*i:2*i+2] = (q[:3] - feet_pos)[:2]
        
        # put state vector together
        state = np.concatenate(([phase_percentage], v, robot_state, base_wrt_feet))[:n_state-3]
        # print("unnormalized state ",state )
        
        ## Normalize policy input
        if norm_policy_input:
            state_mean = mean_std[0]
            state_std = mean_std[1]
            if (state_std == 0).any():  # Check if any element is zero
                state_std = 1.0
            state[1:] = (state[1:] - state_mean[1:]) / state_std[1:]  # Normalize all but the first entry    
        # print("normalized state = ", state)
        # input()
        
        # put policy network input together
        x = np.concatenate([np.array(state), np.array(v_des)])[:n_state]
        x_tensor = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
        
        # predict action
        y_tensor = policy_net(x_tensor)
        action = y_tensor.cpu().detach().numpy().reshape(-1) # action is a PD target
        
        q_des = action
        tau = kp * (q_des - q[7:]) - kd * v[6:]
        # print("current torque = ", tau)
        
        # Apply torque
        sim.mj_data.ctrl[:] = tau
        
        # Step simulation
        mujoco.mj_step(sim.mj_model, sim.mj_data)
        # input()
        
        # Record data
        state_history[t] = state
        action_history[t] = action
        recorder.record(sim.mj_data)
        
        # Sleep to match real-time
        time.sleep(SIM_DT)
    
    end_time = time.time()
    print(f"✅ Rollout complete in {end_time - start_time:.2f} seconds.")
    
    # Save data
    recorder.save()
    
    # Save video if required
    if record_video:
        sim.save_video("./rollout_videos/")

    # Stop the viewer thread
    viewer_running[0] = False
    viewer_t.join()
    
    print("🎉 Rollout finished successfully.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rollout a trained policy on the Go2 robot.")
    parser.add_argument("--policy", type=str, required=False, help="Path to trained policy (.pth)")
    parser.add_argument("--time", type=float, default=5.0, help="Simulation time in seconds")
    parser.add_argument("--gait", type=str, default="trot", help="Gait pattern (default: trot)")
    parser.add_argument("--record_video", action="store_true", help="Record rollout video")
    
    args = parser.parse_args()
    policy_path = '/home/atari/workspace/iterative_supervised_learning/examples/data/behavior_cloning/trot/Feb_21_2025_10_42_21/network/policy_final.pth'
    


    rollout_policy_multithread(policy_path=policy_path, sim_time=args.time, gait=args.gait, record_video=args.record_video)
    