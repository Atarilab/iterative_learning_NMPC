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
from mj_pin.utils import get_robot_description   # type: ignore

from mpc_controller.mpc import LocomotionMPC
from iterative_supervised_learning.utils.network import GoalConditionedPolicyNet
import torch
from tqdm import tqdm
import threading
import mujoco
import mujoco.viewer

SIM_DT = 1.0e-3
VIEWER_DT = 1/30.
n_state = 39 # state:36 + vc_goal:3
n_action = 12
kp = 2.0
kd = 0.1 

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
        record_step: int = 1,
    ) -> None:
        """
        A simple data recorder that saves simulation data to a .npz file.
        """
        super().__init__(record_dir, record_step)
        self.data = {}
        self.reset()

    def reset(self) -> None:
        self.data = {"time": [], "q": [], "v": [], "ctrl": [],}

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
        self.data["time"].append(round(mj_data.time, 4))
        self.data["q"].append(mj_data.qpos.copy())
        self.data["v"].append(mj_data.qvel.copy())
        self.data["ctrl"].append(mj_data.ctrl.copy())

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

def rollout_policy(policy_path: str, sim_time=5.0,v_des=[0.5,0,0], gait="trot", record_video=True):
    """
    Rollout a trained policy on the Go2 robot in MuJoCo.

    Args:
        policy_path (str): Path to the trained policy model (.pth).
        sim_time (float): Duration of the rollout in seconds.
        gait (str): Gait pattern to follow (default: "trot").
        record_video (bool): Whether to record a video of the rollout.
    """

    # Load trained policy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = GoalConditionedPolicyNet(input_size=n_state, 
                                          output_size=n_action,
                                          num_hidden_layer=3,
                                          hidden_dim=512,
                                          batch_norm=True)
    policy_net.load_state_dict(torch.load(policy_path, map_location=device)['network'])
    policy_net.to(device)
    policy_net.eval()

    # Load robot description
    robot_description = get_robot_description("go2")
    sim = Simulator(robot_description.xml_scene_path, sim_dt=SIM_DT, viewer_dt=VIEWER_DT)
    
    # Set camera view
    sim.vs.track_obj = "base"
    sim.vs.set_side_view()

    # Initialize simulator
    sim.setup()
    
    # Start MuJoCo Viewer
    viewer = mujoco.viewer.launch_passive(sim.mj_model, sim.mj_data, show_left_ui=False, show_right_ui=False)
    
    # Set initial robot state
    q_init = sim.mj_data.qpos.copy()
    v_init = np.zeros_like(sim.mj_data.qvel)

    # Recorder
    recorder = StateDataRecorder()

    # Simulation rollout
    num_steps = int(sim_time / SIM_DT)
    state_history = np.zeros((num_steps, n_state))
    action_history = np.zeros((num_steps, n_action))

    print(f"Starting rollout for {sim_time} seconds ({num_steps} steps)...")
    start_time = time.time()

    for t in tqdm(range(num_steps)):
        # Get state
        q = sim.mj_data.qpos.copy()
        v = sim.mj_data.qvel.copy()
        # print("q", q)
        # input()
        # print("v ",v)
        # input()
        
        # extract base position
        base_pos =  q[:3]
        robot_state = q[2:]
        vc_goal = v_des
        cc_goal = 0
        
        phase_percentage = get_phase_percentage(t*SIM_DT)
        print(phase_percentage)
        
        state = np.concatenate([[phase_percentage],v,robot_state])[:n_state]
        state = np.concatenate([state,v_des])
        # print("state", state)
        # input()
        
        # Predict action
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        action_tensor = policy_net(state_tensor)
        action = action_tensor.cpu().detach().numpy().flatten()
        print("action",action)
        input()

        # Apply action (assuming torques)
        sim.mj_data.ctrl[:] = action

        # Step simulation
        mujoco.mj_step(sim.mj_model, sim.mj_data)

        # Record data
        state_history[t] = state
        action_history[t] = action
        recorder._record(sim.mj_data)

        # Sleep to match real-time
        time.sleep(SIM_DT)

    end_time = time.time()
    print(f"Rollout complete in {end_time - start_time:.2f} seconds.")

    # Save data
    recorder.save()

    # Save video if required
    if record_video:
        sim.save_video("./rollout_videos/")

    print("Rollout finished successfully.")
    
def rollout_policy_multithread(policy_path: str, sim_time=3.0, v_des=[0.3, 0.0, 0.0], gait="trot", record_video=False):
    """
    Rollout a trained policy on the Go2 robot in MuJoCo.

    Args:
        policy_path (str): Path to trained policy file (.pth).
        sim_time (float): Duration of the rollout in seconds.
        gait (str): Gait pattern to follow.
        record_video (bool): Whether to record a video.
    """

    # Load trained policy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = GoalConditionedPolicyNet(input_size=n_state, output_size=n_action, num_hidden_layer=3, hidden_dim=512, batch_norm=True)
    policy_net.load_state_dict(torch.load(policy_path, map_location=device)['network'])
    policy_net.to(device)
    policy_net.eval()

    # Load robot description
    robot_description = get_robot_description("go2")
    sim = Simulator(robot_description.xml_scene_path, sim_dt=SIM_DT, viewer_dt=VIEWER_DT)

    # Set camera view
    sim.vs.track_obj = "base"
    sim.vs.set_side_view()

    # Initialize simulator
    sim.setup()

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
    state_history = np.zeros((num_steps, n_state))
    action_history = np.zeros((num_steps, n_action))

    print(f"🚀 Starting rollout for {sim_time} seconds ({num_steps} steps)...")
    start_time = time.time()

    for t in tqdm(range(num_steps)):
        # Get state
        q = sim.mj_data.qpos.copy()
        v = sim.mj_data.qvel.copy()
        
        # Check for invalid states
        if np.isnan(q).any() or np.isnan(v).any() or np.isinf(q).any() or np.isinf(v).any():
            print(f"❌ Simulation failed at step {t}: NaN/Inf detected! Stopping.")
            break
        
        print("current_time = ", t*SIM_DT)
        phase_percentage = get_phase_percentage(t*SIM_DT)
        print("phase_percentage = ", phase_percentage)
        
        # Construct state vector
        state = np.concatenate([[phase_percentage],v, q[2:], v_des])[:n_state]
        print("state = ",state)
        # input()
        
        # Predict action
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        
        # action is actually a PD target
        action_tensor = policy_net(state_tensor)
        # action = action_tensor.cpu().detach().numpy().flatten()
        action = action_tensor.cpu().detach().numpy().reshape(-1)
        q_des = action
        tau = kp * (q_des - q[7:]) - kd * v[6:]
        print("action = ", action)
        print("tau = ",tau)
        # input()
        
        # Apply action
        sim.mj_data.ctrl[:] = tau

        # Step simulation
        mujoco.mj_step(sim.mj_model, sim.mj_data)

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
    policy_path = '/home/atari/workspace/iterative_supervised_learning/examples/data/behavior_cloning/trot/Feb_12_2025_22_46_48/network/policy_final.pth'
    
    
    # rollout_policy(policy_path=policy_path, sim_time=args.time, gait=args.gait, record_video=args.record_video)
    rollout_policy_multithread(policy_path=policy_path, sim_time=args.time, gait=args.gait, record_video=args.record_video)
    # rollout_policy(policy_path=args.policy, sim_time=args.time, gait=args.gait, record_video=args.record_video)