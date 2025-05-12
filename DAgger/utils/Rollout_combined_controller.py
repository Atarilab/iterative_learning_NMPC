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
from DAgger.utils.RolloutMPC import rollout_mpc_phase_percentage_shift
import scipy.spatial.transform as st
import mujoco 
import matplotlib.pyplot as plt
import time
import torch

# global variables
SIM_DT = 1.0e-3
VIEWER_DT = 1/30.
gait_period = 0.5 # trotting
kp = 20
kd = 1.5

# VisualCallback
class ReferenceVisualCallback(VisualCallback):
    def __init__(self, combined_controller, simulator=None, update_step=1):
        super().__init__(update_step)
        self.controller = combined_controller
        self.mpc = self.controller.mpc_controller
        self.simulator = simulator
        self.radius = 0.01

    def add_visuals(self, mj_data):
        # === Contact points
        for i, foot_cnt in enumerate(self.mpc.solver.dyn.feet):
            cnt_pos = self.mpc.solver.params[foot_cnt.plane_point.name]
            cnt_pos_unique = np.unique(cnt_pos, axis=1).T
            for pos in cnt_pos_unique:
                if np.sum(pos) == 0.:
                    continue
                self.add_sphere(pos, self.radius, self.colors.id(i))

       # Use color to indicate which controller is active
        if self.controller.control_mode == "mpc":
            rgba = self.colors.name("black")  # Expert: MPC
        else:
            rgba = self.colors.name("blue")   # Learner: policy

        # Current MPC reference
        base_ref = self.mpc.solver.cost_ref[self.mpc.solver.dyn.base_cost.name][:, 0]
        self.add_box(base_ref[:3], rot_euler=base_ref[3:6][::-1], size=[0.08, 0.04, 0.04], rgba=rgba)

        # Terminal MPC reference (optional, could remain black)
        term_ref = self.mpc.solver.cost_ref_terminal[self.mpc.solver.dyn.base_cost.name]
        self.add_box(term_ref[:3], rot_euler=term_ref[3:6][::-1], size=[0.08, 0.04, 0.04],  rgba=rgba)



# State Data Recorder
class StateDataRecorder(DataRecorder):
    def __init__(
        self,
        record_dir: str = "",
        record_step: int = 1,
        v_des: np.ndarray = np.array([0,0,0]),
        current_time: float = 0.0,
        nominal_flag = True,
        replanning_point = 0,
        nth_traj_per_replanning = 0) -> None:
        """
        A simple data recorder that saves simulation data to a .npz file.
        """
        super().__init__(record_dir, record_step)
        self.data = {}
        self.vc_goals = v_des
        self.cc_goals = np.random.normal(loc=0.0, scale=0.1, size=(8,))
        self.current_time = current_time
        self.nominal_flag = nominal_flag
        self.replanning_point = replanning_point
        self.nth_traj_per_replanning_point = nth_traj_per_replanning
        
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
                     "cc_goals":[],
                     "contact_vec":[],
                     "is_expert": []}

    def save(self) -> None:
        if not self.record_dir:
            self.record_dir = os.getcwd()
        os.makedirs(self.record_dir, exist_ok=True)
        timestamp = self.get_date_time_str()

        if self.nominal_flag:
            file_path = os.path.join(self.record_dir, f"traj_nominal_{timestamp}.npz")
        else:
            file_path = os.path.join(self.record_dir, f"traj_{self.replanning_point}_{self.nth_traj_per_replanning_point}.npz")

        try:
            # Uncomment to save data
            np.savez(file_path, **self.data)
            print(f"Data successfully saved to {file_path}")
            return file_path
        except Exception as e:
            print(f"Error saving data: {e}")
            return ""
    
    def record(self, mj_data,is_expert = 0) -> None:
        """
        Record simulation data at the current simulation step.
        """
        # Record time and state
        q = mj_data.qpos.copy()
        v = mj_data.qvel.copy()
        self.data["time"].append(round(mj_data.time + self.current_time, 4))
        # print("current time is = ",round(mj_data.time + self.current_time, 4))
        
        self.data["q"].append(q) # in the order of [FL,FR,RL,RR]
        self.data["v"].append(v) # in the order of [FL,FR,RL,RR]
        self.data["ctrl"].append(mj_data.ctrl.copy()) # in the order of [FR,FL,RR,RL]
        
        # Record feet position in the world (x,y,z)
        feet_pos_all = []
        ee_in_contact = []
        base_wrt_feet = np.zeros(2*len(self.feet_names))
        
        for i, f_name in enumerate(self.feet_names):
            feet_pos = mj_frame_pos(self.mj_model, mj_data, f_name)
            feet_pos_all.extend(feet_pos)
            base_wrt_feet[2*i:2*i+2] = (q[:3] - feet_pos)[:2]
        
        self.data["feet_pos_w"].append(np.array(feet_pos_all))
        
        # base with right to feet in world frame
        self.data["base_wrt_feet"].append(np.array(base_wrt_feet))
        
        geom_to_frame_name = {
            20: "FL_foot",
            32: "FR_foot",
            44: "RL_foot",
            56: "RR_foot"
        }

        ee_in_contact = get_feet_in_contact_by_id(mj_data, geom_to_frame_name)
        
        contact_vec = np.array([
            int("FL_foot" in ee_in_contact),
            int("FR_foot" in ee_in_contact),
            int("RL_foot" in ee_in_contact),
            int("RR_foot" in ee_in_contact)
        ])
        self.data["contact_vec"].append(contact_vec)
        
        ## form state variable
        # the format of state = [[phase_percentage],v,q[2:],base_wrt_feet]
        # if in replanning step, phase percentage is not starting from 0
        phase_percentage = np.round([get_phase_percentage(mj_data.time + self.current_time)], 4)
        # phase_percentage = np.round([get_phase_percentage(mj_data.time)], 4)
        #==========================================================================================
        # state with base_wrt_feet
        state = np.concatenate([phase_percentage, v, q[2:], base_wrt_feet])
        
        # # state without base_wrt_feet
        # state = np.concatenate([phase_percentage, v, q[2:]])
        
        self.data["state"].append(np.array(state)) # here is unnormalized state
        #=========================================================================================
        
        
        # transform action from torque to PD target and store
        tau_frflrrrl = mj_data.ctrl.copy() # in the order of [FR,FL,RR,RL]
        FR_torque = tau_frflrrrl[0:3]
        FL_torque = tau_frflrrrl[3:6]
        RR_torque = tau_frflrrrl[6:9]
        RL_torque = tau_frflrrrl[9:]
        tau_flfrrlrr = np.concatenate([FL_torque,FR_torque,RL_torque,RR_torque])
        # print("tau is = ", tau_frflrrrl)
        # print("FR torque is ")
        # print(FR_torque)
        # print("FL torque is ")
        # print(FL_torque)
        # print("RR torque is ")
        # print(RR_torque)
        # print("RL torque is ")
        # print(RL_torque)
        # print("transformed tau is = ")
        # print(tau_flfrrlrr)
        # input()
        
        action = (tau_flfrrlrr + kd * v[6:])/kp + q[7:] # in the order of [FL,FR,RL,RR]
        # print("current action is = ",action)
        self.data["action"].append(np.array(action))
        
        # record the velocity conditioned goals
        self.data["vc_goals"].append(self.vc_goals)
        
        # record contact conditioned goals(currently just a random noise)
        self.cc_goals = np.random.normal(loc=0.0, scale=0.1, size=(8,))
        self.data["cc_goals"].append(self.cc_goals)
        
        # record expert flag
        self.data["is_expert"].append(is_expert)

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

def get_feet_in_contact_by_id(mj_data, geom_to_frame_name: Dict[int, str], ground_geom_id: int = 0) -> List[str]:
    """
    Get the list of feet in contact, returning Pinocchio frame names (e.g., 'FL_foot').

    Args:
        mj_data: MuJoCo mjData.
        geom_to_frame_name: Dictionary mapping geom_id (int) to Pinocchio frame name (str).
        ground_geom_id (int): The geom ID of the ground.

    Returns:
        List[str]: Frame names of feet in contact (e.g., ['FL_foot', 'RR_foot']).
    """
    contact_feet = []

    for i in range(mj_data.ncon):
        contact = mj_data.contact[i]
        g1, g2 = contact.geom1, contact.geom2

        if g1 == ground_geom_id and g2 in geom_to_frame_name:
            contact_feet.append(geom_to_frame_name[g2])
        elif g2 == ground_geom_id and g1 in geom_to_frame_name:
            contact_feet.append(geom_to_frame_name[g1])

    return contact_feet

# define combined controller
class CombinedController(Controller):
    def __init__(self, 
                 mpc_controller: None,
                 policy_controller: None, 
                 joint_name2act_id: Dict[str, int],
                 v_des: np.ndarray = np.array([0.3, 0.0, 0.0]),
                 mj_model = None,
                 control_mode: str = "policy",
                 nu: int = 12) -> None:
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
        
        # about switching
        self.mpc_active_counter = 0
        self.mpc_min_steps = 2500  # minimum steps to stay in MPC (0.2s if dt=0.001s)
        # the first 50 timesteps let policy take over
        self.step_counter = 0
        self.delay_steps = 100
    
    def check_unsafe_state_v1(self, mj_data):
        """Check if robot is in unsafe/fall-prone or stall-prone state."""
        # --- Access base orientation ---
        base_quat = mj_data.qpos[3:7]
        base_rotmat = pin.Quaternion(base_quat[0], base_quat[1], base_quat[2], base_quat[3]).toRotationMatrix()
        rpy = pin.rpy.matrixToRpy(base_rotmat)
        roll, pitch, yaw = rpy

        # --- Access base height ---
        base_height = mj_data.qpos[2]

        # --- Access base angular velocity ---
        base_ang_vel = mj_data.qvel[3:6]

        # --- Access base linear velocity ---
        base_lin_vel = mj_data.qvel[0:3]  # Linear part

        # --- Thresholds ---
        roll_thresh = np.deg2rad(30)   # 30 degrees
        pitch_thresh = np.deg2rad(10)  # 30 degrees
        height_lower_bound = 0.18           # meters
        height_upper_bound = 0.45          # meters
        ang_vel_thresh = 5.0           # rad/s
        stall_vel_thresh = 0.015        # m/s, if lower than this = stall
        stall_time_window = 0.2        # s (not used yet, can improve)

        # --- Check dynamic instability ---
        fall_detected = (
            abs(roll) > roll_thresh or
            abs(pitch) > pitch_thresh or
            base_height < height_lower_bound or
            base_height > height_upper_bound or
            np.linalg.norm(base_ang_vel) > ang_vel_thresh
        )

        # --- Check stall (robot commanded to move, but no motion) ---
        commanded_forward = abs(self.v_des[0]) > 0.05  # e.g., 0.05 m/s
        actual_forward = base_lin_vel[0]
        stall_detected = commanded_forward and abs(actual_forward) < stall_vel_thresh

        # --- Final unsafe detection ---
        unsafe = fall_detected or stall_detected

        # --- Print debug info ---
        print(f"roll (deg): {np.rad2deg(roll):.2f}, pitch (deg): {np.rad2deg(pitch):.2f}")
        print(f"base_height: {base_height:.3f}")
        print(f"base_ang_vel norm: {np.linalg.norm(base_ang_vel):.3f}")
        print(f"base_lin_vel x: {actual_forward:.3f}")
        print(f"fall_detected: {fall_detected}, stall_detected: {stall_detected}")
        print(f"unsafe: {unsafe}")

        return unsafe

    def check_unsafe_state_dummy(self, mj_data):
        """Hard-coded switch to MPC after 2.0 seconds."""
        sim_time = mj_data.time  # get simulation time from MuJoCo data
        if sim_time > 2.0:
            return True  # force switch to MPC
        else:
            return False  # stay with policy

    def check_unsafe_state_v2(self, mj_data):
        """Check if robot is in unsafe/fall-prone or joint-limit-violating state."""
        q = mj_data.qpos
        v = mj_data.qvel

        # --- Extract base pose ---
        base_quat = q[3:7]
        base_rotmat = pin.Quaternion(*base_quat).toRotationMatrix()
        rpy = pin.rpy.matrixToRpy(base_rotmat)
        roll, pitch, _ = rpy
        base_height = q[2]

        # --- Thresholds ---
        roll_thresh = np.deg2rad(25)
        pitch_thresh = np.deg2rad(25)
        height_bounds = (0.18, 0.45)
        vel_tracking_tol = 0.10 

        # --- Joint Limits from Table ---
        joint_deg = np.rad2deg(q[7:])  # convert joint positions to degrees

        joint_names = [
            "FL_hip", "FL_thigh", "FL_knee",
            "FR_hip", "FR_thigh", "FR_knee",
            "RL_hip", "RL_thigh", "RL_knee",
            "RR_hip", "RR_thigh", "RR_knee"
        ]

        joint_bounds= {
            # Hip abduction/adduction (hip roll)
            "FL_hip": (-70, 70),   # aggressive range observed
            "FR_hip": (-70, 70),
            "RL_hip": (-70, 70),
            "RR_hip": (-70, 70),

            # Hip flexion/extension (hip pitch)
            "FL_thigh": (25, 115),
            "FR_thigh": (25, 115),
            "RL_thigh": (25, 115),
            "RR_thigh": (25, 115),

            # Knee flexion/extension (knee pitch)
            "FL_knee": (-155, -60),
            "FR_knee": (-155, -60),
            "RL_knee": (-155, -60),
            "RR_knee": (-155, -60),
        }

        # --- Check base pose ---
        unsafe_pose = (
            abs(roll) > roll_thresh or
            abs(pitch) > pitch_thresh or
            base_height < height_bounds[0] or base_height > height_bounds[1]
        )

        # --- Check joint limits ---
        joint_violation = False
        for i, name in enumerate(joint_names):
            lower, upper = joint_bounds[name]
            if not (lower <= joint_deg[i] <= upper):
                print(f"⚠️ Joint {name} out of bounds: {joint_deg[i]:.2f} deg (limit: {lower}, {upper})")
                joint_violation = True
        
        # --- Base velocity tracking ---
        base_lin_vel_xy = v[0:2]         # only vx, vy
        goal_vel_xy = self.v_des[0:2]    # only vx_des, vy_des
        vel_error = np.abs(base_lin_vel_xy - goal_vel_xy)
        unsafe_vel_tracking = np.any(vel_error > vel_tracking_tol)

        # --- Final check ---
        unsafe = unsafe_pose or joint_violation or unsafe_vel_tracking

        # --- Debug ---
        print(f"Base height: {base_height:.3f} | Roll: {np.rad2deg(roll):.2f}°, Pitch: {np.rad2deg(pitch):.2f}°")
        print(f"Unsafe base: {unsafe_pose}, Unsafe joints: {joint_violation}")
        print(f"Base vel [vx, vy]: {base_lin_vel_xy}, Goal: {goal_vel_xy}, Error: {vel_error}")
        print(f"Unsafe velocity tracking: {unsafe_vel_tracking}")
        print(f"Unsafe: {unsafe}")

        return unsafe

        
    def set_current_control_mode(self, mj_data):
        self.step_counter += 1

        if self.step_counter < self.delay_steps:
            # Always run policy for initial N steps
            print("[INFO] Running POLICY controller for initial phase.")
            self.control_mode = "policy"
            return
        
        if self.control_mode == "mpc":
            # If already in MPC, stay at least for mpc_min_steps
            self.mpc_active_counter += 1
            if self.mpc_active_counter < self.mpc_min_steps:
                # Force continue using MPC
                return
            else:
                # After minimum time, allow normal switching
                unsafe = self.check_unsafe_state_v2(mj_data)
                if unsafe:
                    self.control_mode = "mpc"
                    # No change -> no need to pause
                else:
                    print("[INFO] Switching back to POLICY controller.")
                    # input("[PAUSE] Press Enter to continue...")
                    self.control_mode = "policy"
                    self.mpc_active_counter = 0  # Reset counter when returning to policy
        else:
            # Currently in policy mode
            unsafe = self.check_unsafe_state_v2(mj_data)
            if unsafe:
                print("[INFO] Switching to MPC controller.")
                input("[PAUSE] Press Enter to continue...")
                self.control_mode = "mpc"
                self.mpc_active_counter = 0  # Reset counter when entering MPC

    def compute_torques_dof(self, mj_data):
        # Always compute both controllers in background
        self.policy_controller.compute_torques_dof(mj_data)
        self.mpc_controller.compute_torques_dof(mj_data)  # Always update MPC plan!

        # Decide which torques to apply
        self.set_current_control_mode(mj_data)

        if self.control_mode == "policy":
            self.mpc_active = False
            self.policy_active = True
            self.torques_dof = self.policy_controller.torques_dof.copy()
        else:
            self.mpc_active = True
            self.policy_active = False
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
    v_des: np.ndarray = np.array([0.0, 0.0, 0.0]),
    record_video: bool = False,
    visualize: bool = True,
    save_data: bool = True,
    record_dir: str = "./data/",
    interactive: bool = False,
    
    # policy related
    policy_path: str = "",
    reference_mpc_path: str = "",
    norm_policy_input: bool = True,
    n_state = 47, # state:44 + vc_goal:3
    n_action = 12, # action:4*3  
    
    control_mode: str = "policy",
    nominal_flag = True,
):
    
    # define robot related parameters
    robot_desc = get_robot_description(robot_name)
    feet_frame_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    feet_name = ["FL", "FR", "RL", "RR"]
    
    # define mpc related parameters
    # interactive = False
    
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
    
    # setup data recorder
    # for now
    replanning_point = 0
    nth_traj_per_replanning = 0
    data_recorder = StateDataRecorder(record_dir,
                                      v_des=v_des,
                                      current_time = start_time,
                                      nominal_flag = nominal_flag,
                                      replanning_point = replanning_point,
                                      nth_traj_per_replanning = nth_traj_per_replanning) if save_data else None
    
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
        v_des=v_des,
    )
    print("Combined controller initialized")
    
    # setup visual callback
    vis_feet_pos = ReferenceVisualCallback(combined_controller, simulator=sim)
    
    sim.run(
        sim_time=sim_time,
        use_viewer=visualize,
        controller=combined_controller,
        record_video=record_video,
        data_recorder=data_recorder,
        visual_callback=vis_feet_pos,
    )
    print("🎉 Policy rollout finished successfully.")
    
    
if __name__ == "__main__":
    # define robot related parameters
    robot_name = "go2"
    control_mode = "policy"
    
    # simulator related parameters
    sim_time = 20.0
    start_time = 0.0
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
    
    
    
    
    