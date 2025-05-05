import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) )
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from typing import Tuple, List, Dict
import pinocchio as pin
import numpy as np
from mj_pin.abstract import VisualCallback, DataRecorder
from mj_pin.simulator import Simulator  # type: ignore
from mj_pin.utils import get_robot_description, mj_frame_pos # type: ignore
from mpc_controller.mpc import LocomotionMPC
import scipy.spatial.transform as st
import mujoco 
import matplotlib.pyplot as plt
import time

# initialize global variables
SIM_DT = 1.0e-3
VIEWER_DT = 1/30.
gait_period = 0.5 # trotting

# for phase percentage shift
# t0 = 0.028
t0 = 0.0

# with base_wrt_feet
n_state = 44

nq = 19
nv = 17
kp = 20.0
kd = 1.5

# initialize pertubation variables
mu_base_pos = 0.0
sigma_base_pos = 0.1
mu_joint_pos = 0.0
sigma_joint_pos = 0.5
mu_base_ori = 0.0
sigma_base_ori = 0.7
mu_vel = 0.0
sigma_vel = 1.5

# VisualCallback
class ReferenceVisualCallback(VisualCallback):
    def __init__(self, mpc_controller, simulator=None, update_step=1):
        super().__init__(update_step)
        self.mpc = mpc_controller
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

        # === Base reference (current and terminal)
        BLACK = self.colors.name("black")
        base_ref = self.mpc.solver.cost_ref[self.mpc.solver.dyn.base_cost.name][:, 0]
        self.add_box(base_ref[:3], rot_euler=base_ref[3:6][::-1], size=[0.08, 0.04, 0.04], rgba=BLACK)

        base_ref = self.mpc.solver.cost_ref_terminal[self.mpc.solver.dyn.base_cost.name]
        self.add_box(base_ref[:3], rot_euler=base_ref[3:6][::-1], size=[0.08, 0.04, 0.04], rgba=BLACK)

        # === 🔴 Force vector visualization ===
        if self.simulator and self.simulator.visualize_force and self.simulator._force_body_id >= 0:
            fx = self.simulator.force_vec[:3]
            p0 = mj_data.xpos[self.simulator._force_body_id]
            norm = np.linalg.norm(fx)
            # print(fx)
            # print(p0)
            # print(self.simulator._force_body_id)
            # input()
            if norm > 1e-6:
                scale = .5  # visualization scale factor
                p1 = p0 + scale * fx / norm
                direction = p1 - p0
                length = np.linalg.norm(direction)
                mid = (p0 + p1) / 2
                z = direction / (length + 1e-8)

                # Orthonormal frame construction for capsule orientation
                x = np.array([1, 0, 0]) if abs(z[0]) < 0.99 else np.array([0, 1, 0])
                y = np.cross(z, x); y /= np.linalg.norm(y)
                x = np.cross(y, z)
                R = np.stack([x, y, z], axis=-1)

                self._add_geom(
                    geom_type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                    pos=mid,
                    rot=R,
                    size=[0.01, length / 2, 0],  # radius, half-length
                    rgba=[1, 0, 0, 1],  # red
                )
        
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
    # if t<t0:
    #     return 0
    # else:
    #     phi = ((t-t0) % gait_period)/gait_period
    #     return phi

def rotate_jacobian(controller, jac, index):
    """change jacobian frame

    Args:
        sim (_type_): simulation object
        jac (_type_): jacobian
        index (_type_): ee index

    Returns:
        jac: rotated jacobian
    """    
    world_R_joint = pin.SE3(controller.pin_data.oMf[index].rotation, pin.utils.zero(3))
    return world_R_joint.action @ jac

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

def rollout_mpc_phase_percentage_shift(robot_name = "go2",
                interactive = False,
                v_des = [0.3,0,0],
                save_data = True,
                record_dir = "./data",
                sim_time = 5.0,
                current_time = 0.0,
                visualize = True,
                show_plot = True,
                record_video = False,
                randomize_on_given_state = None,
                ee_in_contact = [],
                nominal_flag = True,
                replanning_point = 0,
                nth_traj_per_replanning = 0,
                force_start_time: float = None,
                force_duration: float = 0.5,
                force_vec: np.ndarray = None,
                ):
    # init robot description
    robot_desc = get_robot_description(robot_name)
    feet_frame_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    feet_name = ["FL", "FR", "RL", "RR"]
    ee_in_contact = ee_in_contact 
    # initialize mpc controller
    mpc = LocomotionMPC(
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
        mpc.set_command(v_des,0.0)
            
    #initialize data recorder
    data_recorder = StateDataRecorder(record_dir,
                                      v_des=v_des,
                                      current_time = current_time,
                                      nominal_flag=nominal_flag,
                                      replanning_point=replanning_point,
                                      nth_traj_per_replanning = nth_traj_per_replanning) if save_data else None
    
    # initialize simulator
    sim = Simulator(robot_desc.xml_scene_path,
                    sim_dt=SIM_DT,
                    viewer_dt=VIEWER_DT)
    
    sim.vs.track_obj = "base"
    
    #initialize visual callback
    vis_feet_pos = ReferenceVisualCallback(mpc, simulator=sim)
    
    # ======================================== Force Perturbation ==================================================
    # Apply external force perturbation to base
    if force_vec is not None and force_start_time is not None:
        sim.apply_force = True
        sim.force_body_name = "base"
        sim.force_vec = force_vec
        sim.force_start_step = int(force_start_time / sim.sim_dt)
        sim.force_end_step = int((force_start_time + force_duration) / sim.sim_dt)

    if randomize_on_given_state is not None:
        q_mj = randomize_on_given_state[:nq]
        v_mj = randomize_on_given_state[nq:-1]
    else:
        q_mj = robot_desc.q0
        v_mj = np.zeros(mpc.pin_model.nv)
       
    # Set mujoco model from unperturbed/perturbed initial state
    sim.set_initial_state(q0=q_mj,v0=v_mj)
    
    sim.run(
        sim_time = sim_time,
        controller=mpc,
        visual_callback=vis_feet_pos,
        data_recorder=data_recorder,
        use_viewer=visualize,
        record_video=record_video,
        allowed_collision=["FL", "FR", "RL", "RR","floor"]
    )
    
    if show_plot:
        mpc.print_timings()
        mpc.plot_traj("q")
        mpc.plot_traj("f")
        mpc.plot_traj("tau")
        mpc.show_plots()
    
    # record_path is name of the recorded file with time_stamp
    record_path = ""
    if save_data and os.path.exists(record_dir):
        # Find the latest file in the directory by modification time
        record_path = max([os.path.join(record_dir, f) for f in os.listdir(record_dir)], key=os.path.getmtime)
        print(f"Latest file found: {record_path}")
    
    data = np.load(record_path)
    sim_over = data["time"][-1]
    tolerance = 1e-2
    early_termination = False
    
    if (sim_time - (sim_over - current_time)) > tolerance:
        early_termination = True
    
    if early_termination:
        # delete the file of record_path
        print("sim_over time = ", sim_over)
        print("real sim time = ", sim_over - current_time)
        os.remove(record_path)
        record_path = ""
        
    return early_termination, record_path

if __name__ == "__main__":
    record_dir = rollout_mpc_phase_percentage_shift(show_plot=False)
    print(record_dir)