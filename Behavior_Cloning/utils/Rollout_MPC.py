# this is a updated version of RolloutMPC combining all the features that I tested
# firstly, in an unperturbed setting, the MPC can run
# secondly, include feature that allow taking any state as initial state and perform nullspace perturbation
# thirdly, include feature that allow taking any state as initial state and perform force perturbation
# lastly, put every tunable variable in config file to simplify tuning

# imports
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../..')
from mj_pin.abstract import VisualCallback, DataRecorder
from mj_pin.simulator import Simulator  # type: ignore
from mj_pin.utils import get_robot_description, mj_frame_pos # type: ignore
from mpc_controller.mpc import LocomotionMPC

import numpy as np
import mujoco
from typing import Tuple, List, Dict

from omegaconf import OmegaConf
import hydra
from hydra.utils import instantiate
from pathlib import Path

# define utils
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

def get_feet_in_contact_by_id(mj_data, geom_to_frame_name: Dict[int, str], ground_geom_id: int = 0) -> List[str]:
    """
    Get the list of feet in contact from simulator data, returning Pinocchio frame names (e.g., 'FL_foot').

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

# Data recorder
class StateDataRecorder(DataRecorder):
    def __init__(
        self,
        feet_names = ["FL", "FR", "RL", "RR"],
        robot_name = "go2",
        record_dir: str = "",
        record_step: int = 1,
        v_des: np.ndarray = np.array([0,0,0]),
        current_time: float = 0.0,
        nominal_flag = True,
        replanning_point = 0,
        nth_traj_per_replanning = 0,
        kp = 20.0,
        kd = 1.5,
        ) -> None:
        """
        A simple data recorder that saves simulation data to a .npz file.
        """
        super().__init__(record_dir, record_step)
        
        # initialize global parameter
        self.kp = kp
        self.kd = kd
        
        # initialize data struct
        self.data = {}
        
        # initialize user input
        self.vc_goals = v_des
        self.cc_goals = np.random.normal(loc=0.0, scale=0.1, size=(8,))
        
        # get recording start time
        self.current_time = current_time
        
        # determine if the recording is reference traj or perturbed traj
        self.nominal_flag = nominal_flag
        
        # import perturbation related variables
        self.replanning_point = replanning_point
        self.nth_traj_per_replanning_point = nth_traj_per_replanning
        
        # initialization of robot model
        self.feet_names = feet_names
        self.robot_name = robot_name
        xml_path = get_robot_description(self.robot_name).xml_path
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        
        # reset
        self.reset()

    def reset(self) -> None:
        # define data struct keys
        self.data = {"time": [], # time steps
                     "q": [],  # robot pos configuration
                     "v": [], # robot velocity
                     "ctrl": [], # applied torque
                     "feet_pos_w":[], # feet position in world frame
                     "base_wrt_feet":[], # base with right to feet in world frame
                     "state":[], # combined state for supervised learning
                     "action":[], # calculated PD target for supervised learning
                     "vc_goals":[], # desired velocity from user input
                     "cc_goals":[], # TODO
                     "contact_vec":[], # contact condition from simulation
                     }

    def save(self) -> None:
        # define record dir
        if not self.record_dir:
            self.record_dir = os.getcwd()
        os.makedirs(self.record_dir, exist_ok=True)
        timestamp = self.get_date_time_str()
        
        # define saving name depending on whether the trajectory is nominal
        if self.nominal_flag:
            file_path = os.path.join(self.record_dir, f"traj_nominal_{timestamp}.npz")
        else:
            file_path = os.path.join(self.record_dir, f"traj_{self.replanning_point}_{self.nth_traj_per_replanning_point}.npz")

        # save data to a npz file for each individual rollout
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
        # extract q and v from simulator
        q = mj_data.qpos.copy()
        v = mj_data.qvel.copy()
        
        # get current time (sim time + record start time) and save
        self.data["time"].append(round(mj_data.time + self.current_time, 4))
        # print("current time is = ",round(mj_data.time + self.current_time, 4))
        
        # save raw data for debugging: q, v and torque  
        self.data["q"].append(q) # in the order of [FL,FR,RL,RR]
        self.data["v"].append(v) # in the order of [FL,FR,RL,RR]
        self.data["ctrl"].append(mj_data.ctrl.copy()) # in the order of [FR,FL,RR,RL]
        
        # Record feet position in the world frame (x,y,z)
        feet_pos_all = []
        ee_in_contact = []
        base_wrt_feet = np.zeros(2*len(self.feet_names))
        
        for i, f_name in enumerate(self.feet_names):
            feet_pos = mj_frame_pos(self.mj_model, mj_data, f_name)
            feet_pos_all.extend(feet_pos)
            base_wrt_feet[2*i:2*i+2] = (q[:3] - feet_pos)[:2]
        
        self.data["feet_pos_w"].append(np.array(feet_pos_all))
        
        # Record base with right to feet in world frame
        self.data["base_wrt_feet"].append(np.array(base_wrt_feet))
        
        geom_to_frame_name = {
            20: "FL_foot",
            32: "FR_foot",
            44: "RL_foot",
            56: "RR_foot"
        }

        # Record ee in contact from simulator data
        ee_in_contact = get_feet_in_contact_by_id(mj_data, geom_to_frame_name)
        
        contact_vec = np.array([
            int("FL_foot" in ee_in_contact),
            int("FR_foot" in ee_in_contact),
            int("RL_foot" in ee_in_contact),
            int("RR_foot" in ee_in_contact)
        ])
        self.data["contact_vec"].append(contact_vec)
        
        ## form state variable and save
        # the format of state = [[phase_percentage],v,q[2:],base_wrt_feet]
        # if in replanning step, phase percentage is not starting from 0
        phase_percentage = np.round([get_phase_percentage(mj_data.time + self.current_time)], 4)
        #==========================================================================================
        # state with base_wrt_feet
        state = np.concatenate([phase_percentage, v, q[2:], base_wrt_feet])
        self.data["state"].append(np.array(state)) # here is unnormalized state
        #=========================================================================================
        # transform action from torque to PD target and save
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
        
        action = (tau_flfrrlrr + self.kd * v[6:])/self.kp + q[7:] # in the order of [FL,FR,RL,RR]
        # print("current action is = ",action)
        self.data["action"].append(np.array(action))
        
        # record the velocity conditioned goals
        self.data["vc_goals"].append(self.vc_goals)
        
        # record contact conditioned goals(currently just a random noise)
        self.cc_goals = np.random.normal(loc=0.0, scale=0.1, size=(8,))
        self.data["cc_goals"].append(self.cc_goals)

class Rollout_MPC():
    """
    Given mpc controller, simulator, datarecorder, visualcallback, 
    and a bunch of settings from cfg,
    rollout mpc and record simulation data for further process
    """
    
    def __init__(self,cfg):
        # global
        self.robot_name = cfg.robot_name
        self.feet_frame_names = cfg.feet_names
        self.robot_desc = get_robot_description(self.robot_name)
        self.sim_dt = float(cfg.SIM_DT)
        self.viewer_dt = 1/30.
        
        # mpc controller setting
        self.interactive = cfg.interactive
        self.visualize = cfg.visualize
        self.show_plot = cfg.show_plot
        self.record_video = cfg.record_video
        self.v_des = cfg.v_des
        
        # simulator setting
        self.save_data = cfg.save_data
        self.record_dir = cfg.record_dir
        self.sim_time = cfg.sim_time
        
        # perturbation setting
        self.randomize_on_given_state = None
        self.ee_in_contact = []
        
        # replanning related
        self.current_time = cfg.current_time
        self.nominal_flag = cfg.nominal_flag
        self.replanning_point = 0
        self.nth_traj_per_replanning = 0
        self.current_phase_percentage = None
        self.q0 = None
        self.v0 = None
        
        # force perturbation related
        self.force_start_time = None
        self.force_duration = None
        self.force_vec = None
        
        print("All config initialization for MPC rollout finished")
        
    def setup_nominal_rollout(self,
              sim_time = 2.0,
              current_time = 0.0,
              record_dir = "."):
        # this function is for customizing some parameters
        self.sim_time = sim_time
        self.current_time = current_time
        self.record_dir = record_dir
    
    def setup_force_perturbation(self,
              record_dir = ".",
              replan_instruction = None,
              perturbation = None,
              sim_time = 2.0):
        

        # unpack replan_instruction and perturbation if given
        if replan_instruction is not None:
            self.current_time = replan_instruction["current_time"]
            self.current_phase_percentage = replan_instruction["current_phase_percentage"]
            self.q0 = replan_instruction["q0"]
            self.v0 = replan_instruction["v0"]
            self.replanning_point = replan_instruction["replanning_point"]
            self.nth_traj_per_replanning = replan_instruction["nth_traj_per_replanning"]
            self.nominal_flag = replan_instruction["nominal_flag"
                                                   ]
        if perturbation is not None:
            force_struct = perturbation
            self.force_start_time = force_struct["start_time"]
            self.force_duration = force_struct["duration"]
            self.force_vec = force_struct["force_vec"]
            
        # update internal parameters
        self.sim_time = sim_time
        self.record_dir = record_dir
         
    def setup_mpc(self):
        # initialize mpc controller
        self.mpc = LocomotionMPC(
            path_urdf=self.robot_desc.urdf_path,
            feet_frame_names=self.feet_frame_names,
            robot_name=self.robot_name,
            joint_ref=self.robot_desc.q0,
            interactive_goal = self.interactive,
            sim_dt = self.sim_dt,
            print_info= False,
            solve_async= True            
        )
        print("MPC controller initialized")
        
        if not self.interactive:
           self.mpc.set_command(self.v_des,0.0)
        print("MPC command set")
        
    def setup_simulator(self):
        # initialize data_recorder
        self.data_recorder = StateDataRecorder(
            record_dir=self.record_dir,
            v_des = self.v_des,
            current_time = self.current_time,
            nominal_flag = self.nominal_flag,
            replanning_point = self.replanning_point,
            nth_traj_per_replanning = self.nth_traj_per_replanning
        )
        print("Data recorder initialized")

        # initialize simulator
        self.sim = Simulator(
            xml_path = self.robot_desc.xml_scene_path,
            sim_dt=self.sim_dt,
            viewer_dt=self.viewer_dt
        )
        self.sim.vs.track_obj = "base"
        print("Mujoco simulator initialized")
        
        # initialize visual callback
        self.vis_feet_pos = ReferenceVisualCallback(
            mpc_controller= self.mpc,
            simulator=self.sim,    
        )
        print("Visual Callback initialized")
        
    def setup(self):
        # default setup
        self.setup_mpc()
        self.setup_simulator()
    
    def setup_nullspace_perturbation(self):
        pass
    
    def check_early_termination(self,
                                save_data,
                                record_dir):
        record_path = ""
        if save_data and os.path.exists(record_dir):
            # Find the latest file in the directory by modification time
            record_path = max([os.path.join(record_dir, f) for f in os.listdir(record_dir)], key=os.path.getmtime)
            print(f"Latest file found: {record_path}")
        else:
            print("Data not saved or record_dir not exist!")
            return
        data = np.load(record_path)
        sim_over = data["time"][-1]
        tolerance = 1e-2
        early_termination = False
        if (self.sim_time - (sim_over - self.current_time)) > tolerance:
            early_termination = True
        return early_termination, sim_over, record_path
                
    def run(self):
        # NOTE: if force perturbation is activated, always call setup_force_perturbation before executing run
        # NOTE: if nullspace perturbation is activated, always call setup_nullspace_perturbation before executing run
        
        self.setup()
        print("Default setup finished")
        # Based on whether perturbed, set perturbation related parameters
        if self.nominal_flag:
            q_mj = self.robot_desc.q0
            v_mj = np.zeros(self.mpc.pin_model.nv)
        else:
            q_mj = self.q0
            v_mj = self.v0
            if self.force_vec is not None and self.force_start_time is not None:
                self.sim.apply_force = True
                self.sim.force_body_name = "base"
                self.sim.force_vec = self.force_vec
                self.sim.force_start_step = int(self.force_start_time / self.sim.sim_dt)
                self.sim.force_end_step = int((self.force_start_time + self.force_duration) / self.sim.sim_dt)
            
        self.sim.set_initial_state(q0 = q_mj, v0 = v_mj)
        self.sim.run(
            sim_time = self.sim_time,
            controller = self.mpc,
            visual_callback = self.vis_feet_pos,
            data_recorder=self.data_recorder,
            use_viewer = self.visualize,
            record_video = self.record_video,
            allowed_collision=["FL", "FR", "RL", "RR","floor"]
        )
        
        # after simulation, show plot if requested
        if self.show_plot:
            self.mpc.print_timings()
            self.mpc.plot_traj("q")
            self.mpc.plot_traj("f")
            self.mpc.plot_traj("tau")
            self.mpc.show_plots()
        
        # Determine if the rollout is early terminated
        early_termination, sim_over, record_path = self.check_early_termination(self.save_data,
                                                        self.record_dir)
        
        # delete file if early termination
        if early_termination:
            # delete the file of record_path
            print("sim_over time = ", sim_over)
            print("real sim time = ", sim_over - self.current_time)
            os.remove(record_path)
            record_path = ""
        
        return early_termination, record_path
        

@hydra.main(config_path="../examples/cfgs/", config_name="bc_experimental.yaml", version_base=None)
def main(cfg):
    rollout = Rollout_MPC(cfg)
    # rollout.setup_mpc()
    # rollout.setup_simulator()
    rollout.run()

# for testing
if __name__ == "__main__":
    main()             