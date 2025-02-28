import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../..')
from typing import Tuple, List
import pinocchio as pin
import numpy as np
from mj_pin.abstract import VisualCallback, DataRecorder
from mj_pin.simulator import Simulator  # type: ignore
from mj_pin.utils import get_robot_description, mj_frame_pos # type: ignore
from mpc_controller.mpc import LocomotionMPC
import scipy.spatial.transform as st
import mujoco 
import matplotlib.pyplot as plt

# initialize global variables
SIM_DT = 1.0e-3
VIEWER_DT = 1/30.
gait_period = 0.5 # trotting
n_state = 44
nq = 19
nv = 17
kp = 40
kd = 5.0

# initialize pertubation variables
mu_base_pos = 0.0
sigma_base_pos = 0.1
mu_joint_pos = 0.0
sigma_joint_pos = 0.2
mu_base_ori = 0.0
sigma_base_ori = 0.4
mu_vel = 0.0
sigma_vel = 0.2

# VisualCallback
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
        self.add_box(base_ref[:3], rot_euler=base_ref[3:6][::-1], size=[0.08, 0.04, 0.04], rgba=BLACK)
        
        # Base terminal reference
        base_ref = self.mpc.solver.cost_ref_terminal[self.mpc.solver.dyn.base_cost.name]
        self.add_box(base_ref[:3], rot_euler=base_ref[3:6][::-1], size=[0.08, 0.04, 0.04], rgba=BLACK)
        
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
        # print("time to be recorded is = ",round(mj_data.time + self.current_time, 4))
        # input()
        
        self.data["q"].append(q)
        self.data["v"].append(v)
        self.data["ctrl"].append(mj_data.ctrl.copy())
        
        # # Record feet position in the world (x,y,z)
        feet_pos_all = []
        base_wrt_feet = np.zeros(2*len(self.feet_names))
        
        for i, f_name in enumerate(self.feet_names):
            feet_pos = mj_frame_pos(self.mj_model, mj_data, f_name)
            # print("feet_pos = ",feet_pos)
            # print("base_pos = ",q[:3])
            feet_pos_all.extend(feet_pos)
            base_wrt_feet[2*i:2*i+2] = (q[:3] - feet_pos)[:2]  # Correct indexing
        
        # print("feet positions are = ",feet_pos_all)
        # print("shape of feet_pos_all is  = ", np.shape(feet_pos_all))
        # input()
        
        # print("base_wrt_feet = ", base_wrt_feet)
        # input()
        self.data["feet_pos_w"].append(np.array(feet_pos_all))
        self.data["base_wrt_feet"].append(np.array(base_wrt_feet))
        
        ## form state variable
        # the format of state = [[phase_percentage],v,q[2:],base_wrt_feet]
        # if in replanning step, phase percentage is not starting from 0
        phase_percentage = np.round([get_phase_percentage(mj_data.time + self.current_time)], 4)
        state = np.concatenate([phase_percentage, v, q[2:], base_wrt_feet])
        self.data["state"].append(np.array(state))
        
        # transform action from torque to PD target and store
        tau = mj_data.ctrl.copy()
        action = (tau + kd * v[6:])/kp + q[7:]
        # print("current action is = ",action)
        self.data["action"].append(np.array(action))
        
        # record the velocity conditioned goals
        self.data["vc_goals"].append(self.vc_goals)
        
        # record contact conditioned goals(currently just a random noise)
        self.cc_goals = np.random.normal(loc=0.0, scale=0.1, size=(8,))
        self.data["cc_goals"].append(self.cc_goals)

def get_phase_percentage(t:int):
    """get current gait phase percentage based on gait period

    Args:
        t (int): current sim step (NOT sim time in seconds!)

    Returns:
        phi: current gait phase. between 0 - 1
    """ 
       
    # for trot
    gait_period = 0.5
    phi = (t % gait_period)/gait_period
    return phi

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

def rollout_mpc(robot_name = "go2",
                interactive = False,
                v_des = [0.3,0,0],
                save_data = True,
                record_dir = "./data",
                sim_time = 5.0,
                current_time = 0.0,
                visualize = True,
                show_plot = True,
                randomize_on_given_state = None):
    # init robot description
    robot_desc = get_robot_description(robot_name)
    feet_frame_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    feet_name = ["FL", "FR", "RL", "RR"] 
    # initialize mpc controller
    mpc = LocomotionMPC(
            path_urdf=robot_desc.urdf_path,
            feet_frame_names=feet_frame_names,
            robot_name=robot_name,
            joint_ref=robot_desc.q0,
            interactive_goal=interactive,
            sim_dt=SIM_DT,
            print_info=False,
            solve_async=True,
        )
    
    if not interactive:
        mpc.set_command(v_des,0.0)
        
    #initialize visual callback
    vis_feet_pos = ReferenceVisualCallback(mpc)
    
    #initialize data recorder
    data_recorder = StateDataRecorder(record_dir,v_des=v_des,current_time = current_time) if save_data else None
    
    #initialize simulator
    sim = Simulator(robot_desc.xml_scene_path,sim_dt=SIM_DT,viewer_dt=VIEWER_DT)
    sim.vs.track_obj = "base"
    
    #get default position and velocity
    q_mj = robot_desc.q0
    v_mj = np.zeros(mpc.pin_model.nv)
    q_pin, v_pin = mpc.solver.dyn.convert_from_mujoco(q_mj,v_mj)
    
    # set robot to initial position
    pin.forwardKinematics(mpc.pin_model,mpc.pin_data,q_pin,v_pin)
    pin.updateFramePlacements(mpc.pin_model,mpc.pin_data)
    # input()
    
    # TODO: randomize on given state
    # if randomize_on_given_state is not None:
    #     nominal_state = randomize_on_given_state
    #     # xml_path = get_robot_description(robot_name).xml_path
    #     # mj_model = mujoco.MjModel.from_xml_path(xml_path)
        
    #     while True:
    #         q_mj = nominal_state[:nq] 
    #         v_mj = nominal_state[nq:-1]
            
    #         # apply pertubation on quatenion
    #         nominal_quat = q_mj[3:7]
    #         perturbed_quat = apply_quaternion_perturbation(nominal_quat, sigma_base_ori)

    #         perturbation_q = np.concatenate((np.random.normal(mu_base_pos, sigma_base_pos, 3),\
    #                                                     perturbed_quat ,\
    #                                                     np.random.normal(mu_joint_pos,sigma_joint_pos,len(q_mj)-7)))
    #         perturbation_v = np.random.normal(mu_vel,sigma_vel,len(v_mj))
            
    #         q_mj += perturbation_q
    #         v_mj += perturbation_v
            
    #         q_pin, v_pin = mpc.solver.dyn.convert_from_mujoco(q_mj, v_mj)
    #         mpc.solver.dyn.update_pin(q_pin, v_pin)
    #         feet_pos = mpc.solver.dyn.get_feet_position_w()
    #         # feet_pos = mj_frame_pos(mj_model, sim.mj_data, feet_frame_names)
    #         if np.all(feet_pos[:,-1] >= 0):
    #             print("feet_pos = ", feet_pos)
    #             input()
    #             break
    
    # TODO: implement null space randomization
    if randomize_on_given_state is not None:
        nominal_state = randomize_on_given_state
        # TODO: keep randomizing until all the feet are above the ground
        q_mj = nominal_state[:nq] 
        v_mj = nominal_state[nq:-1]
        q_pin, v_pin = mpc.solver.dyn.convert_from_mujoco(q_mj,v_mj)
        
        # perform forward kinematics and compute jacobian
        pin.computeJointJacobians(mpc.pin_model, mpc.pin_data, q_pin)
        pin.framesForwardKinematics(mpc.pin_model,mpc.pin_data,q_pin)
        
        # find end-effector in contact
        ee_in_contact = []
        # Extract the contact plan
        #==========================================================================
        contact_plan = mpc.contact_planner.get_contacts(0, mpc.config_opt.n_nodes+1)
        contact_plan = contact_plan[:,:int(gait_period/mpc.solver.dt_nodes)]
        # print("Contact Plan:")
        # print(contact_plan)
        # print("shape of contact_plan is = ", np.shape(contact_plan))
        # print(mpc.solver.dt_nodes)
        # input()
        #====================================================================
        
        # extract current contact condition
        phase_percentage = nominal_state[-1]
        phase_steps = contact_plan.shape[1]
        phase_index = int(phase_percentage*phase_steps) % phase_steps
        current_contact  = contact_plan[:,phase_index]
        # Display the current contact status
        print(f"Phase Percentage: {phase_percentage * 100:.1f}%")
        print(f"Phase Index: {phase_index}")
        print("Current Contact Condition (1 = Contact, 0 = Swing):")
        print(f"FL_foot: {current_contact[0]}, FR_foot: {current_contact[1]}, RL_foot: {current_contact[2]}, RR_foot: {current_contact[3]}")
        
        for ee in range(len(feet_frame_names)):
            if current_contact[ee] == 1:
                ee_in_contact.append(feet_frame_names[ee])
        
        # print("current in contact ee is  = ", ee_in_contact)
        # input()
        
        # initialize jacobian matrix
        cnt_jac = np.zeros((3*len(ee_in_contact),len(v_pin)))
        cnt_jac_dot = np.zeros((3*len(ee_in_contact),len(v_pin)))
        
        # compute jacobian of the end-effector in contact and its derivative
        for ee in range(len(ee_in_contact)):
            jac = pin.getFrameJacobian(mpc.pin_model,mpc.pin_data,mpc.pin_model.getFrameId(ee_in_contact[ee]), pin.ReferenceFrame.LOCAL)
            jac_dot = pin.getFrameJacobianTimeVariation(mpc.pin_model,\
                        mpc.pin_data,\
                        mpc.pin_model.getFrameId(ee_in_contact[ee]),\
                        pin.ReferenceFrame.LOCAL)
            cnt_jac[3*ee:3*(ee+1),] = rotate_jacobian(mpc, jac,\
                        mpc.pin_model.getFrameId(ee_in_contact[ee]))[0:3,]
            
            cnt_jac_dot[3*ee:3*(ee+1),] = rotate_jacobian(mpc, jac_dot,\
                        mpc.pin_model.getFrameId(ee_in_contact[ee]))[0:3,]
            # print("jac = ", jac)
            # print("jac_dot = ", jac_dot)
            # print("cnt_jac = ", cnt_jac)
            # print("cnt_jac_dot = ", cnt_jac_dot)
            # input()
                    
        # apply pertubation
        min_ee_height = 0.0
        # NOTE: apply pertubation until no foot is below the ground
        while min_ee_height >= 0:
            perturbation_pos = np.concatenate((np.random.normal(mu_base_pos, sigma_base_pos, 3),\
                                                np.random.normal(mu_base_ori, sigma_base_ori, 3), \
                                                np.random.normal(mu_joint_pos, sigma_joint_pos, len(v_pin)-6)))
            perturbation_vel = np.random.normal(mu_vel, sigma_vel, len(v_pin))
            
            if ee_in_contact == []:
                random_pos_vec = perturbation_pos
                random_vel_vec = perturbation_vel
            else:
                random_pos_vec = (np.identity(len(v_pin)) - np.linalg.pinv(cnt_jac)@\
                                        cnt_jac) @ perturbation_pos
                jac_vel = cnt_jac_dot * perturbation_pos + cnt_jac * perturbation_vel
                random_vel_vec = (np.identity(len(v_pin)) - np.linalg.pinv(jac_vel)@\
                                        jac_vel) @ perturbation_pos
            
            # add pertubation to nominal position and velocity (pin data form)
            new_v0 = v_pin + random_vel_vec
            new_q0 = pin.integrate(mpc.pin_model, q_pin, random_pos_vec)
            
            # check if the swing foot is below the ground
            pin.framesForwardKinematics(mpc.pin_model,mpc.pin_data,new_q0)
            ee_below_ground = []
            for e in range(len(feet_frame_names)):
                frame_id = mpc.pin_model.getFrameId(feet_frame_names[e])
                if mpc.pin_data.oMf[frame_id].translation[2] < 0.:
                    ee_below_ground.append(feet_frame_names[e])
            if ee_below_ground == []:
                min_ee_height = -1.
                
            q_mj, v_mj = mpc.solver.dyn.convert_to_mujoco(new_q0,new_v0)
    
    # Set randomized state and simulate from there
    sim.set_initial_state(q0=q_mj,v0=v_mj)
    
    sim.run(
        sim_time = sim_time,
        controller=mpc,
        visual_callback=vis_feet_pos,
        data_recorder=data_recorder,
        use_viewer=visualize,
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
    record_dir = rollout_mpc(show_plot=False)
    print(record_dir)