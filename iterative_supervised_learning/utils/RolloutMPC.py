## NOTE: This part is for rolling out MPC
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../..')
import argparse
from typing import Tuple, List
import pinocchio as pin
import numpy as np
from mj_pin.abstract import VisualCallback, DataRecorder  # type: ignore
from mj_pin.simulator import Simulator  # type: ignore
from mj_pin.utils import get_robot_description, mj_frame_pos # type: ignore
from mpc_controller.mpc import LocomotionMPC
import scipy.spatial.transform as st
import mujoco

SIM_DT = 1.0e-3
VIEWER_DT = 1/30.

# pertubation variables
mu_base_pos = 0.0
sigma_base_pos = 0.1
mu_joint_pos = 0.0
sigma_joint_pos = 0.2
mu_base_ori = 0.0
sigma_base_ori = 0.1
mu_vel = 0.0
sigma_vel = 0.1

def random_quaternion_perturbation(sigma):
    """
    Generate a small random quaternion perturbation.
    The perturbation is sampled from a normal distribution with standard deviation sigma.
    """
    random_axis = np.random.normal(0, 1, 3)  # Random rotation axis
    random_axis /= np.linalg.norm(random_axis)  # Normalize to unit vector
    angle = np.random.normal(0, sigma)  # Small random rotation angle
    perturb_quat = st.Rotation.from_rotvec(angle * random_axis).as_quat()  # Convert to quaternion
    return perturb_quat

def apply_quaternion_perturbation(nominal_quat, sigma_base_ori):
    """
    Apply a small random rotation perturbation to a given quaternion.
    """
    perturb_quat = random_quaternion_perturbation(sigma_base_ori)
    perturbed_quat = st.Rotation.from_quat(nominal_quat) * st.Rotation.from_quat(perturb_quat)
    return perturbed_quat.as_quat()  # Convert back to quaternion
    
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


class RolloutMPC:
    def __init__(self, args):
        self.args = args

    def run_traj_opt(self):
        robot_desc = get_robot_description(self.args.robot_name)
        feet_frame_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

        mpc = LocomotionMPC(
            path_urdf=robot_desc.urdf_path,
            feet_frame_names=feet_frame_names,
            robot_name=self.args.robot_name,
            joint_ref=robot_desc.q0,
            sim_dt=SIM_DT,
            print_info=True,
        )
        mpc.set_command(self.args.v_des, 0.0)
        mpc.set_convergence_on_first_iter()

        q = robot_desc.q0
        v = np.zeros(mpc.pin_model.nv)
        q_plan, v_plan, _, _, dt_plan = mpc.optimize(q, v)

        q_plan_mj = np.array([
            mpc.solver.dyn.convert_to_mujoco(q_plan[i], v_plan[i])[0] for i in range(len(q_plan))
        ])
        time_traj = np.concatenate(([0], np.cumsum(dt_plan)))

        sim = Simulator(robot_desc.xml_scene_path, sim_dt=SIM_DT, viewer_dt=VIEWER_DT)
        sim.vs.set_high_quality()
        sim.visualize_trajectory(q_plan_mj, time_traj, record_video=self.args.record_video)
    
    def run_open_loop(self):
        robot_desc = get_robot_description(self.args.robot_name)
        feet_frame_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

        mpc = LocomotionMPC(
            path_urdf=robot_desc.urdf_path,
            feet_frame_names=feet_frame_names,
            robot_name=self.args.robot_name,
            joint_ref=robot_desc.q0,
            interactive_goal=False,
            sim_dt=SIM_DT,
            print_info=False,
        )
        mpc.set_command(self.args.v_des, 0.0)

        q = robot_desc.q0
        v = np.zeros(mpc.pin_model.nv)
        q_traj = mpc.open_loop(q, v, self.args.sim_time)

        mpc.print_timings()
        sim = Simulator(robot_desc.xml_scene_path, sim_dt=SIM_DT)
        sim.vs.set_high_quality()
        sim.visualize_trajectory(q_traj, record_video=self.args.record_video)

    def run_mpc(self):
        # init
        nq = 19 # base(3[x,y,z] + 4[quatenion]) + joints(4*3) = 19
        nv = 18 # base(3[vx,vy,vz] + 3[wx,wy,wz]) + joints(4*3) = 18
        
        robot_desc = get_robot_description(self.args.robot_name)
        feet_frame_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]        
        mpc = LocomotionMPC(
            path_urdf=robot_desc.urdf_path,
            feet_frame_names=feet_frame_names,
            robot_name=self.args.robot_name,
            joint_ref=robot_desc.q0,
            interactive_goal=self.args.interactive,
            sim_dt=SIM_DT,
            print_info=False,
            solve_async=True,
        )
        if not self.args.interactive:
            mpc.set_command(self.args.v_des, 0.0)

        vis_feet_pos = ReferenceVisualCallback(mpc)
        data_recorder = StateDataRecorder(self.args.record_dir) if self.args.save_data else None
        sim = Simulator(robot_desc.xml_scene_path, sim_dt=SIM_DT, viewer_dt=VIEWER_DT)
        sim.vs.track_obj = "base"
        q_mj = robot_desc.q0  # Default initial position
        v_mj = np.zeros(mpc.pin_model.nv)  # Default initial velocity
        
        #===============================================
        if self.args.randomize_on_given_state is not None:
            nominal_state = np.array(self.args.randomize_on_given_state)  # Convert list to NumPy array
            # print(customized_state)
            # print("shape of customized_state", customized_state)
            # input()
            while True:
                # keep randomizing if some feet is under the ground
                q_mj = nominal_state[:nq] 
                v_mj = nominal_state[nq:-1]
                
                # current_time = nominal_state[-1]
                # phase_percentage = get_phase_percentage(current_time * SIM_DT)
                # print("shape of q_mj when unpacking nominal state = ", np.shape(q_mj))
                # print("shape of v_mj when unpacking nominal state = ", np.shape(v_mj))
                # print(nominal_state)
                # print("current time = ", current_time)
                # print("phase_percentage = ", phase_percentage)
                
                nominal_quat = q_mj[3:7]
                perturbed_quat = apply_quaternion_perturbation(nominal_quat, sigma_base_ori)
                perturbation_q = np.concatenate((np.random.normal(mu_base_pos, sigma_base_pos, 3),\
                                                        perturbed_quat ,\
                                                        np.random.normal(mu_joint_pos,sigma_joint_pos,len(q_mj)-7)))
                perturbation_v = np.random.normal(mu_vel,sigma_vel,len(v_mj))
                
                q_mj += perturbation_q
                v_mj += perturbation_v
                
                q_pin, v_pin = mpc.solver.dyn.convert_from_mujoco(q_mj, v_mj)
                mpc.solver.dyn.update_pin(q_pin, v_pin)
                feet_pos_w = mpc.solver.dyn.get_feet_position_w()
                
                # TODO:
                # if feet_pos_w is under the ground, randomize again, until all the feet are above the ground
                if np.all(feet_pos_w[:,-1]>=0):
                    # print("feet_pos_w = ", feet_pos_w)
                    # print("shape of feet_pos_w is = ", np.shape(feet_pos_w))
                    break
                
            
        #===============================================      
        if self.args.randomize_initial_state:
            q_mj += np.random.uniform(low=-0.05, high=0.05, size=q_mj.shape)
            # print("q",q_mj)
            # print("v",v_mj)
            # Set customized initial condition
        
        #==================================================
        sim.set_initial_state(q0=q_mj,v0=v_mj)
       
        sim.run(
            sim_time=self.args.sim_time,
            controller=mpc,
            visual_callback=vis_feet_pos,
            data_recorder=data_recorder,
            use_viewer=self.args.visualize,
            allowed_collision=["FL", "FR", "RL", "RR","floor"]
        )
            
        if self.args.show_plot:
            mpc.print_timings()
            mpc.plot_traj("q")
            mpc.plot_traj("f")
            mpc.plot_traj("tau")
            mpc.show_plots()

    def run(self):
        if self.args.mode == 'traj_opt':
            self.run_traj_opt()
        elif self.args.mode == 'open_loop':
            self.run_open_loop()
        elif self.args.mode == 'close_loop':
            self.run_mpc()


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

def rollout_mpc(mode: str = "close_loop",
                sim_time: float = 5,
                sim_dt: float = 0.001,
                start_time: float = 0.0,
                robot_name: str = "go2",
                record_dir: str = "./data/",
                v_des: List[float] = [0.5, 0.0, 0.0],
                save_data: bool = True,
                interactive: bool = False,
                record_video: bool = False,
                visualize: bool = False,
                randomize_initial_state: bool = False,
                randomize_on_given_state: List[float] = None,
                show_plot:bool= True) -> Tuple[str, List[float], List[List[float]], List[List[float]], List[List[float]],]:

    # Create argparse-like structure
    class Args:
        def __init__(self):
            self.mode = mode
            self.sim_time = sim_time
            self.robot_name = robot_name
            self.record_dir = record_dir
            self.v_des = v_des
            self.save_data = save_data
            self.interactive = interactive
            self.record_video = record_video
            self.visualize = visualize
            self.randomize_initial_state = randomize_initial_state
            self.randomize_on_given_state = randomize_on_given_state
            self.show_plot = show_plot
    args = Args()

    # NOTE: Why is phase percentage a part of vc_goals?
    
    # define some global variables
    n_state = 44
    n_action = 12
    nv = 18
    nq = 17
    f_arr = ["FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT"]
    kp = 40.0
    kd = 5.0
    
    
    # Ensure the record directory exists
    if save_data:
        os.makedirs(record_dir, exist_ok=True)

    # Instantiate the RolloutMPC class
    rollout_mpc = RolloutMPC(args)

    # Run the appropriate simulation
    if mode == 'traj_opt':
        rollout_mpc.run_traj_opt()
    elif mode == 'open_loop':
        rollout_mpc.run_open_loop()
    elif mode == 'close_loop':
        rollout_mpc.run_mpc()
    else:
        raise ValueError("Invalid mode. Choose from 'traj_opt', 'open_loop', or 'close_loop'.")

    if args.randomize_on_given_state is not None:
        current_timestep = args.randomize_on_given_state[-1]
    else:
        current_timestep = 0
    print("current_timestep = ",current_timestep)
    # input()
    
    # Collect and return recorded data
    if save_data:
        files = None
        
        # TODO: return as a compact state and action:
        """
        state: 
            1- v (robot velocity): 6 base velocities(3 linear, 3 angular) + 12 joint velocities(4 legs * 3 joints/leg) = 18
            2- base_wrt_foot(q): relative x,y distances from the robot's base to each foot (4 feet * 2 values(x,y)) = 8  -- this thing is now not implemented
            3- q[2:] :q is full configuration vector: base position(x,y,z), base orientation(quaternion: x,y,z,w), 12 joint angles, total 19
                we exclude first 2 elements of q which is (x,y), and have (z,quaternion(4),12 joint angles) -> 17
            
            Finally: n_state = 18+8+17 = 43
        
        action: 4 legs * 3 joints/leg
        base: q[0:3]
        """
        # define return variables
        num_time_steps = int(sim_time / sim_dt) - int(start_time/sim_dt)
        state_history = np.zeros((num_time_steps, n_state)) # [phase_percentage, q[2:], v, base_wrt_feet]
        base_history = np.zeros((num_time_steps, 3)) # [x,y,z]
        vc_goal_history = np.zeros((num_time_steps, 3)) # [vx,vy,w]
        cc_goal_history = np.zeros((num_time_steps, 3))  # Assuming it should be 3D
        action_history = np.zeros((num_time_steps, n_action)) # define action space
        
    
        # for file in os.listdir(record_dir):
        #     if file.startswith("simulation_data_") and file.endswith(".npz"):
        #         data_file = os.path.join(record_dir, file)
        #         break
        
        files = [
        os.path.join(record_dir, file)
        for file in os.listdir(record_dir)
        if file.startswith("simulation_data_") and file.endswith(".npz")
        ]
        
        if files:
            # Sort files by modification time (latest file first)
            files.sort(key=os.path.getmtime, reverse=True)
            latest_file = files[0]  # Get the most recent file
            
            data = np.load(latest_file)
            print("data loaded from", latest_file)
            
            time_array = np.array(data["time"])
            q_array = np.array(data["q"])
            v_array = np.array(data["v"])
            ctrl_array = np.array(data["ctrl"])
            base_wrt_feet = np.array(data["base_wrt_feet"])
            
            # print("length of time_array = ",len(time_array))
            # print("num_time_steps = ", num_time_steps)
            # print("base_wrt_feet is  = ", base_wrt_feet[:2])
            
            # if simulation failed middleway, return empty state history
            if len(time_array)<num_time_steps:
                return record_dir, [], [], [], [], []
            
            # form state and action history
            for i in range(num_time_steps):
                current_time = time_array[i] + current_timestep * sim_dt # Get current simulation time
                q = q_array[i]
                v = v_array[i]
                
                # Extract base position (x, y, z)
                base_history[i] = q[:3]

                # # Extract phase percentage
                # if args.randomize_on_given_state is not None:
                #     print("current_time while replanning = ",current_time)
                #     print("phase_percentage while replanning= ",get_phase_percentage(current_time))
                #     input()
                    
                state_history[i, 0] = get_phase_percentage(current_time)

                # Store velocity in state_history (starting from column 1)
                state_history[i, 1:nv+1] = v

                # Store base-relative foot positions (shifted accordingly)
                # state_history[i, nv + 1:nv + 1 + 2 * len(f_arr)] = base_wrt_foot(q)

                # Store configuration (excluding first two elements)
                state_history[i, nv+1:n_state-8] = q[2:]
                
                # Add base_wrt_foot information in the state space
                state_history[i,n_state-8:] = base_wrt_feet[i]
                
                # print("state is ", state_history[i])
                # input()
                
                # Store vc_goal_history
                vc_goal_history[i,:] = v_des
                
                # Store cc_goal history
                # cc_goal_history = np.zeros((num_time_steps, 1))  # Prevent empty entries
                cc_goal_history = np.full((num_time_steps, 1), 1e-6)  # Safe default
                
                # construct action history
                tau = ctrl_array[i,:]
                # switch from torque to PD target
                action_history[i,:] = (tau + kd * v[6:])/kp + q[7:]
                
                # print(state_history)
                # input()
                # print("shape of state_history = ",np.shape(state_history))
                # input()
            return record_dir, state_history, base_history, vc_goal_history, cc_goal_history, action_history
    return record_dir, [], [], [], [], []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MPC simulations.")
    parser.add_argument('--mode', type=str, default="close_loop", choices=['traj_opt', 'open_loop', 'close_loop'], help='Mode to run the simulation.')
    parser.add_argument('--sim_time', type=float, default=5, help='Simulation time.')
    parser.add_argument('--robot_name', type=str, default='go2', help='Name of the robot.')
    parser.add_argument('--record_dir', type=str, default='./data/', help='Directory to save recorded data.')
    parser.add_argument('--v_des', type=float, nargs=3, default=[0.5, 0.0, 0.0], help='Desired velocity.')
    parser.add_argument('--save_data', action='store_true', help='Flag to save data.')
    parser.add_argument('--interactive', action='store_true', help='Use keyboard to set the velocity goal (zqsd).')
    parser.add_argument('--record_video', action='store_true', help='Record a video of the viewer.')
    parser.add_argument('--visualize', default=True, help='Enable or disable simulation visualization.')
    parser.add_argument('--randomize_initial_state', default=False, help='Enable or disable simulation visualization.')
    parser.add_argument('--randomize_on_given_state',default = None)
    parser.add_argument('--show_plot',default=False)
    args = parser.parse_args()

    rollout_mpc = RolloutMPC(args)
    rollout_mpc.run()