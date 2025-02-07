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
from mj_pin.utils import get_robot_description  # type: ignore
from mpc_controller.mpc import LocomotionMPC

SIM_DT = 1.0e-3
VIEWER_DT = 1/30.

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

    def run_mpc(self):
        # init
        nq = 19
        nv = 18
        
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
        if self.args.customized_initial_state is not None:
            customized_state = np.array(self.args.customized_initial_state)  # Convert list to NumPy array
            # print(customized_state)
            # print("shape of customized_state", customized_state)
            # input()
            q_mj = customized_state[:nq] 
            v_mj = customized_state[nq:]

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
        )
            
        if self.args.visualize:
            mpc.print_timings()
            mpc.plot_traj("q")
            mpc.plot_traj("f")
            mpc.plot_traj("tau")
            mpc.show_plots()

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

    def run(self):
        if self.args.mode == 'traj_opt':
            self.run_traj_opt()
        elif self.args.mode == 'open_loop':
            self.run_open_loop()
        elif self.args.mode == 'close_loop':
            self.run_mpc()

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
                set_initial_state: List[float] = None) -> Tuple[str, List[float], List[List[float]], List[List[float]], List[List[float]],]:

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
            self.customized_initial_state = set_initial_state
    args = Args()

    # NOTE: Why is phase percentage a part of vc_goals?
    
    # define some global variables
    n_state = 35
    n_action = 12
    nv = 18
    nq = 17
    f_arr = ["FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT"]
    kp = 2.0
    kd = 0.1

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

    # Collect and return recorded data
    if save_data:
        data_file = None
        
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
        num_time_steps = int(sim_time / sim_dt) - int(start_time / sim_dt)
        state_history = np.zeros((num_time_steps, n_state))
        base_history = np.zeros((num_time_steps, 3))
        vc_goal_history = np.zeros((num_time_steps, 3))
        cc_goal_history = np.zeros((num_time_steps, 3))  # Assuming it should be 3D
        action_history = np.zeros((num_time_steps, n_action)) # define action space
        
    
        for file in os.listdir(record_dir):
            if file.startswith("simulation_data_") and file.endswith(".npz"):
                data_file = os.path.join(record_dir, file)
                break
        
        if data_file:
            data = np.load(data_file)
            print("data loaded from", data_file)
            
            time_array = np.array(data["time"])
            q_array = np.array(data["q"])
            v_array = np.array(data["v"])
            ctrl_array = np.array(data["ctrl"])
            
            
            # form state and action history
            for i in range(num_time_steps):
                current_time = time_array[i]  # Get current simulation time
                q = q_array[i]
                v = v_array[i]
                
                # Extract base position (x, y, z)
                base_history[i] = q[:3]

                # Store simulation time in first column
                # state_history[i, 0] = current_time

                # Store velocity in state_history (starting from column 1)
                state_history[i, :nv] = v

                # Store base-relative foot positions (shifted accordingly)
                # state_history[i, nv + 1:nv + 1 + 2 * len(f_arr)] = base_wrt_foot(q)

                # Store configuration (excluding first two elements)
                state_history[i, nv:] = q[2:]
                # print("state is ", state_history[i])
                # input()
                
                # Store vc_goal_history
                vc_goal_history[i,:] = v_des
                
                # Store cc_goal history
                # cc_goal_history = np.zeros((num_time_steps, 1))  # Prevent empty entries
                cc_goal_history = np.full((num_time_steps, 1), 1e-6)  # Safe default
                
                # construct action history
                tau = ctrl_array[i,:]
                action_history[i,:] = (tau + kd * v[6:])/kp + q[7:]
                
                # print(state_history)
                # input()
                # print("shape of state_history = ",np.shape(state_history))
                # input()
            return record_dir, state_history, base_history, vc_goal_history, cc_goal_history, action_history
    return record_dir, [], [], [], [], [], []

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
    parser.add_argument('--randomize_initial_state', default=True, help='Enable or disable simulation visualization.')
    args = parser.parse_args()

    rollout_mpc = RolloutMPC(args)
    rollout_mpc.run()