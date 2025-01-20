import os
import argparse
from typing import List
import pinocchio as pin
import numpy as np
from mj_pin.abstract import VisualCallback, DataRecorder # type: ignore
from mj_pin.simulator import Simulator # type: ignore
from mj_pin.utils import load_mj_pin   # type: ignore

from mpc_controller.mpc import LocomotionMPC

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
                self.add_sphere(pos, self.radius, self.colors_id[i])

        # Base reference
        BLACK = VisualCallback.BLACK
        BLACK[-1] = 0.5
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

    def _record(self, mj_data) -> None:
        """
        Record simulation data at the current simulation step.
        """
        # Record time and state
        self.data["time"].append(mj_data.time)
        self.data["q"].append(mj_data.qpos)
        self.data["v"].append(mj_data.qvel)
        self.data["ctrl"].append(mj_data.ctrl)

def run_traj_opt(args):
    SIM_DT = args.sim_dt
    ROBOT_NAME = args.robot_name
    V_DES = args.v_des

    # MPC Controller
    mj_model, _, robot_desc = load_mj_pin(ROBOT_NAME, from_mjcf=False)
    feet_frame_names = [f + "_foot" for f in robot_desc.eeff_frame_name]

    mpc = LocomotionMPC(
        path_urdf=robot_desc.urdf_path,
        feet_frame_names = feet_frame_names,
        robot_name=ROBOT_NAME,
        joint_ref = robot_desc.q0,
        sim_dt=SIM_DT,
        print_info=True,
        )
    mpc.set_command(V_DES, 0.0)
    mpc.set_convergence_on_first_iter()

    q = robot_desc.q0
    v = np.zeros(mj_model.nv)
    q_plan, v_plan, _, _, dt_plan = mpc.optimize(q, v)
    
    q_plan_mj = np.array([mpc.solver.dyn.convert_to_mujoco(q_plan[i], v_plan[i])[0] for i in range(len(q_plan))])
    time_traj = np.concatenate(([0], np.cumsum(dt_plan)))

    # Simulator with visual callback and state data recorder
    vis_feet_pos = ReferenceVisualCallback(mpc)

    sim = Simulator(mj_model, sim_dt=SIM_DT, viewer_dt=1/50)
    sim.visualize_trajectory(q_plan_mj, time_traj, vis_feet_pos)

def run_mpc(args):
    SIM_TIME = args.sim_time
    SIM_DT = args.sim_dt
    ROBOT_NAME = args.robot_name
    RECORD_DIR = args.record_dir
    V_DES = args.v_des
    INTERACTIVE = args.interactive
    USE_VIEWER = True

    # MPC Controller
    mj_model, _, robot_desc = load_mj_pin(ROBOT_NAME, from_mjcf=False)
    feet_frame_names = [f + "_foot" for f in robot_desc.eeff_frame_name]

    mpc = LocomotionMPC(
        path_urdf=robot_desc.urdf_path,
        feet_frame_names = feet_frame_names,
        robot_name=ROBOT_NAME,
        joint_ref = robot_desc.q0,
        interactive_goal=INTERACTIVE,
        sim_dt=SIM_DT,
        print_info=False,
        solve_async=True,
        )
    if not INTERACTIVE:
        mpc.set_command(V_DES, 0.0)

    # Simulator with visual callback and state data recorder
    vis_feet_pos = ReferenceVisualCallback(mpc)
    data_recorder = StateDataRecorder(RECORD_DIR) if args.save_data else None

    sim = Simulator(mj_model, sim_dt=SIM_DT, viewer_dt=1/50)
    sim.run(
        viewer=USE_VIEWER,
        sim_time=SIM_TIME,
        controller=mpc,
        visual_callback=vis_feet_pos,
        data_recorder=data_recorder,)
    
    mpc.print_timings()
    mpc.plot_traj("q")
    mpc.plot_traj("f")
    mpc.plot_traj("tau")
    mpc.show_plots()

def run_open_loop(args):
    SIM_TIME = args.sim_time
    SIM_DT = args.sim_dt
    ROBOT_NAME = args.robot_name
    V_DES = args.v_des

    # MPC Controller
    mj_model, _, robot_desc = load_mj_pin(ROBOT_NAME, from_mjcf=False)
    feet_frame_names = [f + "_foot" for f in robot_desc.eeff_frame_name]

    mpc = LocomotionMPC(
        path_urdf=robot_desc.urdf_path,
        feet_frame_names = feet_frame_names,
        robot_name=ROBOT_NAME,
        joint_ref = robot_desc.q0,
        interactive_goal=False,
        sim_dt=SIM_DT,
        print_info=False,
        )
    mpc.set_command(V_DES, 0.0)

    q = robot_desc.q0
    v = np.zeros(mj_model.nv)
    q_traj = mpc.open_loop(q, v, SIM_TIME)
   
    mpc.print_timings()

    sim = Simulator(mj_model, sim_dt=SIM_DT, viewer_dt=1/50)
    sim.visualize_trajectory(q_traj)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MPC simulations.")
    parser.add_argument('--mode', type=str, default="close_loop", choices=['traj_opt', 'open_loop', 'close_loop'], help='Mode to run the simulation.')
    parser.add_argument('--sim_time', type=float, default=5, help='Simulation time.')
    parser.add_argument('--sim_dt', type=float, default=1.0e-3, help='Simulation time step.')
    parser.add_argument('--robot_name', type=str, default='go2', help='Name of the robot.')
    parser.add_argument('--record_dir', type=str, default='./data/', help='Directory to save recorded data.')
    parser.add_argument('--v_des', type=float, nargs=3, default=[0.5, 0.0, 0.0], help='Desired velocity.')
    parser.add_argument('--save_data', action='store_true', help='Flag to save data.')
    parser.add_argument('--interactive', action='store_true', help='Use keyboard to set the velocity goal (zqsd).')
    args = parser.parse_args()

    if args.mode == 'traj_opt':
        run_traj_opt(args)
    elif args.mode == 'open_loop':
        run_open_loop(args)
    elif args.mode == 'close_loop':
        run_mpc(args)