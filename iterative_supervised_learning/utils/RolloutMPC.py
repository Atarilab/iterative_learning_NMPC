## NOTE: This part is for rolling out MPC
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../..')
import argparse
from typing import List
import pinocchio as pin
import numpy as np
from mj_pin.abstract import VisualCallback, DataRecorder  # type: ignore
from mj_pin.simulator import Simulator  # type: ignore
from mj_pin.utils import get_robot_description  # type: ignore
from mpc_controller.mpc import LocomotionMPC

SIM_DT = 1.0e-3
VIEWER_DT = 1/30.

class ReferenceVisualCallback(VisualCallback):
    def __init__(self, mpc_controller, update_step=1):
        super().__init__(update_step)
        self.mpc = mpc_controller
        self.radius = 0.01

    def add_visuals(self, mj_data):
        for i, foot_cnt in enumerate(self.mpc.solver.dyn.feet):
            cnt_pos = self.mpc.solver.params[foot_cnt.plane_point.name]
            cnt_pos_unique = np.unique(cnt_pos, axis=1).T
            for pos in cnt_pos_unique:
                if np.sum(pos) == 0.:
                    continue
                self.add_sphere(pos, self.radius, self.colors.id(i))

        BLACK = self.colors.name("black")
        base_ref = self.mpc.solver.cost_ref[self.mpc.solver.dyn.base_cost.name][:, 0]
        R_WB = pin.rpy.rpyToMatrix(base_ref[3:6][::-1]).flatten()
        self.add_box(base_ref[:3], rot=R_WB, size=[0.08, 0.04, 0.04], rgba=BLACK)

        base_ref = self.mpc.solver.cost_ref_terminal[self.mpc.solver.dyn.base_cost.name]
        R_WB = pin.rpy.rpyToMatrix(base_ref[3:6][::-1]).flatten()
        self.add_box(base_ref[:3], rot=R_WB, size=[0.08, 0.04, 0.04], rgba=BLACK)

class StateDataRecorder(DataRecorder):
    def __init__(self, record_dir: str = "", record_step: int = 1):
        super().__init__(record_dir, record_step)
        self.data = {}
        self.reset()

    def reset(self):
        self.data = {"time": [], "q": [], "v": [], "ctrl": []}

    def save(self):
        if not self.record_dir:
            self.record_dir = os.getcwd()
        os.makedirs(self.record_dir, exist_ok=True)

        timestamp = self.get_date_time_str()
        file_path = os.path.join(self.record_dir, f"simulation_data_{timestamp}.npz")

        try:
            np.savez(file_path, **self.data)
            print(f"Data successfully saved to {file_path}")
        except Exception as e:
            print(f"Error saving data: {e}")

    def _record(self, mj_data):
        self.data["time"].append(mj_data.time)
        self.data["q"].append(mj_data.qpos)
        self.data["v"].append(mj_data.qvel)
        self.data["ctrl"].append(mj_data.ctrl)

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
        sim.run(
            sim_time=self.args.sim_time,
            controller=mpc,
            visual_callback=vis_feet_pos,
            data_recorder=data_recorder,
            use_viewer=self.args.visualize
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
    args = parser.parse_args()

    rollout_mpc = RolloutMPC(args)
    rollout_mpc.run()