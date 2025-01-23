## NOTE: This part is for rolling out MPC
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../..')

from typing import List
import pinocchio as pin
import numpy as np
from mj_pin.abstract import VisualCallback, DataRecorder  # type: ignore
from mj_pin.simulator import Simulator  # type: ignore
from mj_pin.utils import load_mj_pin  # type: ignore

from mpc_controller.mpc import LocomotionMPC


class ReferenceVisualCallback(VisualCallback):
    def __init__(self, mpc_controller, update_step=1):
        super().__init__(update_step)
        self.mpc = mpc_controller
        self.radius = 0.01

    def _add_visuals(self, mj_data):
        for i, foot_cnt in enumerate(self.mpc.solver.dyn.feet):
            cnt_pos = self.mpc.solver.params[foot_cnt.plane_point.name]
            cnt_pos_unique = np.unique(cnt_pos, axis=1).T
            for pos in cnt_pos_unique:
                if np.sum(pos) == 0.:
                    continue
                self.add_sphere(pos, self.radius, self.colors_id[i])

        BLACK = VisualCallback.BLACK
        BLACK[-1] = 0.5
        base_ref = self.mpc.solver.cost_ref[self.mpc.solver.dyn.base_cost.name][:, 0]
        R_WB = pin.rpy.rpyToMatrix(base_ref[3:6][::-1]).flatten()
        self.add_box(base_ref[:3], rot=R_WB, size=[0.08, 0.04, 0.04], rgba=BLACK)

        base_ref = self.mpc.solver.cost_ref_terminal[self.mpc.solver.dyn.base_cost.name]
        R_WB = pin.rpy.rpyToMatrix(base_ref[3:6][::-1]).flatten()
        self.add_box(base_ref[:3], rot=R_WB, size=[0.08, 0.04, 0.04], rgba=BLACK)


class StateDataRecorder(DataRecorder):
    def __init__(self, record_dir: str = "", record_step: int = 1) -> None:
        super().__init__(record_dir, record_step)
        self.data = {}
        self.reset()

    def reset(self) -> None:
        self.data = {"time": [], "q": [], "v": [], "ctrl": []}

    def save(self) -> None:
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

    def _record(self, mj_data) -> None:
        self.data["time"].append(mj_data.time)
        self.data["q"].append(mj_data.qpos)
        self.data["v"].append(mj_data.qvel)
        self.data["ctrl"].append(mj_data.ctrl)


class RolloutMPC:
    def __init__(self, sim_time: int, sim_dt: float, robot_name: str, record_dir: str, v_des: List[float]):
        self.sim_time = sim_time
        self.sim_dt = sim_dt
        self.robot_name = robot_name
        self.record_dir = record_dir
        self.v_des = v_des

        self.mpc = None
        self.sim = None

        # Variables to store recorded data
        self.recorded_time = []
        self.recorded_q = []
        self.recorded_v = []
        self.recorded_ctrl = []

    def initialize(self):
        """
        Initialize the MPC controller, simulator, and callbacks.
        """
        # Load robot model and description
        mj_model, _, robot_desc = load_mj_pin(self.robot_name, from_mjcf=False)
        feet_frame_names = [f + "_foot" for f in robot_desc.eeff_frame_name]

        # Initialize MPC
        self.mpc = LocomotionMPC(
            path_urdf=robot_desc.urdf_path,
            feet_frame_names=feet_frame_names,
            robot_name=self.robot_name,
            joint_ref=robot_desc.q0,
            sim_dt=self.sim_dt,
            print_info=False,
            record_traj=True,
        )
        self.mpc.set_command(self.v_des, 0.0)

        # Initialize simulator
        vis_feet_pos = ReferenceVisualCallback(self.mpc)
        self.data_recorder = StateDataRecorder(self.record_dir)

        self.sim = Simulator(mj_model, sim_dt=self.sim_dt)
        self.sim.visual_callback = vis_feet_pos
        self.sim.data_recorder = self.data_recorder

    def run(self):
        """
        Run the simulation and extract recorded data into separate variables.
        """
        if self.sim is None or self.mpc is None:
            raise RuntimeError("RolloutMPC is not initialized. Call `initialize` first.")

        # Run the simulation
        self.sim.run(sim_time=self.sim_time, 
                     controller=self.mpc,
                     visual_callback=self.sim.visual_callback,
                     data_recorder=self.sim.data_recorder)

        # Extract recorded data
        if self.sim.data_recorder:
            self.recorded_time = self.sim.data_recorder.data["time"]
            self.recorded_q = self.sim.data_recorder.data["q"]
            self.recorded_v = self.sim.data_recorder.data["v"]
            self.recorded_ctrl = self.sim.data_recorder.data["ctrl"]


    def execute(self):
        """
        Wrapper function to initialize, run, and save results in one call.
        """
        self.initialize()
        self.run()



# Example usage
if __name__ == "__main__":
    rollout_mpc = RolloutMPC(
        sim_time=5,
        sim_dt=1.0e-3,
        robot_name="go2",
        record_dir="./data/",
        v_des=[0.5, 0.0, 0.0],
    )
    rollout_mpc.initialize()
    rollout_mpc.run()
    rollout_mpc.save_results()
