import os
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

if __name__ == "__main__":
    SIM_TIME = 5
    SIM_DT = 1.0e-3
    ROBOT_NAME = "go2"
    RECORD_DIR = "./data/"
    V_DES = [1.5, 0.0, 0.0]

    # MPC Controller
    mj_model, _, robot_desc = load_mj_pin(ROBOT_NAME, from_mjcf=False)
    feet_frame_names = [f + "_foot" for f in robot_desc.eeff_frame_name]

    mpc = LocomotionMPC(
        path_urdf=robot_desc.urdf_path,
        feet_frame_names = feet_frame_names,
        robot_name=ROBOT_NAME,
        joint_ref = robot_desc.q0,
        sim_dt=SIM_DT,
        print_info=False,
        record_traj=True,
        )
    mpc.set_command(V_DES, 0.0)

    # Simulator with visual callback and state data recorder
    vis_feet_pos = ReferenceVisualCallback(mpc)
    data_recorder = StateDataRecorder(RECORD_DIR)

    sim = Simulator(mj_model, sim_dt=SIM_DT, viewer_dt=1/50)
    sim.run(
        viewer=False,
        sim_time=SIM_TIME,
        controller=mpc,
        visual_callback=vis_feet_pos,
        data_recorder=data_recorder,)
    
    mpc.print_timings()
    mpc.plot_traj("f")
    mpc.plot_traj("tau")
    mpc.show_plots()