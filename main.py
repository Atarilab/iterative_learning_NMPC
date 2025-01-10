from typing import List
import pinocchio as pin
import numpy as np
from mj_pin.abstract import VisualCallback # type: ignore
from mj_pin.simulator import Simulator # type: ignore
from mj_pin.utils import load_mj_pin   # type: ignore

from mpc_controller.mpc import LocomotionMPC

class ReferenceVisualCallback(VisualCallback):
    def __init__(self, mpc_controller, update_step = 1):
        super().__init__(update_step)
        self.mpc = mpc_controller
        self.radius = 0.01

    def _add_visuals(self, mj_data):
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
        
        # Base reference
        BLACK = VisualCallback.BLACK
        BLACK[-1] = 0.5
        base_ref = self.mpc.solver.cost_ref_terminal[self.mpc.solver.dyn.base_cost.name]
        R_WB = pin.rpy.rpyToMatrix(base_ref[3:6][::-1]).flatten()
        self.add_box(base_ref[:3], rot=R_WB, size=[0.08, 0.04, 0.04], rgba=BLACK)


if __name__ == "__main__":
    SIM_DT = 1.0e-3
    ROBOT_NAME = "go2"

    mj_model, _, robot_desc = load_mj_pin(ROBOT_NAME, from_mjcf=False)
    feet_frame_names = [f + "_foot" for f in robot_desc.eeff_frame_name]

    mpc = LocomotionMPC(
        path_urdf=robot_desc.urdf_path,
        feet_frame_names = feet_frame_names,
        robot_name=ROBOT_NAME,
        joint_ref = robot_desc.q0,
        sim_dt=SIM_DT,
        print_info=False,
        )
    
    v_des = [0.5, 0.0, 0.0]
    mpc.set_command(v_des, 0.0)
    
    vis_feet_pos = ReferenceVisualCallback(mpc)

    sim = Simulator(mj_model, sim_dt=SIM_DT)

    sim.run(
        sim_time=10,
        controller=mpc,
        visual_callback=vis_feet_pos)
    
    mpc.print_timings()
    mpc.plot_traj("f")
    mpc.plot_traj("tau")
    mpc.show_plots()