from typing import List
import os
import numpy as np
from mj_pin.abstract import VisualCallback # type: ignore
from mj_pin.simulator import Simulator # type: ignore
from mj_pin.utils import load_mj_pin, load_pin, mj_frame_pos # type: ignore

from mpc_controller.mpc import LocomotionMPC

class FeetVisualCallback(VisualCallback):
    def __init__(self, mj_model, feet_names : List[str], update_step = 1):
        super().__init__(update_step)
        self.mj_model = mj_model
        self.feet_names = feet_names

    def _add_visuals(self, mj_data):
        radius = 0.03
        for i, f_name in enumerate(self.feet_names):
            pos = mj_frame_pos(self.mj_model, mj_data, f_name)
            self.add_sphere(pos, radius, self.colors_id[i])

if __name__ == "__main__":
    mj_model, pin_model, robot_desc = load_mj_pin("go2", from_mjcf=False)
    feet_frame_names = [f + "_foot" for f in robot_desc.eeff_frame_name]

    mpc = LocomotionMPC(
        pin_model,
        path_urdf=robot_desc.urdf_path,
        feet_frame_names = feet_frame_names,
        joint_ref = robot_desc.q0,
        sim_dt=mj_model.opt.timestep,
        print_info=False,
        )
    
    v_des = [.5, 0., 0.]
    mpc.set_command(v_des)
    
    vis_feet_pos = FeetVisualCallback(mj_model, robot_desc.eeff_frame_name)

    sim = Simulator(mj_model)

    sim.run(controller=mpc,
            visual_callback=vis_feet_pos)