import mujoco
import mujoco.viewer
import pinocchio as pin
import numpy as np

from mj_pin_wrapper.abstract.controller import ControllerAbstract
from mujoco._structs import MjData
from ..mpc import LocomotionMPC

UPDATE_VISUALS_STEPS = 50 # Update position every <UPDATE_VISUALS_STEPS> sim steps
FEET_COLORS = [
    [1., 0., 0., 1.], # FR
    [0., 1., 0., 1.], # FL
    [0., 0., 1., 1.], # RR
    [1., 1., 1., 1.], # RL
]
N_NEXT_CONTACTS = 12
SPHERE_RADIUS = 0.012
    
def desired_contact_locations_callback(viewer,
                                       sim_step: int,
                                       q: np.ndarray,
                                       v: np.ndarray,
                                       robot_data: MjData,
                                       controller: LocomotionMPC) -> None:
    """
    Visualize the desired contact plan locations in the MuJoCo viewer.
    """
    
    if sim_step % UPDATE_VISUALS_STEPS == 0 and controller.solver.config_opt.restrict_cnt_loc:
        
        viewer.user_scn.ngeom = 0
        i_geom = 0
        for i_foot, foot_cnt in enumerate(controller.solver.dyn.feet):

            contacts_w = np.unique(
                controller.solver.params[foot_cnt.plane_point.name],
                axis=1
                ).T

            i_cnt = 0
            for cnt_w in contacts_w:
                if (cnt_w == np.zeros(3)).all():
                    continue

                # Add visuals
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[i_geom],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[SPHERE_RADIUS / (i_cnt + 3) * 3, 0, 0],
                    pos=cnt_w,
                    mat=np.eye(3).flatten(),
                    rgba=FEET_COLORS[i_foot % len(FEET_COLORS)],
                )
                i_cnt += 1
                i_geom += 1

        viewer.user_scn.ngeom = i_geom