import mujoco
import mujoco.viewer
import pinocchio as pin
import numpy as np

from mj_pin_wrapper.abstract.controller import ControllerAbstract
from mujoco._structs import MjData
from .solver import QuadrupedAcadosSolver

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
                                       solver: QuadrupedAcadosSolver) -> None:
    """
    Visualize the desired contact plan locations in the MuJoCo viewer.
    """
    if (sim_step % UPDATE_VISUALS_STEPS == 0):
        viewer.user_scn.ngeom = 0
        i_geom = 0

        # Plot contact plan
        if (solver.config_opt.cnt_patch_restriction or
             solver.config_opt.cnt_patch_restriction):
            
            for i_foot, foot_cnt in enumerate(solver.dyn.feet):

                contacts_w = np.unique(
                    solver.params[foot_cnt.plane_point.name],
                    axis=1
                    ).T

                non_zero = lambda cnt : (cnt[:2] != np.zeros(2)).any()
                for i_cnt, cnt_w in enumerate(filter(non_zero, contacts_w)):

                    # Add visuals
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[i_geom],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[SPHERE_RADIUS / (i_cnt + 3) * 3, 0, 0],
                        pos=cnt_w,
                        mat=np.eye(3).flatten(),
                        rgba=FEET_COLORS[i_foot % len(FEET_COLORS)],
                    )
                    i_geom += 1

            base_ref = solver.cost_ref[solver.dyn.base_cost.name][:, 0]
            R_WB = pin.rpy.rpyToMatrix(base_ref[3:6][::-1])

            mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[i_geom],
                        type=mujoco.mjtGeom.mjGEOM_BOX,
                        size=[0.08, 0.04, 0.04],
                        pos=base_ref[:3],
                        mat=R_WB.flatten(),
                        rgba=[0.1, 0.1, 0.1, .3],
                    )
            i_geom += 1

            viewer.user_scn.ngeom = i_geom