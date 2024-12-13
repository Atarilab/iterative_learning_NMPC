import numpy as np
from mj_pin_wrapper.sim_env.utils import RobotModelLoader
from mj_pin_wrapper.mj_pin_robot import MJPinQuadRobotWrapper
from mj_pin_wrapper.simulator import Simulator
from mpc_controller.utils.solver import QuadrupedAcadosSolver
from mpc_controller.utils.sim_visuals import desired_contact_locations_callback

robot = MJPinQuadRobotWrapper(
    *RobotModelLoader.get_paths("go2", mesh_dir="assets"),
    rotor_inertia=0.5*0.250*(0.09/2)**2,
    gear_ratio=6.33,
    )
q0, v0 = robot.get_state()

solver = QuadrupedAcadosSolver(
    robot.pin,
    gait_name="trot",
    print_info=True,
)

solver.set_max_iter(100)
solver.set_nlp_tol(solver.config_opt.nlp_tol / 10.)
solver.set_qp_tol(solver.config_opt.qp_tol / 10.)

v_des = np.array([0.2, 0.0, 0.0])
base_ref = np.zeros(12)
base_ref[:3] += v_des * solver.config_opt.time_horizon
base_ref[2] = q0[2]
base_ref[6:9] = v_des

solver.init(q0, base_ref, base_ref, v0, v_des=v_des)
q_traj, v_traj, _, f_traj, dt_traj = solver.solve()

sim = Simulator(robot.mj)

visual_callback = (lambda viewer, step, q, v, data :
    desired_contact_locations_callback(viewer, step, q, v, data, solver))

sim.vis_trajectory(
    q_traj,
    dt_traj,
    loop=True,
    record_video=False,
    playback_speed=0.5,
    visual_callback_fn=visual_callback,
    )