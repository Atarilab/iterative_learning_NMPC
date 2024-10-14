import numpy as np
from mj_pin_wrapper.sim_env.utils import RobotModelLoader
from mj_pin_wrapper.mj_pin_robot import MJPinQuadRobotWrapper
from mj_pin_wrapper.simulator import Simulator
from mpc_controller.mpc import LocomotionMPC

robot = MJPinQuadRobotWrapper(
    *RobotModelLoader.get_paths("go2", mesh_dir="assets"),
    rotor_inertia=0.5*0.250*(0.09/2)**2,
    gear_ratio=6.33,
    )

sim = Simulator(robot.mj)

mpc = LocomotionMPC(
    robot.pin,
    "trot"
)

v_des = np.array([0.3, 0., 0.])
mpc.set_command(v_des)
q0 = robot.get_state()[0]

q_traj = mpc.get_traj(q0, 3)

sim.vis_trajectory(q_traj, loop=True)
