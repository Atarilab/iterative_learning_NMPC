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

mpc = LocomotionMPC(
    robot.pin,
    "trot",
    debug=True,
)
sim = Simulator(robot.mj, controller=mpc)

v_des = np.array([0.0, 0.0, 0.0])
mpc.set_command(np.array([0.0, 0.0, 0.27, 0., 0., 0.]), v_des)
q0 = robot.get_state()[0]

# Open loop MPC
# q_traj = mpc.get_traj(q0, 1)
# sim.vis_trajectory(q_traj, loop=True, record_video=False)

# Close loop MPC in simulation
sim.run(20, record_video=False, playback_speed=1)