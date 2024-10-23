import numpy as np
from mj_pin_wrapper.sim_env.utils import RobotModelLoader
from mj_pin_wrapper.mj_pin_robot import MJPinQuadRobotWrapper
from mj_pin_wrapper.simulator import Simulator
from mpc_controller.utils.solver import QuadrupedAcadosSolver

robot = MJPinQuadRobotWrapper(
    *RobotModelLoader.get_paths("go2", mesh_dir="assets"),
    rotor_inertia=0.5*0.250*(0.09/2)**2,
    gear_ratio=6.33,
    )

sim = Simulator(robot.mj)
q0, _ = robot.get_state()

solver = QuadrupedAcadosSolver(
    robot.pin,
    "trot",
    True,
)

solver.set_max_iter(50)

q_des = np.array([0.5, 0., 0.265, 0., 0., 0.])
v_des = np.array([0.5, 0.0, 0.0])
solver.init(q0, np.zeros(18), q_des=q_des, v_des=v_des)
q_all = [q0]
dt_all = [0.]
q_traj, v_traj, _, f_traj, dt_traj = solver.solve()
solver.set_max_iter(10)
n = 4
for i in range(15):
    q0 = q_traj[n, :]
    v0 = v_traj[n, :]
    # input()
    q_all.extend([q_traj[i, :] for i in range(1, n + 1)])
    dt_all.extend([dt_traj[1]] * n)
    i_node = i * n + n
    solver.init(q0, v0, q_des=q_des, v_des=v_des, i_node=i_node)
    q_traj, v_traj, _, f_traj, dt_traj = solver.solve()
solver.print_contact_constraints()

print(q_all[-1])
print(len(dt_all))

sim.vis_trajectory(q_all, dt_all, loop=True, record_video=False, playback_speed=0.5)
