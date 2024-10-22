import numpy as np
import argparse
from mj_pin_wrapper.sim_env.utils import RobotModelLoader
from mj_pin_wrapper.mj_pin_robot import MJPinQuadRobotWrapper
from mj_pin_wrapper.simulator import Simulator
from mpc_controller.mpc import LocomotionMPC

def main(close_loop: bool = True,
         sim_time: float = 3.0,
         print_info: bool = False,
         record_video: bool = False):
    
    robot = MJPinQuadRobotWrapper(
        *RobotModelLoader.get_paths("go2", mesh_dir="assets"),
        rotor_inertia=0.5 * 0.250 * (0.09 / 2) ** 2,
        gear_ratio=6.33,
    )

    mpc = LocomotionMPC(
        robot,
        "trot",
        print_info=print_info,
        record_traj=True
    )
    v_des = np.array([0.0, 0.0, 0.0])
    mpc.set_command(v_des)

    sim = Simulator(robot.mj, controller=mpc)

    # Close loop MPC in simulation
    if close_loop:
        sim.run(sim_time, record_video=record_video, playback_speed=1)
        mpc.plot_traj('q')
        mpc.plot_traj('f')
    # Open loop MPC
    else:
        q_traj = mpc.open_loop(sim_time)
        mpc.plot_traj('q')
        mpc.plot_traj('v')
        mpc.plot_traj('f')
        sim.vis_trajectory(q_traj, loop=True, record_video=record_video, playback_speed=1)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Simulate a quadruped robot using MPC.")

    parser.add_argument('--close_loop', action='store_true', default=False,
                        help="Run in close-loop mode (default: False for open-loop).")
    parser.add_argument('--sim_time', type=float, default=3.0,
                        help="Simulation time in seconds (default: 3.0).")
    parser.add_argument('--print_info', action='store_true', default=False,
                        help="Print additional info during simulation (default: False).")
    parser.add_argument('--record_video', action='store_true', default=False,
                        help="Record simulation video (default: False).")

    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(close_loop=args.close_loop,
         sim_time=args.sim_time,
         print_info=args.print_info,
         record_video=args.record_video)
