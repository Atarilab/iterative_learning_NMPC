from typing import Any, Tuple
import numpy as np
from bisect import bisect_right

from mj_pin_wrapper.pin_robot import PinQuadRobotWrapper
from mj_pin_wrapper.abstract.controller import ControllerAbstract
from .utils.solver import QuadrupedAcadosSolver

class LocomotionMPC(ControllerAbstract):
    """
    Abstract base class for an MPC controller.
    This class defines the structure for an MPC controller
    where specific optimization methods can be implemented by inheriting classes.
    """
    def __init__(self,
                 pin_robot: PinQuadRobotWrapper,
                 gait_name: str = "trot",
                 sim_dt : float = 1.0e-3,
                 **kwargs,
                 ) -> None:
        super().__init__(pin_robot)
        self.gait_name = gait_name

        self.solver = QuadrupedAcadosSolver(pin_robot, gait_name)
        config_opt = self.solver.config_opt

        self.sim_dt = sim_dt
        self.dt_nodes : float = self.solver.dt_nodes
        self.replanning_freq : int = config_opt.replanning_freq
        self.replanning_steps : int = int(1 / (self.replanning_freq * sim_dt))
        self.sim_step : int = 0
        self.current_opt_node : int = 0

        self.v_des : np.ndarray = np.zeros(3)
        self.w_yaw : float = 0.

        self.diverged : bool = False

    def _replan(self) -> bool:
        """
        Returns true if replanning step.
        """
        return self.sim_step % self.replanning_steps == 0

    def set_command(self, v_des: np.ndarray = np.zeros((3,)), w_des: float = 0.) -> None:
        """
        Set velocity commands for the MPC.
        """
        self.v_des = v_des
        self.w_yaw = w_des

    def reset(self) -> None:
        """
        Reset the controller state and reinitialize parameters.
        """
        self.node_dt : float = 0.
        self.replanning_steps : int = 0
        self.sim_step : int = 0
        self.current_opt_node : int = 0

        self.v_des : np.ndarray = np.zeros(3)
        self.w_yaw : float = 0.

        self.diverged : bool = False

    def optimize(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        return optimized trajectories.
        """
        self.solver.init(self.current_opt_node, self.v_des, self.w_yaw)
        q_traj, tau_traj, dt_traj = self.solver.solve()

        return q_traj, tau_traj, dt_traj

    def get_torques(self, q: np.ndarray, v: np.ndarray, robot_data: Any) -> dict:
        """
        Abstract method to compute torques based on the state and planned trajectories.
        Should be implemented by child classes.
        """
        if self._replan():
            # Update configuration from simulation
            self.robot.update(q, v)
            # Replan trajectory
            q_traj, _, dt_traj = self.optimize()
            # Interpolate at sim_dt intervals
            time_traj = np.cumsum(dt_traj)
            q_traj_interp = self.interpolate_trajectory(q_traj, time_traj, self.sim_dt)


    @staticmethod
    def interpolate_trajectory(traj : np.ndarray, time_traj : np.ndarray, sim_dt:float) -> np.ndarray:
        """
        Interpolate traj at a sim_dt period.

        Args:
            traj (np.ndarray): Trajectory to interpolate.
            time_traj (np.ndarray): Time at each trajectory elements.

        Returns:
            np.ndarray: trajectory interpolated at a 1/sim_freq
        """        
        # Generate new time points at 1/sim_frequency intervals
        t_interpolated = np.arange(0, time_traj[-1], sim_dt)
        
        # Interpolate each dimension of the trajectory
        interpolated_traj = np.vstack([
            np.interp(t_interpolated, time_traj, traj[:, dim]) 
            for dim in range(traj.shape[1])
        ]).T

        return interpolated_traj

    def get_traj(self,
                 q0: np.ndarray,
                 trajectory_time : float,
                 ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes trajectory in a MPC fashion starting at q0

        Args:
            q0 (np.ndarray): Initial state
            trajectory_time (float): Total trajectory time

        Returns:
            np.ndarray: _description_
        """
        q_full_traj = [q0]

        current_trajectory_time = 0.
        while current_trajectory_time < trajectory_time:
            
            self.robot.update(q_full_traj[-1])
            # Replan trajectory
            q_traj, _, dt_traj = self.optimize()
            # Interpolate at sim_dt intervals
            time_traj = np.cumsum(dt_traj)
            q_traj_interp = self.interpolate_trajectory(q_traj, time_traj, self.sim_dt)
            time_traj += current_trajectory_time

            # Apply trajectory virtually to the robot
            for q_t in q_traj_interp:
                self.sim_step += 1
                current_trajectory_time = round(self.sim_dt + current_trajectory_time, 4)

                # Exit loop to replan
                if self._replan() or current_trajectory_time >= trajectory_time:
                    # Find the corresponding optimization node
                    self.current_opt_node += bisect_right(time_traj, current_trajectory_time)
                    print("Replan", "node:", self.current_opt_node, "time:", current_trajectory_time)
                    break

                # Record trajectory
                q_full_traj.append(q_t)                

        return q_full_traj