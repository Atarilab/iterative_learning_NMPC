from typing import Any, Dict, Tuple
import numpy as np
from bisect import bisect_right
from scipy.interpolate import interp1d

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
        self.debug = kwargs.get("debug", False)
        self.gait_name = gait_name

        self.solver = QuadrupedAcadosSolver(pin_robot, gait_name)
        config_opt = self.solver.config_opt

        self.Kp = self.solver.config_cost.Kp
        self.Kd = self.solver.config_cost.Kd

        self.sim_dt = sim_dt
        self.dt_nodes : float = self.solver.dt_nodes
        self.replanning_freq : int = config_opt.replanning_freq
        self.replanning_steps : int = int(1 / (self.replanning_freq * sim_dt))
        self.sim_step : int = 0
        self.current_opt_node : int = 0

        self.v_des : np.ndarray = np.zeros(3)
        self.w_yaw : float = 0.
        self.q_plan = None
        self.v_plan = None
        self.a_plan = None
        self.f_plan = None
        self.time_traj = np.array([])

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
        if self.debug:
            self.solver.print_contact_constraints()
        q_sol, v_sol, a_sol, f_sol, dt_sol = self.solver.solve(print_stats=self.debug, print_time=self.debug)

        return q_sol, v_sol, a_sol, f_sol, dt_sol

    def interpolate_trajectory(
        self,
        traj : np.ndarray,
        time_traj : np.ndarray
        ) -> np.ndarray:
        """
        Interpolate traj at a sim_dt period.

        Args:
            traj (np.ndarray): Trajectory to interpolate.
            time_traj (np.ndarray): Time at each trajectory elements.

        Returns:
            np.ndarray: trajectory interpolated at a 1/sim_freq
        """        
        # Create an interpolation object that supports multi-dimensional input
        interp_func = interp1d(
            time_traj,
            traj,
            axis=0,
            kind=self.solver.config_opt.interpolation_mode,
            fill_value="extrapolate"
            )
        
        # Interpolate the trajectory for all dimensions at once
        t_interpolated = np.linspace(0., time_traj[-1], int(time_traj[-1]/self.sim_dt)+1)
        interpolated_traj = interp_func(t_interpolated)
        
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
            q_traj, _, _, _, dt_traj = self.optimize()
            # Interpolate at sim_dt intervals
            time_traj = np.cumsum(dt_traj)
            q_traj_interp = self.interpolate_trajectory(q_traj, time_traj)
            time_traj += current_trajectory_time

            # Apply trajectory virtually to the robot
            for q_t in q_traj_interp:
                self.sim_step += 1
                current_trajectory_time = round(self.sim_dt + current_trajectory_time, 4)
                # Record trajectory
                q_full_traj.append(q_t) 

                # Exit loop to replan
                if self._replan() or current_trajectory_time >= trajectory_time:
                    # Find the corresponding optimization node
                    self.current_opt_node += bisect_right(time_traj, current_trajectory_time) + 1
                    print("Replan", "node:", self.current_opt_node, "time:", current_trajectory_time)
                    break               

        return q_full_traj
    
    def get_torques(self, q: np.ndarray, v: np.ndarray, robot_data: Any) -> Dict[str, float]:
        """
        Abstract method to compute torques based on the state and planned trajectories.
        Should be implemented by child classes.
        """
        
        if self._replan():
            sim_time = robot_data.time
            plan_step = 0

            # Find the optimization node of the last plan corresponding to the current simulation time
            self.current_opt_node += bisect_right(self.time_traj, sim_time)
            print("Replan", "node:", self.current_opt_node)
        
            # Update configuration from simulation
            self.robot.update(q, v)
            # Replan trajectory
            q_sol, v_sol, a_sol, f_sol, dt_sol = self.optimize()

            # Interpolate plam at sim_dt intervals
            traj_full = np.concatenate((
                q_sol,
                v_sol,
                a_sol,
                f_sol.reshape(-1, 12),
            ), axis=-1)
            self.time_traj = np.cumsum(dt_sol)
            traj_full_interp = self.interpolate_trajectory(traj_full, self.time_traj)
            self.time_traj += sim_time

            self.q_plan, self.v_plan, self.a_plan, self.f_plan = (
                np.split(traj_full_interp, np.cumsum([
                    q_sol.shape[-1],
                    v_sol.shape[-1],
                    a_sol.shape[-1],
                ]),
                axis=-1)
            )
            self.f_plan = self.f_plan.reshape(-1, 4, 3)

        plan_step = self.sim_step % self.replanning_steps
        torques = self.solver.dyn.get_torques(
            self.robot.model,
            self.robot.data,
            self.solver.feet_frame_names,
            self.q_plan[plan_step],
            self.v_plan[plan_step],
            self.a_plan[plan_step],
            self.f_plan[plan_step],
        )

        torques_pd = (torques +
                      self.Kp * (self.q_plan[plan_step, -12:] - q[-12:]) +
                      self.Kd * (self.v_plan[plan_step, -12:] - v[-12:]))

        torque_map = {
            j_name : torques_pd[joint_id]
            for j_name, joint_id
            in self.robot.joint_name2act_id.items()
        }
        self.sim_step += 1

        return torque_map