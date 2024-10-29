from collections import defaultdict
import time
from typing import Any, Dict, Tuple
from matplotlib import pyplot as plt
import numpy as np
from bisect import bisect_left
from scipy.interpolate import interp1d
from math import floor, ceil
from concurrent.futures import ThreadPoolExecutor, Future
import pinocchio as pin

from mj_pin_wrapper.mj_pin_robot import MJPinQuadRobotWrapper
from mj_pin_wrapper.abstract.controller import ControllerAbstract
from .utils.solver import QuadrupedAcadosSolver
from .utils.transform import quat_to_ypr_state
from .utils.profiling import time_fn, print_timings

class LocomotionMPC(ControllerAbstract):
    """
    Abstract base class for an MPC controller.
    This class defines the structure for an MPC controller
    where specific optimization methods can be implemented by inheriting classes.
    """
    def __init__(self,
                 robot: MJPinQuadRobotWrapper,
                 gait_name: str = "trot",
                 sim_dt : float = 1.0e-3,
                 print_info : bool = True,
                 record_traj : bool = False,
                 compute_timings : bool = True,
                 **kwargs
                 ) -> None:
        super().__init__(robot.pin)
        self.mj_robot = robot.mj
        self.gait_name = gait_name
        self.print_info = print_info

        self.solver = QuadrupedAcadosSolver(robot.pin, gait_name, print_info)
        config_opt = self.solver.config_opt

        self.Kp = self.solver.config_opt.Kp
        self.Kd = self.solver.config_opt.Kd

        self.sim_dt = sim_dt
        self.dt_nodes : float = self.solver.dt_nodes
        self.replanning_freq : int = config_opt.replanning_freq
        self.replanning_steps : int = int(1 / (self.replanning_freq * sim_dt))
        self.sim_step : int = 0
        self.plan_step : int = 0
        self.current_opt_node : int = 0
        self.use_delay : bool = config_opt.use_delay
        self.delay : int = 0
        self.last : int = 0

        self.v_des : np.ndarray = np.zeros(3)
        self.w_des : np.ndarray = np.zeros(3)
        self.base_ref : np.ndarray = np.zeros(12)
        self.q_plan : np.ndarray = None
        self.v_plan : np.ndarray = None
        self.a_plan : np.ndarray = None
        self.f_plan : np.ndarray = None
        self.q_opt : np.ndarray = None
        self.v_opt : np.ndarray = None
        self.time_traj : np.ndarray = np.array([])

        # For plots
        self.record_traj = record_traj
        self.q_full = []
        self.v_full = []
        self.a_full = []
        self.f_full = []
        self.tau_full = []

        self.diverged : bool = False

        # Setup timings
        self.compute_timings = compute_timings
        self.timings = defaultdict(list)

        # Multiprocessing
        self.executor = ThreadPoolExecutor(max_workers=1)  # One thread for asynchronous optimization
        self.optimize_future: Future = None                # Store the future result of optimize
        self.plan_submitted = False                        # Flag to indicate if a new plan is ready

    def _replan(self) -> bool:
        """
        Returns true if replanning step.
        Record trajectory of the last 
        """
        replan = self.sim_step % self.replanning_steps == 0

        if self.use_delay:
            replan = replan and (self.optimize_future is None or self.optimize_future.done())

        return replan
    
    def _step(self) -> None:
        self.increment_base_ref_position()
        self.sim_step += 1
        self.plan_step += 1

    def _record_plan(self) -> None:
        """
        Record trajectory of the last plan until self.plan_step.
        """
        self.q_full.append(self.q_plan[self.delay : self.plan_step])
        self.v_full.append(self.v_plan[self.delay : self.plan_step])
        self.a_full.append(self.a_plan[self.delay : self.plan_step])
        self.f_full.append(self.f_plan[self.delay : self.plan_step])

    def set_command(self, v_des: np.ndarray = np.zeros((3,)), w_yaw: float = 0.) -> None:
        """
        Set velocity commands for the MPC.
        """
        self.v_des = v_des
        self.w_des[2] = w_yaw

    def compute_base_ref(self) -> np.ndarray:
        """
        Compute base reference for the solver.
        """
        t_horizon = self.solver.config_opt.time_horizon
        # Height to config
        base_ref = self.base_ref.copy()
        base_ref[2] = self.solver.config_gait.nom_height
        # Horizontal base
        base_ref[4:6] = 0.

        # Setup reference velocities in local frame
        # v_des is in local frame already
        # w_yaw in global frame
        R_WB = pin.rpy.rpyToMatrix(base_ref[3:6][::-1])
        w_des_local = (R_WB.T @ self.w_des)[::-1]
        base_ref[-3:] = np.round(w_des_local, 1)

        # Compute velocity in global frame
        # Apply angular velocity
        R_yaw = pin.rpy.rpyToMatrix(self.w_des * t_horizon)
        base_ref[6:9] = R_yaw @ self.v_des
        v_des_glob = np.round(R_WB @ self.v_des, 1)

        # Update position and orientation according to desired velocities
        base_ref[:2] += v_des_glob[:2] * t_horizon
        base_ref[3] += self.w_des[-1] * t_horizon

        return base_ref

    def increment_base_ref_position(self, ):
        R_WB = pin.rpy.rpyToMatrix(self.base_ref[3:6][::-1])
        v_des_glob = np.round(R_WB @ self.v_des, 1)
        self.base_ref[:2] += v_des_glob[:2] * self.sim_dt
        self.base_ref[3] += self.w_des[-1] * self.sim_dt

    def reset(self) -> None:
        """
        Reset the controller state and reinitialize parameters.
        """
        self.node_dt : float = 0.
        self.replanning_steps : int = 0
        self.sim_step : int = 0
        self.current_opt_node : int = 0

        self.v_des : np.ndarray = np.zeros(3)
        self.w_des : float = np.zeros(3)

        self.diverged : bool = False
    
    @time_fn("optimize")
    def optimize(self,
                 q : np.ndarray,
                 v : np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        return optimized trajectories.
        """
        base_ref = self.compute_base_ref()
        self.solver.init(q, base_ref, v, self.current_opt_node, self.v_des, self.w_des[2])
        q_sol, v_sol, a_sol, f_sol, dt_sol = self.solver.solve()

        return q_sol, v_sol, a_sol, f_sol, dt_sol

    @time_fn("interpolate_trajectory")
    def interpolate_trajectory(
        self,
        traj : np.ndarray,
        time_traj : np.ndarray,
        kind: str = "",
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
            kind = kind if kind else self.solver.config_opt.interpolation_mode,
            fill_value="extrapolate",
            bounds_error=False,
            assume_sorted=True,
            )
        
        # Interpolate the trajectory for all dimensions at once
        t_interpolated = np.linspace(0., time_traj[-1], int(time_traj[-1]/self.sim_dt)+1)
        interpolated_traj = interp_func(t_interpolated)

        return interpolated_traj
    
    def interpolate_sol(self,
                        q_sol : np.ndarray,
                        v_sol : np.ndarray,
                        a_sol : np.ndarray,
                        f_sol : np.ndarray,
                        dt_sol : np.ndarray,
                        ) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
        """
        Interpolate solution found by the solver at sim dt time intervals.
        Repeat for inputs.
        Linear interpolation for states.
        """
        # Interpolate plan at sim_dt intervals
        state_full = np.concatenate((
            q_sol,
            v_sol,
        ), axis=-1)
        input_full = np.concatenate((
            a_sol,
            f_sol.reshape(-1, 12),
        ), axis=-1)

        time_traj = np.cumsum(dt_sol) - dt_sol

        state_full_interp = self.interpolate_trajectory(state_full, time_traj)
        input_full_interp = self.interpolate_trajectory(input_full, time_traj[:-1], kind='zero')
        q_plan, v_plan = np.split(
            state_full_interp,
            [q_sol.shape[-1]],
                axis=-1
        )
        a_plan, f_plan = np.split(
            input_full_interp,
            [a_sol.shape[-1]],
                axis=-1
        )
        f_plan = f_plan.reshape(-1, 4, 3)

        return q_plan, v_plan, a_plan, f_plan, time_traj
            
    def open_loop(self,
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
        q0, v0 = self.mj_robot.get_state()
        q_full_traj = [q0]
        self.q_plan = [q0]
        self.v_plan = [v0]
        sim_time = 0.
        time_traj = []

        while sim_time <= trajectory_time:

            # Replan trajectory    
            if self._replan():

                # Record trajectory
                if self.record_traj and self.sim_step > 0:
                    self._record_plan()

                self._convergence_on_first_iter()
                
                # Find the corresponding optimization node
                self.current_opt_node += bisect_left(time_traj, sim_time)
                print("Replan", "node:", self.current_opt_node, "time:", sim_time)

                q, v = self.q_plan[self.plan_step], self.v_plan[self.plan_step]

                q_sol, v_sol, a_sol, f_sol, dt_sol = self.optimize(q.copy(), v.copy())
                q_sol[0] = q
                v_sol[0] = v

                (
                self.q_plan,
                self.v_plan,
                self.a_plan,
                self.f_plan,
                time_traj,
                ) = self.interpolate_sol(q_sol, v_sol, a_sol, f_sol, dt_sol)

                time_traj += sim_time
                self.plan_step = 0
                self.delay = 0
            
            # Simulation step
            q_full_traj.append(self.q_plan[self.plan_step])
            self._step()
            sim_time = round(sim_time + self.sim_dt, 4)

        return q_full_traj
    
    def _convergence_on_first_iter(self):
        if self.sim_step == 0:
            self.solver.set_max_iter(50)
            self.solver.set_nlp_tol(self.solver.config_opt.nlp_tol / 10.)
            self.solver.set_qp_tol(self.solver.config_opt.qp_tol / 5.)
        elif self.sim_step <= self.replanning_steps:
            self.solver.set_max_iter(self.solver.config_opt.max_iter)
            self.solver.set_nlp_tol(self.solver.config_opt.nlp_tol)
            self.solver.set_qp_tol(self.solver.config_opt.qp_tol)
    
    def get_torques(self, q: np.ndarray, v: np.ndarray, robot_data: Any) -> Dict[str, float]:
        """
        Abstract method to compute torques based on the state and planned trajectories.
        Should be implemented by child classes.
        """
        # Start a new optimization asynchronously if it's time to replan
        if self._replan():
            self.start_time = time.time()

            # Find the optimization node of the last plan corresponding to the current simulation time
            sim_time = round(robot_data.time, 4)
            self.current_opt_node += bisect_left(self.time_traj, sim_time)
            self.q_opt, self.v_opt = q.copy(), v.copy()
        
            # On first iteration
            self._convergence_on_first_iter()

            # Set up asynchronous optimize call
            self.optimize_future = self.executor.submit(self.optimize, self.q_opt, self.v_opt)
            self.time_start_plan = sim_time
            self.plan_submitted = True

            # Wait for the solver if no delay
            while not self.use_delay and not self.optimize_future.done():
                time.sleep(1.0e-3)

        # Check if the future is done and the new plan is ready to be used
        if (self.plan_submitted and self.optimize_future.done() or
            self.sim_step == 0):
            sim_time = round(robot_data.time, 4)

            try:
                # Retrieve new plan from future
                q_sol, v_sol, a_sol, f_sol, dt_sol = self.optimize_future.result()

                # Record trajectory
                if self.record_traj and self.sim_step > 0:
                    self._record_plan()

                # Interpolate plan
                self.q_plan, self.v_plan, self.a_plan, self.f_plan, self.time_traj = self.interpolate_sol(q_sol, v_sol, a_sol, f_sol, dt_sol)

                # Apply delay
                if (self.use_delay and self.sim_step != 0):
                    replanning_time = time.time() - self.start_time
                    self.delay = round(replanning_time / self.sim_dt)
                else:
                    q_sol[0] = self.q_opt
                    v_sol[0] = self.v_opt
                    self.delay = 0

                self.plan_step = self.delay
                self.time_traj += self.time_start_plan
                self.plan_submitted = False

            except Exception as e:
                print("Optimization error:\n", e)
                self.plan_submitted = False
                
        torques = self.solver.dyn.get_torques(
            self.robot.model,
            self.robot.data,
            q,
            v,
            self.a_plan[self.plan_step],
            self.f_plan[self.plan_step],
        )

        torques_pd = (torques +
                      self.Kp * (self.q_plan[self.plan_step, -12:] - q[-12:]) +
                      self.Kd * (self.v_plan[self.plan_step, -12:] - v[-12:]))

        torque_map = {
            j_name : torques_pd[joint_id]
            for j_name, joint_id
            in self.robot.joint_name2act_id.items()
        }

        # Record trajectories
        if self.record_traj:
            self.tau_full.append(torques_pd)
        
        self._step()

        return torque_map
    
    def plot_traj(self, var_name: str):
        """
        Plot one of the recorded plans using time as the x-axis in a subplot with 3 columns.

        Args:
            var_name (str): Name of the plan to plot. Should be one of:
                            'q', 'v', 'a', 
                            'f', 'dt', 'tau'.
        """
        # Check if the plan name is valid
        var_name += "_full"
        if not hasattr(self, var_name):
            raise ValueError(f"Plan '{var_name}' does not exist. Choose from: 'q', 'v', 'a', 'f', 'dt', 'tau'.")

        # Get the selected plan and the time intervals (dt)
        plan = getattr(self, var_name)
        plan = np.vstack(plan)

        N = len(plan)
        plan = plan.reshape(N, -1)
        time = np.linspace(start=0., stop=(N+1)*self.sim_dt, num=N)

        # Number of dimensions in the plan (columns)
        num_dimensions = plan.shape[1]

        # Calculate the number of rows needed for the subplots
        num_rows = (num_dimensions + 2) // 3  # +2 to account for remaining dimensions if not divisible by 3

        # Create subplots with 3 columns
        fig, axs = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))
        axs = axs.flatten()  # Flatten the axes for easy iteration

        # Plot each dimension of the plan on a separate subplot
        for i in range(num_dimensions):
            axs[i].plot(time, plan[:, i])
            axs[i].set_title(f'{var_name} dimension {i+1}')
            axs[i].set_xlabel('Time (s)')
            axs[i].set_ylabel(f'{var_name} values')
            axs[i].grid(True)

        # Turn off unused subplots if there are any
        for i in range(num_dimensions, len(axs)):
            fig.delaxes(axs[i])

        plt.tight_layout()
    
    def show_plots(self):
        plt.show()

    def print_timings(self):
        print()
        print_timings(self.timings)
        print_timings(self.solver.timings)