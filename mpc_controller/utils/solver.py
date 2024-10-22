from typing import Any, Dict, List, Tuple
import numpy as np
import pinocchio as pin

from contact_tamp.traj_opt_acados.interface.problem_formuation import ProblemFormulation
# from contact_tamp.traj_opt_acados.models.quadruped import Quadruped
from mj_pin_wrapper.pin_robot import PinQuadRobotWrapper
from contact_tamp.traj_opt_acados.interface.acados_helper import AcadosSolverHelper
from ..config.quadruped.utils import get_quadruped_config
from ..config.config_abstract import MPCCostConfig
from .gait_planner import GaitPlanner
from .dynamics import QuadrupedDynamics
from .transform import *

class QuadrupedAcadosSolver(AcadosSolverHelper):
    NAME = "quadruped_solver"

    def __init__(self,
                 pin_robot : PinQuadRobotWrapper,
                 gait_name : str = "trot",
                 print_info : bool = False,
                 ):
        self.print_info = print_info
        self.pin_robot = pin_robot
        self.robot_name = pin_robot.model.name
        self.feet_frame_names = list(pin_robot.eeff_id2name.values())
        # TODO: no hard code
        self.robot_name = "go2"
        self.feet_frame_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        
        self.config_gait, self.config_opt, self.config_cost = get_quadruped_config(gait_name, self.robot_name)
        dt_min, dt_max = self.config_opt.get_dt_bounds()
        self.dt_nodes = self.config_opt.get_dt_nodes()
        self.enable_time_opt = self.config_opt.enable_time_opt
        problem = ProblemFormulation(self.dt_nodes, dt_min, dt_max, self.enable_time_opt)

        self.dyn = QuadrupedDynamics(self.robot_name, pin_robot.path_urdf, self.feet_frame_names)
        self.dyn.setup(problem)

        # Gait planner
        self.gait_planner = GaitPlanner(self.feet_frame_names, self.dt_nodes, self.config_gait)

        # Nominal state
        self.q0, _ = self.pin_robot.get_state()

        super().__init__(
            problem,
            self.config_opt.n_nodes,
            QuadrupedAcadosSolver.NAME,
            self.config_cost.reg_eps,
            self.config_cost.reg_eps_e,
            )
        
        self.setup(self.config_opt.recompile,
                   self.config_opt.use_cython,
                   self.config_opt.real_time_it,
                   self.config_opt.max_qp_iter)
        self.data = self.get_data_template()

        # Solver warm start
        self.set_max_iter(self.config_opt.max_iter)
        self.set_warm_start_inner_qp(self.config_opt.warm_start_qp)
        self.set_warm_start_nlp(self.config_opt.warm_start_nlp)
        self.set_qp_tol(self.config_opt.qp_tol)
        self.set_nlp_tol(self.config_opt.nlp_tol)
        
        # Init cost
        self.set_cost()
        self.update_cost_weights()

        # Init solution
        self.q_sol_euler = np.zeros_like(self.states[self.dyn.q.name])
        self.v_sol_euler = np.zeros_like(self.states[self.dyn.v.name])
        self.a_sol = np.zeros_like(self.inputs[self.dyn.a.name])
        self.f_sol = np.zeros((self.config_opt.n_nodes, 4, 3))
        self.dt_node_sol = np.zeros((self.config_opt.n_nodes))

        self.default_normal = np.array([0., 0., 1.])
        self.last_node = 0

    def update_cost(self, config_cost : MPCCostConfig):
        """
        Update MPC cost.
        """
        self.config_cost = config_cost
        self.set_cost()
        self.update_cost_weights()

    def set_cost(self):
        """
        Set up the running and terminal cost for the solver using weights from the config file.
        """       
        # Terminal cost weights (W_e) for base position, orientation, and velocity
        self.data["W_e"][self.dyn.base_cost.name] = np.array(self.config_cost.W_e_base)
        # Running cost weights (W) for base position, orientation, and velocity
        self.data["W"][self.dyn.base_cost.name] = np.array(self.config_cost.W_base)
        # Acceleration cost weights
        self.data["W"][self.dyn.acc_cost.name] = np.array(self.config_cost.W_acc)
        # Swing cost weights (for each foot)
        self.data["W"][self.dyn.swing_cost.name] = np.array(self.config_cost.W_swing)
        if self.enable_time_opt:
            self.data["W"]["dt"][0] = 0
        # Joint cost to ref
        self.data["W"][self.dyn.joint_cost.name] = np.array(self.config_cost.W_joint)
        self.data["W_e"][self.dyn.joint_cost.name] = np.array(self.config_cost.W_e_joint)
        self.data["W_e"][self.dyn.swing_cost.name] = np.array(self.config_cost.W_swing)

        # Foot force regularization weights (for each foot)
        for i, foot_cnt in enumerate(self.dyn.feet):
            self.data["W"][foot_cnt.f_reg.name] = np.array(self.config_cost.W_cnt_f_reg[i])
        
        # Apply these costs to the solver
        self.set_cost_weight_constant(self.data["W"])
        self.set_cost_weight_terminal(self.data["W_e"])

    @staticmethod
    def _repeat_last(arr: np.ndarray, n_repeat: int, axis=-1):
        # Get the last values along the specified axis
        last_value = np.take(arr, [-1], axis=axis)
        # Repeat the last values n times along the specified axis
        repeated_values = np.repeat(last_value, n_repeat, axis=axis)
        # Concatenate the original array with the repeated values
        result = np.concatenate([arr, repeated_values], axis=axis)
        return result

    def setup_reference(self,
                        q_base_des : np.ndarray,
                        v_des : np.ndarray,
                        w_yaw : float,
                        ):
        """
        Set up the reference trajectory (yref).
        """       
        # Set the nominal time step
        if self.enable_time_opt:
            self.data["yref"]["dt"][0] = self.dt_nodes
            self.data["u"]["dt"][0] = self.dt_nodes
        
        # Setup reference velocities
        w_des = np.array([0., 0., w_yaw])

        # Base reference and terminal states
        base_ref = np.concatenate((q_base_des, v_des, w_des))
        base_ref_e = base_ref.copy()
        # # Set height to nominal height
        # base_ref_e[2] = self.config_gait.nom_height
        # Reference roll, pitch terminal orientation is horizontal
        # base_ref_e[4:6] = 0.
        # # Yaw orientation according to desired velocity
        # base_ref_e[3] += w_yaw * self.config_opt.time_horizon
        # # Desired final position according to desired vel
        # base_ref_e[:2] += v_des[:2] * self.config_opt.time_horizon

        self.data["yref"][self.dyn.base_cost.name] = base_ref
        self.data["yref"][self.dyn.swing_cost.name][:] = self.config_gait.step_height
        self.data["yref_e"][self.dyn.base_cost.name] = base_ref_e
        self.data["yref_e"][self.dyn.swing_cost.name][:] = self.config_gait.step_height

        # Joint reference is nominal position with zero velocities
        joint_ref = np.concatenate((self.q0[-self.pin_robot.nu:], np.zeros(self.pin_robot.nu)))
        self.data["yref"][self.dyn.joint_cost.name] = joint_ref
        self.data["yref_e"][self.dyn.joint_cost.name] = joint_ref.copy()

        self.set_ref_constant(self.data["yref"])
        self.set_ref_terminal(self.data["yref_e"])

    def setup_initial_state(self,
                            q_euler : np.ndarray,
                            v_euler : np.ndarray | Any = None,
                            set_state_constant : bool = True):
        """
        Initialize the state (x) of the robot in the solver.
        """        
        self.data["x"][self.dyn.q.name] = q_euler
        # Set the state constant in the solver
        if set_state_constant:
            self.set_state_constant(self.data["x"])
            self.set_input_constant(self.data["u"])
        self.set_initial_state(self.data["x"])

        if v_euler is not None:
            self.data["x"][self.dyn.v.name] = v_to_euler_derivative(q_euler, v_euler)
            pin.computeCentroidalMomentum(self.pin_robot.model, self.pin_robot.data)
            self.data["x"][self.dyn.h.name] = self.pin_robot.data.hg.np

    def setup_initial_feet_pos(self,
                               plane_normal : List[np.ndarray] | Any = None,
                               plane_origin : List[np.ndarray] | Any = None,):
        for i_foot, foot_cnt in enumerate(self.dyn.feet):
            # Contact normal
            if plane_normal is None or plane_origin is None:
                self.data["p"][foot_cnt.plane_normal.name] = self.default_normal
            else:
                assert len(plane_normal) == len(plane_origin) == len(self.feet_frame_names),\
                    "plane_normal and plane_origine should be the same length"
                self.data["p"][foot_cnt.plane_point.name] = plane_origin[i_foot]
                self.data["p"][foot_cnt.plane_normal.name] = plane_normal[i_foot]

        for i_foot, foot_cnt in enumerate(self.dyn.feet):
            # Will be overriden by contact plan
            self.data["p"][foot_cnt.active.name][0] = 1
            self.data["p"][foot_cnt.p_gain.name][0] = self.config_cost.foot_pos_constr_stab[i_foot]

        self.set_parameters_constant(self.data["p"])

    def setup_initial_feet_contact(self,
                                   contact_state: Dict[str, int] = {}):
        """
        Set up the initial position of the feet based on the current
        contact mode and robot configuration.
        """
        feet_pos = self.pin_robot.get_frames_position_world(self.feet_frame_names)

        for (foot_cnt, foot_height) in zip(self.dyn.feet, feet_pos[:, 2]):
            # Contact status
            if contact_state:
                is_cnt = contact_state[foot_cnt.frame_name]
            else:
                is_cnt = self.params[foot_cnt.active.name][0, 0]

            if is_cnt == 1:
                self.params[foot_cnt.plane_point.name][2, :] = foot_height

                if self.print_info: print(f'Reset foot contact {foot_cnt.frame_name}, height {foot_height}')

    # TODO: Add contact positions
    def setup_gait_contacts(self, i_node: int = 0):
        """
        Setup contact status in the optimization nodes.
        """
        # Offset applied to the current node to setup the optimization window
        node_w = 0
        switch_node = 0
        
        # While in the current optimization window scheme
        while node_w < self.config_opt.n_nodes:

            # Impact at switch
            if self.config_opt.enable_impact_dyn:
                switch_node = self.gait_planner.next_switch_in(i_node + node_w) + node_w
                if switch_node <= self.config_opt.n_nodes:
                    self.params[self.dyn.impact_active.name][:, switch_node] = 0

            # Setup swing constraints
            last_node = 0
            for foot_cnt in self.dyn.feet:

                # start, peak, end node within the gait
                n_start, n_peak, n_end = self.gait_planner.get_swing_start_peak_end(foot_cnt.frame_name, node_w + i_node)

                # If swinging
                if 0 <= n_start < n_end:
                    # Apply offset
                    n_start += node_w
                    n_end += node_w
                    self.params[foot_cnt.active.name][:, n_start : n_end] = 0 # Not active

                    if (self.config_opt.opt_peak and
                        n_peak > 0):
                        n_peak += node_w
                        if n_peak < self.config_opt.n_nodes:
                            self.params[foot_cnt.peak.name][:, n_peak] = 1 # Peak constrained (not necessary)

                if n_end > last_node:
                    last_node = n_end
                # Full contact phase, get to next switch
                elif last_node == 0:
                    last_node = self.gait_planner.next_switch_in(i_node + node_w) + node_w

            # Update last node of optimization window updated    
            node_w = last_node

        START_TROT_NODE = 10
        if i_node < START_TROT_NODE:
            n_end = START_TROT_NODE - i_node
            for foot_cnt in self.dyn.feet:
                self.params[foot_cnt.active.name][:, :n_end] = 1

    def print_contact_constraints(self):
        print("\nContacts")
        for foot_cnt in self.dyn.feet:
            print(foot_cnt.frame_name, "contact")
            print(self.params[foot_cnt.active.name])

        print("\nImpacts")
        print(self.params[self.dyn.impact_active.name])

    def warm_start_traj(self,
                        start_node: int
                        ):
        """
        Warm start solver with the solution starting after
        start_node of the last solution.

        Args:
            start_node (int): Will use optimized node values after <start_node>
        """       
        # Warm start first values with last solution
        # q, v, a, forces, dt
        n_warm_start = self.config_opt.n_nodes - start_node
        if self.print_info: print(f"Warm start size: {n_warm_start}, start_node {start_node}")

        self.states[self.dyn.q.name][:, 1:n_warm_start+1] = self.q_sol_euler[start_node:].T
        self.states[self.dyn.v.name][:, 1:n_warm_start+1] = self.v_sol_euler[start_node:].T
        self.states[self.dyn.h.name][:, 1:n_warm_start+1] = self.h_sol[start_node:].T
        self.inputs[self.dyn.a.name][:, :n_warm_start] = self.a_sol[start_node:].T
        for i, foot_name in enumerate(self.feet_frame_names):
            self.inputs[f"f_{foot_name}_{self.dyn.name}"][:, :n_warm_start] = self.f_sol[start_node:, i, :].T
        if self.enable_time_opt:
            self.inputs["dt"][:, :n_warm_start] = self.dt_node_sol[start_node:]

        # Set the remaining values
        if n_warm_start < self.config_opt.n_nodes:
            last_q = self.q_sol_euler[-1]
            self.states[self.dyn.q.name][:, n_warm_start+1:] = last_q[:, None]
            self.states[self.dyn.v.name][:, n_warm_start+1:] = 0.
            self.states[self.dyn.h.name][:, n_warm_start+1:] = 0.
            self.inputs[self.dyn.a.name][:, n_warm_start:] = 0.
            for i, foot_name in enumerate(self.feet_frame_names):
                self.inputs[f"f_{foot_name}_{self.dyn.name}"][:, n_warm_start:] = 0.
            # Nominal dt
            if self.enable_time_opt:
                self.inputs["dt"][:, n_warm_start:] = self.dt_nodes
            
    def init(self,
              q : np.ndarray,
              v : np.ndarray | Any = None,
              i_node : int = 0,
              q_des : np.ndarray = np.zeros(6), 
              v_des : np.ndarray = np.zeros(3),
              w_yaw_des : float = 0.,
              contact_state : Dict[str, int] = {},
              ):
        """
        Setup solver depending on the current configuration and the 
        current optimization node.
        """
        self.pin_robot.update(q, v)
        first_it = i_node == 0
        q_euler = quat_to_ypr_state(q)
        self.setup_reference(q_des, v_des, w_yaw_des)
        self.setup_initial_state(q_euler, v, first_it)
        self.setup_initial_feet_pos()
        self.setup_gait_contacts(i_node)
        self.setup_initial_feet_contact(contact_state)

        if self.print_info: self.print_contact_constraints()

        # Warm start nodes
        if (not first_it and
            self.config_opt.warm_start_sol):
            warm_start_node = (i_node - self.last_node) % self.config_opt.n_nodes
            self.warm_start_traj(warm_start_node)
            self.update_states()
            self.update_inputs()

        self.update_parameters()
        self.update_ref()

        self.last_node = i_node

    def solve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve the optimization problem and parse solution.
        """
        super().solve(print_stats=self.print_info, print_time=self.print_info)
        self.parse_sol()

        self.q_sol_euler = self.states[self.dyn.q.name].T[1:, :].copy()

        # positions, [n_nodes, 19]
        q_sol = ypr_to_quat_state_batched(self.q_sol_euler)
        
        # velocities, [n_nodes, 18]
        self.v_sol_euler = self.states[self.dyn.v.name].T[1:, :].copy()
        v_sol = v_to_local_angular_batched(self.q_sol_euler, self.v_sol_euler)

        # Centroidal momentum, [n_nodes, 6]
        self.h_sol = self.states[self.dyn.h.name].T[1:, :].copy()

        # acceleration, [n_nodes, 18]
        self.a_sol = self.inputs[self.dyn.a.name].T.copy()

        # end effector forces, [n_nodes, 4, 3]
        self.f_sol = np.array([
            self.params[foot_cnt.active.name][:, 1:] * self.inputs[f"f_{foot_cnt.frame_name}_{self.dyn.name}"]
            for foot_cnt in self.dyn.feet
        ]).transpose(2, 0, 1).copy()

        # dt time, [n_nodes, ]
        if self.enable_time_opt:
            self.dt_node_sol = self.inputs["dt"].flatten()
        else:
            self.dt_node_sol = np.full((len(q_sol),), self.dt_nodes)

        return q_sol, v_sol, self.a_sol, self.f_sol, self.dt_node_sol
