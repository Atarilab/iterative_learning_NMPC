from collections import defaultdict
from typing import Any, Dict, List, Tuple
import numpy as np
import pinocchio as pin
import time

from contact_tamp.traj_opt_acados.interface.problem_formuation import ProblemFormulation
# from contact_tamp.traj_opt_acados.models.quadruped import Quadruped
from mj_pin_wrapper.pin_robot import PinQuadRobotWrapper
from contact_tamp.traj_opt_acados.interface.acados_helper import AcadosSolverHelper
from ..config.quadruped.utils import get_quadruped_config
from ..config.config_abstract import MPCCostConfig
from .gait_planner import GaitPlanner
from .raibert_contact_planner import RaiberContactPlanner
from .dynamics import QuadrupedDynamics
from .transform import *
from .profiling import time_fn, print_timings

class QuadrupedAcadosSolver(AcadosSolverHelper):
    NAME = "quadruped_solver"
    DEFAULT_RANGE_RADIUS = 0.01
    STAND_PHASE_NODES = 0

    def __init__(self,
                 pin_robot : PinQuadRobotWrapper,
                 gait_name : str = "trot",
                 print_info : bool = False,
                 height_offset : float = 0.,
                 compute_timings : bool = True,
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

        self.dyn = QuadrupedDynamics(
            self.robot_name,
            pin_robot.path_urdf,
            self.feet_frame_names,
            not(self.config_opt.opt_cnt_pos)
            )
        
        self.dyn.setup(problem)

        # Gait planner
        self.gait_planner = GaitPlanner(self.feet_frame_names, self.dt_nodes, self.config_gait)
        # Contact planner
        q, _ = self.pin_robot.get_state()

        offset_hip_b = (self.pin_robot.get_hip_pos_world() - q[None, :3])
        self.contact_planner = RaiberContactPlanner(offset_hip_b, self.config_gait, height_offset)

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

        # Setup timings
        self.compute_timings = compute_timings
        self.timings = defaultdict(list)

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
            self.data["W"]["dt"][0] = 1e4
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
                        q_euler : np.ndarray,
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

        base_ref = q_euler[:12].copy()

        # Horizontal base
        base_ref[4:6] = 0.
        # Height to config
        base_ref[2] = self.config_gait.nom_height

        # Setup reference velocities in local frame
        # v_des is in local frame already
        # w_yaw in global frame
        R_WB = pin.rpy.rpyToMatrix(q_euler[3:6][::-1])
        w_des_global = np.array([0., 0., w_yaw])
        w_des_local = (R_WB.T @ w_des_global)[::-1]
        base_ref[-3:] = np.round(w_des_local, 2)

        # Compute velocity in global frame
        # Apply angular velocity
        R_WB = pin.rpy.rpyToMatrix(base_ref[3:6][::-1])
        R_yaw = pin.rpy.rpyToMatrix(w_des_global * self.config_opt.time_horizon)
        v_des = R_yaw @ v_des
        # Vertical velocity to reach desired height
        base_ref[6:9] = v_des
        v_des_glob = np.round(R_WB @ v_des, 2)

        # Update position and orientation according to desired velocities
        base_ref[:3] += v_des_glob * self.config_opt.time_horizon
        base_ref[3] += w_yaw * self.config_opt.time_horizon

        # Base reference and terminal states
        base_ref_e = base_ref.copy()
        # Base height
        base_ref_e[2] = self.config_gait.nom_height
        # Base horizontal vel
        base_ref_e[8] = 0. 
        # pitch roll vel
        base_ref_e[-2:] = 0. 

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
                            v_local : np.ndarray | Any = None,
                            set_state_constant : bool = True):
        """
        Initialize the state (x) of the robot in the solver.
        """        
        self.data["x"][self.dyn.q.name] = q_euler
        if v_local is not None:
            self.data["x"][self.dyn.v.name] = v_to_euler_derivative(q_euler, v_local)
            pin.computeCentroidalMomentum(self.pin_robot.model, self.pin_robot.data)
            self.data["x"][self.dyn.h.name] = self.pin_robot.data.hg.np
        # Set the state constant in the solver
        if set_state_constant:
            self.set_state_constant(self.data["x"])
            self.set_input_constant(self.data["u"])
        self.set_initial_state(self.data["x"])

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
            if not self.config_opt.opt_cnt_pos:
                self.data["p"][foot_cnt.restrict.name][0] = 1

        self.set_parameters_constant(self.data["p"])

    def setup_initial_feet_contact(self,
                                   contact_state: Dict[str, int] = {}):
        """
        Set up the initial position of the feet based on the current
        contact mode and robot configuration.
        """
        feet_pos = self.pin_robot.get_frames_position_world(self.feet_frame_names)

        for (foot_cnt, pos) in zip(self.dyn.feet, feet_pos):
            # Contact status
            if contact_state:
                is_cnt = contact_state[foot_cnt.frame_name]
            else:
                is_cnt = self.params[foot_cnt.active.name][0, 0]

            if is_cnt == 1:
                if self.config_opt.opt_cnt_pos:
                    self.params[foot_cnt.plane_point.name][2, :] = pos[2]
                    if self.print_info: print(f'Reset foot contact {foot_cnt.frame_name}, height {pos[2]}')
                
                else:
                    # Will be override by contact plan
                    self.params[foot_cnt.plane_point.name][:, :] = pos[:, None]
                    self.params[foot_cnt.range_radius.name][:, :] = QuadrupedAcadosSolver.DEFAULT_RANGE_RADIUS

    @time_fn("setup_gait_contacts")
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
                    self.params[self.dyn.impact_active.name][:, switch_node] = 1

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
                    
            # Full contact phase, go to next switch
            if last_node == 0:
                last_node = self.gait_planner.next_switch_in(i_node + node_w) + node_w

            # Update last node of optimization window updated    
            node_w = last_node

    @time_fn("setup_contact_locations")
    def setup_contact_locations(self,
                                q : np.ndarray,
                                i_node : int,
                                v_des : np.ndarray = np.zeros(3),
                                w_yaw : float = 0.,
                                ):
        
        # Offset applied to the current node to setup the optimization window
        node_w = 0
        
        com_xyz = pin.centerOfMass(self.pin_robot.model, self.pin_robot.data)

        # While in the current optimization window scheme
        while node_w < self.config_opt.n_nodes:

            # Setup position contact constraints
            last_node = 0
            for i_foot, foot_cnt in enumerate(self.dyn.feet):

                # start, peak, end node within the gait
                n_start, _, n_end = self.gait_planner.get_swing_start_peak_end(foot_cnt.frame_name, node_w + i_node)

                # If swinging, the last swinging node is the first cnt node
                if 0 <= n_start < n_end:
                    # Offset to the current node in the window
                    n_end += node_w
                    # Time to reach contact
                    time_to_cnt = n_end * self.dt_nodes

                    # Set contact location constraint
                    cnt_loc_w = self.contact_planner.next_contact_location(i_foot, q, com_xyz, time_to_cnt, v_des, w_yaw)
                    self.params[foot_cnt.plane_point.name][:2, n_end:] = cnt_loc_w[:2, None]
                    self.params[foot_cnt.plane_point.name][2, n_end:] = 0.0

                if n_end > last_node:
                    last_node = n_end
                    
            # Full contact phase, go to next switch
            if last_node == 0:
                last_node = self.gait_planner.next_switch_in(i_node + node_w) + node_w

            # Update last node of optimization window updated    
            node_w = last_node

    def set_stand_phase(self, n_nodes : int):
        """
        Set up a stand phase for the n_nodes first nodes.
        """
        for foot_cnt, foot_pos in zip(self.dyn.feet, self.pin_robot.get_foot_pos_world()):
            self.params[foot_cnt.active.name][:, :n_nodes] = 1 # Active
            self.params[foot_cnt.plane_point.name][:2, :n_nodes] = foot_pos[:2, None]

    def print_contact_constraints(self):
        print("\nContacts")
        for foot_cnt in self.dyn.feet:
            print(foot_cnt.frame_name, "contact")
            print(self.params[foot_cnt.active.name])
            print(np.unique(self.params[foot_cnt.plane_point.name], axis=1).T)

        print("\nImpacts")
        print(self.params[self.dyn.impact_active.name])

    @time_fn("warm_start_traj")
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

        # States [n_nodes + 1]
        self.states[self.dyn.q.name][:, 1:n_warm_start+1] = self.q_sol_euler[start_node+1:].T
        self.states[self.dyn.v.name][:, 1:n_warm_start+1] = self.v_sol_euler[start_node+1:].T
        self.states[self.dyn.h.name][:, 1:n_warm_start+1] = self.h_sol[start_node+1:].T
        # Inputs [n_nodes]
        self.inputs[self.dyn.a.name][:, :n_warm_start] = self.a_sol[start_node:].T
        for i, foot_name in enumerate(self.feet_frame_names):
            self.inputs[f"f_{foot_name}_{self.dyn.name}"][:, :n_warm_start] = self.f_sol[start_node:, i, :].T
        if self.enable_time_opt:
            self.inputs["dt"][:, :n_warm_start] = self.dt_node_sol[start_node:]

        # Set the remaining values
        if n_warm_start < self.config_opt.n_nodes:
            last_q = self.q_sol_euler[-1]
            last_q[2] = self.config_gait.nom_height
            last_v = self.v_sol_euler[-1]
            last_h = self.h_sol[-1]
            last_a = self.a_sol[-1]
            self.states[self.dyn.q.name][:, n_warm_start+1:] = last_q[:, None]
            self.states[self.dyn.v.name][:, n_warm_start+1:] = last_v[:, None]
            self.states[self.dyn.h.name][:, n_warm_start+1:] = last_h[:, None]
            self.inputs[self.dyn.a.name][:, n_warm_start:] = last_a[:, None]
            for i, foot_name in enumerate(self.feet_frame_names):
                self.inputs[f"f_{foot_name}_{self.dyn.name}"][:, n_warm_start:] = 0.
            # Nominal dt
            if self.enable_time_opt:
                self.inputs["dt"][:, n_warm_start:] = self.dt_nodes
    
    @time_fn("init_solver")
    def init(self,
              q : np.ndarray,
              v : np.ndarray = np.zeros(18),
              i_node : int = 0,
              v_des : np.ndarray = np.zeros(3),
              w_yaw_des : float = 0.,
              contact_state : Dict[str, int] = {},
              ):
        """
        Setup solver depending on the current configuration and the 
        current optimization node.
        """
        # [v_global, w_local, v_joint] -> [v_local, w_local, v_joint]
        v_local = v_global_linear_to_local_linear(q, v)
        self.pin_robot.update(q, v_local)
        first_it = i_node == 0
        q_euler = quat_to_ypr_state(q)

        self.setup_reference(q_euler, v_des, w_yaw_des)
        self.setup_initial_state(q_euler, v_local, first_it)
        self.setup_initial_feet_pos()
        self.setup_gait_contacts(i_node)
        self.setup_initial_feet_contact(contact_state)
        if not self.config_opt.opt_cnt_pos:
            self.setup_contact_locations(q, i_node, v_des, w_yaw_des)

        if i_node < QuadrupedAcadosSolver.STAND_PHASE_NODES:
            self.set_stand_phase(QuadrupedAcadosSolver.STAND_PHASE_NODES - i_node)

        if self.print_info: self.print_contact_constraints()

        # Warm start nodes
        if (not first_it and
            self.config_opt.warm_start_sol):
            warm_start_node = (i_node - self.last_node) % self.config_opt.n_nodes
            self.warm_start_traj(warm_start_node)
            # self.update_states()
            # self.update_inputs()

        self.update_parameters()
        # self.update_ref()

        self.last_node = i_node

    @time_fn("solve")
    def solve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve the optimization problem and parse solution.
        returns pinocchio states with FreeFlyer
        """
        super().solve(print_stats=self.print_info, print_time=self.print_info)
        self.parse_sol()

        self.q_sol_euler = self.states[self.dyn.q.name].T.copy()

        # positions, [n_nodes + 1, 19]
        q_sol = ypr_to_quat_state_batched(self.q_sol_euler)
        
        # velocities, [n_nodes + 1, 18]
        self.v_sol_euler = self.states[self.dyn.v.name].T.copy()
        v_sol = v_glob_to_local_batched(self.q_sol_euler, self.v_sol_euler)

        # centroidal momentum, [n_nodes + 1, 6]
        self.h_sol = self.states[self.dyn.h.name].T.copy()

        # acceleration, [n_nodes, 18]
        self.a_sol = self.inputs[self.dyn.a.name].T.copy()

        # end effector forces, [n_nodes, 4, 3]
        self.f_sol = np.array([
            self.inputs[f"f_{foot_cnt.frame_name}_{self.dyn.name}"]
            for foot_cnt in self.dyn.feet
        ]).transpose(2, 0, 1).copy()

        # dt time, [n_nodes, ]
        if self.enable_time_opt:
            self.dt_node_sol = self.inputs["dt"].flatten().copy()
        else:
            self.dt_node_sol = np.full((len(q_sol),), self.dt_nodes)

        return q_sol, v_sol, self.a_sol, self.f_sol, self.dt_node_sol
    
    def print_timings(self):
        print_timings(self.timings)