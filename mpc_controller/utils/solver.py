from collections import defaultdict
from typing import Any, Dict, List, Tuple
import numpy as np
import pinocchio as pin
import time

from contact_tamp.traj_opt_acados.interface.problem_formuation import ProblemFormulation
from contact_tamp.traj_opt_acados.interface.acados_helper import AcadosSolverHelper
from ..config.quadruped.utils import get_quadruped_config
from ..config.config_abstract import MPCCostConfig
from .gait_planner import GaitPlanner
from .raibert_contact_planner import RaiberContactPlanner
from .dynamics import QuadrupedDynamics
from .transform import *
from .profiling import time_fn, print_timings
from mj_pin.utils import pin_frame_pos

class QuadrupedAcadosSolver(AcadosSolverHelper):
    NAME = "quadruped_solver"

    def __init__(self,
                 pin_model,
                 pin_data,
                 path_urdf : str,
                 feet_frame_names : List[str],
                 gait_name : str = "trot",
                 print_info : bool = False,
                 compute_timings : bool = True,
                 ):
        
        self.pin_model = pin_model
        self.pin_data = pin_data
        self.robot_name = pin_model.name
        self.feet_frame_names = feet_frame_names
        self.print_info = print_info

        self.config_gait, self.config_opt, self.config_cost = get_quadruped_config(gait_name, self.robot_name)
        dt_min, dt_max = self.config_opt.get_dt_bounds()
        self.dt_nodes = self.config_opt.get_dt_nodes()
        self.enable_time_opt = self.config_opt.enable_time_opt

        problem = ProblemFormulation(self.dt_nodes, dt_min, dt_max, self.enable_time_opt)

        self.dyn = QuadrupedDynamics(
            path_urdf,
            self.feet_frame_names,
            self.config_opt.cnt_patch_restriction,
            )

        self.dyn.setup(problem)

        # Gait planner
        self.gait_planner = GaitPlanner(self.feet_frame_names, self.dt_nodes, self.config_gait)

        # Contact planner
        # q, _ = self.pin_model.get_state()
        # offset_hip_b = (self.pin_model.get_hip_pos_world() - q[None, :3])
        # self.contact_planner = RaiberContactPlanner(
        #     offset_hip_b,
        #     self.config_gait,
        #     0.,
        #     y_offset=0.1,
        #     x_offset=0.025)

        # Init solver
        super().__init__(
            problem,
            self.config_opt.n_nodes,
            QuadrupedAcadosSolver.NAME,
            self.config_cost.reg_eps,
            self.config_cost.reg_eps_e,
            )
    
        # Solver params
        self.setup(self.config_opt.recompile,
                   self.config_opt.use_cython,
                   self.config_opt.real_time_it,
                   self.config_opt.max_qp_iter,
                   self.config_opt.hpipm_mode)
        self.set_max_iter(self.config_opt.max_iter)
        self.set_warm_start_inner_qp(self.config_opt.warm_start_qp)
        self.set_warm_start_nlp(self.config_opt.warm_start_nlp)
        self.set_qp_tol(self.config_opt.qp_tol)
        self.set_nlp_tol(self.config_opt.nlp_tol)

        self.data = self.get_data_template()

        # Init cost
        self.set_cost_weights()
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
        self.set_cost_weights()
        self.update_cost_weights()

    def set_cost_weights(self):
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
        # Joint cost to ref
        self.data["W"][self.dyn.joint_cost.name] = np.array(self.config_cost.W_joint)
        self.data["W_e"][self.dyn.joint_cost.name] = np.array(self.config_cost.W_e_joint)
        self.data["W_e"][self.dyn.swing_cost.name] = np.array(self.config_cost.W_swing)
        if self.enable_time_opt:
            self.data["W"]["dt"][0] = np.array(self.config_cost.time_opt)

        # Foot force regularization weights (for each foot)
        for i, foot_cnt in enumerate(self.dyn.feet):
            self.data["W"][foot_cnt.f_reg.name] = np.array(self.config_cost.W_cnt_f_reg[i])

            # Foot displacement penalization
            if self.config_opt.cnt_patch_restriction:
                self.data["W"][foot_cnt.pos_cost.name] = self.config_cost.W_foot_displacement
                self.data["W_e"][foot_cnt.pos_cost.name] = self.config_cost.W_foot_displacement

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
                        base_ref : np.ndarray,
                        base_ref_e : np.ndarray,
                        joint_ref : np.ndarray,
                        ):
        """
        Set up the reference trajectory (yref).
        """
        if base_ref_e is None:
            base_ref_e = base_ref.copy()
        
        # Set the nominal time step
        if self.enable_time_opt:
            self.cost_ref["dt"][:] = self.dt_nodes

        self.cost_ref[self.dyn.base_cost.name][:] = base_ref[:, None]
        self.cost_ref[self.dyn.swing_cost.name][:] = self.config_gait.step_height
        self.data["yref_e"][self.dyn.base_cost.name] = base_ref_e
        self.data["yref_e"][self.dyn.swing_cost.name][:] = self.config_gait.step_height

        # Joint reference is nominal position with zero velocities
        joint_ref_vel = np.concatenate((joint_ref, np.zeros_like(joint_ref)))
        self.cost_ref[self.dyn.joint_cost.name] = joint_ref_vel[:, None]
        self.data["yref_e"][self.dyn.joint_cost.name] = joint_ref_vel.copy()

        self.set_ref_terminal(self.data["yref_e"])

    def setup_initial_state(self,
                            q_euler : np.ndarray,
                            v_local : np.ndarray | Any = None):
        """
        Initialize the state (x) of the robot in the solver.
        """        
        self.data["x"][self.dyn.q.name] = q_euler

        if v_local is not None:
            # v local, w local -> v world, v euler world
            self.data["x"][self.dyn.v.name] = v_to_euler_derivative(q_euler, v_local)
            self.data["x"][self.dyn.h.name] = self.pin_data.hg.np

        self.set_initial_state(self.data["x"])

        if self.config_opt.enable_time_opt:
            self.inputs["dt"][:] = self.dt_nodes

    def setup_initial_feet_pos(self,
                                first_it : bool = False,
                                contact_state: Dict[str, int] = {},
                                ):
        """
        Set up the initial position of the feet based on the current
        contact mode and robot configuration.
        """
        feet_pos = np.array([
            pin_frame_pos(self.pin_model, self.pin_data, f_name)
            for f_name in
            self.feet_frame_names])

        for (foot_cnt, pos) in zip(self.dyn.feet, feet_pos):
            # Contact status
            if contact_state:
                is_cnt = contact_state[foot_cnt.frame_name]
            else: # From the contact planner
                is_cnt = self.params[foot_cnt.active.name][0, 0]

            # If contact, setup initial contact location and normal
            if is_cnt == 1 or first_it:
                next_swing = np.argmin(self.params[foot_cnt.active.name][0, :])
                self.params[foot_cnt.plane_point.name][:, :next_swing+1] = pos[:, None]
                if self.print_info: print(f'Reset foot contact {foot_cnt.frame_name}, height {pos[2]}')
    
    def init_contacts_parameters(self):
        # Fill contact parameters, will be overriden by gait planner
        for i_foot, foot_cnt in enumerate(self.dyn.feet):
            self.params[foot_cnt.active.name][:] = 1.
            self.params[foot_cnt.plane_normal.name][:] = self.default_normal[:, None]
            self.params[foot_cnt.plane_point.name][:] = np.zeros((3,1))
            self.params[foot_cnt.p_gain.name][:] = self.config_cost.W_foot_pos_constr_stab[i_foot]
            
            if (self.config_opt.cnt_patch_restriction):
                self.params[foot_cnt.restrict.name][:] = 0.
                if (self.config_opt.cnt_patch_restriction):
                    self.params[foot_cnt.range_radius.name][:] = self.config_cost.cnt_radius
            
    @time_fn("setup_gait_contacts")
    def setup_gait_contacts(self, i_node: int = 0):
        """
        Setup contact status in the optimization nodes.
        """
        self.init_contacts_parameters()

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
                    # Apply window offset
                    n_start += node_w
                    n_end += node_w
                    self.params[foot_cnt.active.name][:, n_start : n_end] = 0 # Not active
                    
                    # Set peak constraints
                    if (self.config_opt.opt_peak and
                        n_peak >= 0):
                        n_peak += node_w
                        if n_peak < self.config_opt.n_nodes:
                            self.params[foot_cnt.peak.name][:, n_peak] = 1

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
        time = i_node * self.dt_nodes
        self.contact_planner.remove_cnt_before(time)
        com_xyz = pin.centerOfMass(self.pin_model, self.pin_data)

        # Offset applied to the current node to setup the optimization window
        node_w = 0
        
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

                    if n_end <= self.config_opt.n_nodes:
                        # Time to reach contact
                        time_cnt = time + n_end * self.dt_nodes

                        # Set contact location constraint
                        cnt_loc_w = self.contact_planner.next_contact_location(
                            i_foot,
                            q,
                            com_xyz,
                            time,
                            time_cnt,
                            v_des, w_yaw)
                        self.params[foot_cnt.plane_point.name][:, n_end:] = cnt_loc_w[:, None]
                        self.cost_ref[foot_cnt.pos_cost.name][:, n_end:] = cnt_loc_w[:, None]

                        # Set restriction constraint (only on the first contact node)
                        self.params[foot_cnt.restrict.name][:, n_end] = 1
                            
                if n_end > last_node:
                    last_node = n_end

            # Full contact phase, go to next switch
            if last_node == 0:
                last_node = self.gait_planner.next_switch_in(i_node + node_w) + node_w

            # Update last node of optimization window updated
            node_w = last_node

        # Set reference for terminal contact
        for i_foot, foot_cnt in enumerate(self.dyn.feet):
            self.cost_ref_terminal[foot_cnt.pos_cost.name] = self.params[foot_cnt.plane_point.name][:, -1]

    def print_contact_constraints(self):
        print("\nContacts")
        for foot_cnt in self.dyn.feet:
            print(foot_cnt.frame_name, "contact")
            print(self.params[foot_cnt.active.name])
            print(np.unique(self.params[foot_cnt.plane_point.name], axis=1).T)

        if self.config_opt.cnt_patch_restriction:
            print("\nRestriction")
            print(self.params[foot_cnt.restrict.name])

    @time_fn("warm_start_solver")
    def warm_start_solver(self,
                        i_node: int,
                        repeat_last : bool = False,
                        ):
        """
        Warm start solver with the solution starting after
        start_node of the last solution.

        Args:
            i_node (int): Current optimization node.
            repeat_last (bool): Repeat last values for the last non warm started nodes.
        """

        # Warm start first values with last solution
        # q, v, a, forces, dt
        start_node = i_node - self.last_node
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
            if repeat_last:
                self.inputs[f"f_{foot_name}_{self.dyn.name}"][:, n_warm_start:] = 0.

        # dt [n_nodes]
        if self.enable_time_opt:
            self.inputs["dt"][:, :n_warm_start] = self.dt_node_sol[start_node:]

        if repeat_last and n_warm_start < self.config_opt.n_nodes:
            self.states[self.dyn.q.name][:, n_warm_start:] = self.q_sol_euler[None, -1].T
            self.states[self.dyn.v.name][:, n_warm_start:] = self.v_sol_euler[None, -1].T
            self.states[self.dyn.h.name][:, n_warm_start:] = self.h_sol[None, -1].T
            self.inputs[self.dyn.a.name][:, n_warm_start:] = self.a_sol[None, -1].T
            
            # set to nominal dt
            if self.enable_time_opt:
                self.inputs["dt"][:, n_warm_start:] = self.dt_nodes

        # Dual variables
        self.warm_start_dual("lam", start_node, repeat_last=True)

        # Update last opt node
        self.last_node = i_node
    
    def update_pin(self, q: np.array, v:np.array) -> None:
        """
        Update pin data.
        """
        pin.forwardKinematics(self.pin_model, self.pin_data, q)
        pin.computeCentroidalMomentum(self.pin_model, self.pin_data)

    @time_fn("update_solver")
    def update_solver(self) -> None:
        """
        Set data in the solver.
        """
        self.update_states()
        self.update_inputs()
        self.update_parameters()
        self.update_ref()

    @time_fn("init_solver")
    def init(self,
              q : np.ndarray,
              v : np.ndarray,
              base_ref : np.ndarray,
              base_ref_e : np.ndarray,
              joint_ref : np.ndarray,
              i_node : int = 0,
              v_des : np.ndarray = np.zeros(3),
              w_yaw_des : float = 0.,
              contact_state : Dict[str, int] = {},
              ):
        """
        Setup solver depending on the current configuration and the
        current optimization node.
        """
        first_it = i_node == 0

        self.update_pin(q, v)

        # Base reference position and velocities in world frame
        self.setup_reference(base_ref, base_ref_e, joint_ref)

        # q quat [x, y, z, w] -> q euler [z, y, x]
        q_euler = quat_to_ypr_state(q)
        self.setup_initial_state(q_euler, v)
        
        # Setup contact sequence and locations
        self.setup_gait_contacts(i_node)
        self.setup_initial_feet_pos(first_it, contact_state)
        if self.config_opt.cnt_patch_restriction:
            self.setup_contact_locations(q, i_node, v_des, w_yaw_des)

        if self.print_info: self.print_contact_constraints()

        # Warm start solver
        if (not first_it and
            self.config_opt.warm_start_sol):
            self.warm_start_solver(i_node)

        self.update_solver()

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
            self.dt_node_sol = np.full((len(self.a_sol),), self.dt_nodes)

        return q_sol, v_sol, self.a_sol, self.f_sol, self.dt_node_sol

    def print_timings(self):
        print_timings(self.timings)