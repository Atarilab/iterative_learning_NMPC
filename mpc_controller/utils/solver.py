from typing import Any, List, Tuple
import numpy as np
import pinocchio as pin

from contact_tamp.traj_opt_acados.interface.problem_formuation import ProblemFormulation
# from contact_tamp.traj_opt_acados.models.quadruped import Quadruped
from mj_pin_wrapper.pin_robot import PinQuadRobotWrapper
from contact_tamp.traj_opt_acados.interface.acados_helper import AcadosSolverHelper
from ..config.quadruped.utils import get_quadruped_config
from .gait_planner import GaitPlanner
from .dynamics import QuadrupedDynamics

class QuadrupedAcadosSolver(AcadosSolverHelper):
    NAME = "quadruped_solver"

    def __init__(self,
                 pin_robot : PinQuadRobotWrapper,
                 gait_name : str = "trot",
                 recompile : bool = True,
                 use_cython : bool = False,
                 ):
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
        # self.dyn = Quadruped(pin_robot.path_urdf)
        self.dyn.setup(problem)

        super().__init__(
            problem,
            self.config_opt.n_nodes,
            QuadrupedAcadosSolver.NAME,
            self.config_opt.reg_eps,
            self.config_opt.reg_eps_e,
            )
        
        self.setup(recompile,
                   use_cython,
                   self.config_opt.real_time_it)
        self.data = self.get_data_template()

        # Gait planner
        self.gait_planner = GaitPlanner(self.feet_frame_names, self.dt_nodes, self.config_gait)
    
    def setup_cost(self):
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

        # Foot force regularization weights (for each foot)
        for i, foot_cnt in enumerate(self.dyn.feet):
            self.data["W"][foot_cnt.f_reg.name] = np.array(self.config_cost.W_cnt_f_reg[i])
        
        # Apply these costs to the solver
        self.set_cost_weight_constant(self.data["W"])
        self.set_cost_weight_terminal(self.data["W_e"])

    @staticmethod
    def _quat_to_euler_state(q_full: np.ndarray):
        # Preallocate q_euler with all necessary values directly
        q_euler = np.hstack([
            q_full[:3],
            pin.rpy.matrixToRpy(pin.Quaternion(q_full[6], *q_full[3:6]).toRotationMatrix()),
            q_full[-12:]
            ])

        return q_euler

    @staticmethod
    def _euler_to_quat_state(q_euler: np.ndarray):
        # Preallocate q_full with all necessary values directly
        q_full = np.hstack([
            q_euler[:3],
            pin.Quaternion(pin.rpy.rpyToMatrix(q_euler[3:6])).coeffs(),
            q_euler[6:]
            ])

        return q_full

    def setup_reference(self,
                        q : np.ndarray,
                        v_des : np.ndarray,
                        w_yaw : float = 0.
                        ):
        """
        Set up the reference trajectory (yref).
        """       
        # Set the nominal time step
        if self.enable_time_opt:
            self.data["yref"]["dt"][0] = self.dt_nodes
            self.data["u"]["dt"][0] = self.dt_nodes
            
        # Setup reference base pose
        pos, quat = q[:3], q[3:7]
        # Set height to nominal height
        pos[2] = self.config_gait.nom_height
        R = pin.Quaternion(quat).toRotationMatrix()
        rpy_angles = pin.rpy.matrixToRpy(R)
        # Reference orientation is horizontal
        rpy_angles[:2] = 0.

        # Setup reference velocities
        w_des = np.array([0., 0., w_yaw])

        # Base reference and terminal states
        base_ref = np.concatenate((pos, rpy_angles, v_des, w_des))
        base_ref_e = base_ref.copy()
        base_ref_e[:2] += v_des[:2] * self.config_opt.time_horizon
        base_ref_e[6] += w_yaw * self.config_opt.time_horizon

        self.data["yref"][self.dyn.base_cost.name] = base_ref
        self.data["yref_e"][self.dyn.base_cost.name] = base_ref_e

        self.set_ref_constant(self.data["yref"])
        self.set_ref_terminal(self.data["yref_e"])

    def setup_initial_state(self, q : np.ndarray, v : np.ndarray | Any = None):
        """
        Initialize the state (x) of the robot in the solver.
        """        
        # Use the initial state q0 defined in the cost config
        self.data["x"][self.dyn.q.name] = self._quat_to_euler_state(q)

        if v is not None:
            self.data["x"][self.dyn.v.name] = v
        
        # Set the state constant in the solver
        self.set_state_constant(self.data["x"])
        self.set_initial_state(self.data["x"])
        self.set_input_constant(self.data["u"])

    def setup_initial_feet_pos(self,
                               i_node : int = 0,
                               plane_normal : List[np.ndarray] | Any = None,
                               plane_origin : List[np.ndarray] | Any = None,
                               ):
        """
        Set up the initial position of the feet based on the current
        contact mode and robot configuration.
        """
        feet_pos = self.pin_robot.get_frames_position_world(self.feet_frame_names)
        
        for i_foot, (foot_cnt, _) in enumerate(zip(self.dyn.feet, feet_pos)):
            # foot_height = foot_pos[2]

            # Contact normal
            if plane_normal is None or plane_origin is None:
                self.data["p"][foot_cnt.plane_point.name] = np.array([0., 0., 0.])
                self.data["p"][foot_cnt.plane_normal.name] = np.array([0., 0., 1.])
            else:
                assert len(plane_normal) == len(plane_origin) == len(self.feet_frame_names),\
                    "plane_normal and plane_origine should be the same length"
                self.data["p"][foot_cnt.plane_point.name] = plane_origin[i_foot]
                self.data["p"][foot_cnt.plane_normal.name] = plane_normal[i_foot]

            # Will be overriden by contact plan
            self.data["p"][foot_cnt.active.name][0] = 1
            self.data["p"][foot_cnt.p_gain.name][0] = self.config_cost.foot_pos_constr_stab[i_foot]


        self.set_parameters_constant(self.data["p"])

    # TODO: Add contact positions
    def setup_contacts(self, i_node: int = 0):
        """
        Setup contact status in the optimization nodes.
        """
        # Offset applied to the current node to setup the optimization window
        node_w = 0
        switch_node = 0
        
        # While in the current optimization window scheme
        while node_w < self.config_opt.n_nodes:

            # Switch
            switch_node += self.gait_planner.next_switch_in(node_w + i_node)
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

                    if n_peak > 0:
                        n_peak += node_w
                        if n_peak < self.config_opt.n_nodes:
                            self.params[foot_cnt.peak.name][:, n_peak] = 1 # Peak constrained (not necessary)

                if n_end > last_node:
                    last_node = n_end

            # Update last node of optimization window updated    
            node_w = last_node


    def print_contact_constraints(self):
        print("\nInitial contacts state")
        for foot_cnt in self.dyn.feet:
            print(self.data["p"][foot_cnt.active.name][0])

        print("\nContacts")
        for foot_cnt in self.dyn.feet:
            print(foot_cnt.frame_name, "contact")
            print(self.params[foot_cnt.active.name])

        print("\nImpacts")
        print(self.params[self.dyn.impact_active.name])

    def init(self,
              i_node : int = 0,
              v_des : np.ndarray = np.zeros(3),
              w_yaw_des : float = 0.,
              ):
        """
        Setup solver depending on the current configuration and the 
        current optimization node.
        """
        q, v = self.pin_robot.get_state()

        self.setup_cost()
        self.setup_reference(q, v_des, w_yaw_des)
        self.setup_initial_state(q, v)
        self.setup_initial_feet_pos(i_node)
        self.setup_contacts(i_node)

        self.update_cost_weights()
        self.update_parameters()

    def solve(self,
              print_stats : bool = False,
              print_time : bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the optimization problem.
        """
        super().solve(print_stats=print_stats, print_time=print_time)
        self.parse_sol()

        q_sol = np.array([
            self._euler_to_quat_state(q_euler) for
            q_euler in self.states[self.dyn.q.name].T
        ])[1:]
        v_sol = (self.states[self.dyn.v.name].T)[1:, :]
        a_sol = (self.inputs[self.dyn.a.name].T)
        forces = np.array(
            [self.inputs[f"f_{foot}_{self.dyn.name}"]
             for foot in self.feet_frame_names]
            ).swapaxes(-1, 1).swapaxes(0, 1)
        
        # First one is the current state (??)
        if self.enable_time_opt:
            dt_node_sol = self.inputs["dt"].flatten()
        else:
            dt_node_sol = np.full((len(q_sol),), self.dt_nodes)

        torques_sol = self.dyn.get_torques(
            self.pin_robot.model,
            self.pin_robot.data,
            self.feet_frame_names,
            q_sol,
            v_sol,
            a_sol,
            forces,
        )
        return q_sol, torques_sol, dt_node_sol
