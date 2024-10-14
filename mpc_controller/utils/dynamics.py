from typing import List
import pinocchio as pin
import numpy as np
from contact_tamp.traj_opt_acados.models.floating_base_dynamics import FloatingBaseDynamics
from contact_tamp.traj_opt_acados.models.point_contact import PointContact
from contact_tamp.traj_opt_acados.utils.model_utils import loadSymModel
from contact_tamp.traj_opt_acados.interface.acados_helper import ProblemFormulation, cs

class QuadrupedDynamics(FloatingBaseDynamics):
    MU_CONTACT = 0.7

    def __init__(self,
                 robot_name: str,
                 model_path: str,
                 feet_frame_names: List[str]):
        
        model, data = loadSymModel(model_path)
        self.feet_frame_names = feet_frame_names
        super().__init__(robot_name, model, data)

        # note: 0.022 is the foot radius
        self.feet = [PointContact(dyn=self, frame=frame_name, mu=QuadrupedDynamics.MU_CONTACT) for frame_name in feet_frame_names]

        self.add_contacts(self.feet)
        self.base_cost = self.add_expr(name="base_cost", expr=self.get_base_cost())
        self.acc_cost = self.add_expr(name="acc_cost", expr=self.get_acc_cost())
        self.swing_cost = self.add_expr(name="sw_cost", expr=self.get_swing_foot_cost())

    def setup(self, problem: ProblemFormulation):
        for f in self.feet:
            f.setup(problem)
        super().setup(problem)
        problem.add_cost(self.base_cost)
        problem.add_cost(self.acc_cost)
        problem.add_cost(self.swing_cost)
        problem.add_cost_terminal(self.swing_cost)
        problem.add_cost_terminal(self.base_cost)

    def get_hg(self):
        return self.h

    def get_base_cost(self):
        r = self.q[:3]  # position cost
        euler = self.q[3:6]
        return cs.vcat([r, euler, self.v[:6]])

    def get_joint_cost(self):
        return cs.vcat([self.q[6:], self.v[6:]])

    def get_acc_cost(self):
        return self.a[6:]

    def get_swing_foot_cost(self):
        z = cs.vcat([c.peak * c.get_position()[2] for c in self.feet])
        return z
    
    @staticmethod
    def get_torques(model : pin.Model,
                    data : pin.Data,
                    foot_frame_name : List[str],
                    q_plan : np.ndarray,
                    v_plan : np.ndarray,
                    acc_plan : np.ndarray,
                    forces : np.ndarray,
                    ) -> np.ndarray:
        """
        Return torque plan.

        Args:
            q_plan (np.ndarray): State position plan
            v_plan (np.ndarray): State velocity plan
            acc_plan (np.ndarray): Acceleration plan
            forces (np.ndarray): Contact forces plan
        
        Return:
            torques: Torques plan.
        """
        torques = np.zeros((forces.shape[0], model.nv))

        for i, (q, v, acc, f_ext) in enumerate(
            zip(q_plan, v_plan, acc_plan, forces)
        ):
            # Set the current state
            pin.forwardKinematics(model, data, q)
            pin.computeJointJacobians(model, data, q)
            pin.updateFramePlacements(model, data)

            # Compute the mass matrix
            M = pin.crba(model, data, q)
            
            # Compute the Coriolis and gravity terms
            b = pin.nle(model, data, q, v)
            
            # Initialize the contact forces vector
            contact_forces = np.zeros((model.nv,))

            # Loop through each end-effector and accumulate external forces
            for ee_name, f_ee in zip(foot_frame_name, f_ext):
                frame_id = model.getFrameId(ee_name)
                J_ee = pin.computeFrameJacobian(model, data, q, frame_id, pin.ReferenceFrame.LOCAL)
                contact_forces += J_ee.T @ np.hstack((f_ee, np.zeros(3)))
            
            # Compute the torques using the inverse dynamics equation
            tau = M @ acc - b - contact_forces
            torques[i, :] = tau
            
        return torques