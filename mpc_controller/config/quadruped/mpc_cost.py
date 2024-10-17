import numpy as np
from dataclasses import dataclass
from typing import Any
from ..config_abstract import MPCCostConfig
import numpy as np
from dataclasses import dataclass
from ..config_abstract import MPCCostConfig

HIP_SHOULDER_ELBOW_SCALE = [20., 3., 0.1]
# PENALIZE JOINT MOTION
W_JOINT = 1.e-5

@dataclass
class Go2CyclicCost(MPCCostConfig):
    # Robot name
    robot_name: str = "Go2"

    # Updated base running cost weights
    W_base: np.ndarray = np.array([
        1e1, 1e1, 1e3,      # Base position weights
        1e2, 1e4, 1e4,      # Base orientation (YRP) weights
        1e4, 1e4, 5e3,      # Base linear velocity weights
        1e3, 1e3, 1e3,      # Base angular velocity weights
    ])

    # Updated base terminal cost weights
    W_e_base: np.ndarray = np.array([
        3e3, 3e3, 5e3,     # Base position weights
        5e2, 1e4, 1e4,     # Base orientation (YRP) weights
        1e1, 1e1, 1e1,     # Base linear velocity weights
        1e-1, 1e2, 1e2     # Base angular velocity weights
    ])

    # Joint running cost to nominal position and vel (hip, shoulder, elbow)
    W_joint: np.ndarray = np.array(HIP_SHOULDER_ELBOW_SCALE * 4 + HIP_SHOULDER_ELBOW_SCALE * 4) * W_JOINT * 10

    # Joint terminal cost to nominal position and vel (hip, shoulder, elbow)
    W_e_joint: np.ndarray = np.array(HIP_SHOULDER_ELBOW_SCALE * 4 + [1., 1., 1.] * 4) * W_JOINT

    # Acceleration cost weights for joints (hip, shoulder, elbow)
    W_acc: np.ndarray = np.array(HIP_SHOULDER_ELBOW_SCALE * 4) * W_JOINT
    
    # swing cost weights
    W_swing: np.ndarray = np.array([W_JOINT] * 4)

    # force regularization weights for each foot
    W_cnt_f_reg: np.ndarray = np.array([1e-2, 1e-2, 1e-3] * 4).reshape(4,3)

    # Feet position constraint stability
    foot_pos_constr_stab: np.ndarray = np.array([1.0e3] * 4)

    reg_eps: float = 1.0e-6
    reg_eps_e: float = 1.0e-5

    Kp: float = .8
    Kd: float = .5

class CostConfigFactory():
    AVAILABLE_GAITS = {
        Go2CyclicCost.robot_name.lower(): Go2CyclicCost(),
    }

    @staticmethod
    def get(robot_name: str) -> MPCCostConfig:

        config = CostConfigFactory.AVAILABLE_GAITS.get(robot_name.lower(), None)

        if config is None:
            raise ValueError(f"{robot_name} not available.")
        
        return config
