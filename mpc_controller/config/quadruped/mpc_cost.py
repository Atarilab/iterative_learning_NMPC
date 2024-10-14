import numpy as np
from dataclasses import dataclass
from typing import Any
from ..config_abstract import MPCCostConfig

@dataclass
class Go2CyclicCost(MPCCostConfig):
    # Robot name
    robot_name: str = "Go2"

    W_e_base: np.ndarray = np.array([
        1.0e2, 1.0e2, 1.0e2,    # Base position weights
        1.0e2, 1.0e2, 1.0e2,    # Base orientation weights
        1.0e2, 1.0e2, 1.0e2,    # Base linear velocity weights
        1.0e1, 1.0e1, 1.0e1     # Base angular velocity weights
    ])
    
    W_base: np.ndarray = np.array([
        1.0e0, 1.0e0, 1.0e0,    # Base position weights
        1.0e2, 1.0e2, 1.0e2,    # Base orientation weights
        1.0e2, 1.0e2, 1.0e2,    # Base linear velocity weights
        1.0e1, 1.0e1, 1.0e1     # Base angular velocity weights
    ])
    
    W_acc: np.ndarray = np.array([  # Acceleration weights
        1.0e-3, 1.0e-3, 1.0e-3,
        1.0e-3, 1.0e-3, 1.0e-3,
        1.0e-3, 1.0e-3, 1.0e-3,
        1.0e-3, 1.0e-3, 1.0e-3
    ])
    
    W_swing: np.ndarray = np.array([1.0e3] * 4)

    W_cnt_f_reg: np.ndarray = np.array([
        [1.0e-2, 1.0e-2, 1.0e-3],
        [1.0e-2, 1.0e-2, 1.0e-3],
        [1.0e-2, 1.0e-2, 1.0e-3],
        [1.0e-2, 1.0e-2, 1.0e-3],
    ])

    W_feet_z_vel: np.ndarray = np.array([1.0e3] * 4)

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
