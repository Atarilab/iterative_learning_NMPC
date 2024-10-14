import numpy as np
from dataclasses import dataclass
from typing import Any
from ..config_abstract import MPCCostConfig
import numpy as np
from dataclasses import dataclass
from ..config_abstract import MPCCostConfig

@dataclass
class Go2CyclicCost(MPCCostConfig):
    # Robot name
    robot_name: str = "Go2"

    # Updated base running cost weights
    W_base: np.ndarray = np.array([
        1e1, 1e1, 1e1,        # Base position weights
        1e-2, 1e-2, 1e-2,     # Base orientation weights
        1e1, 1e1, 1e1,        # Base linear velocity weights
        1e1, 1e1, 1e1         # Base angular velocity weights
    ])

    # Updated base terminal cost weights
    W_e_base: np.ndarray = np.array([
        1e-1, 1e-1, 1e-1,     # Base position weights
        1e-2, 1e-2, 1e-2,     # Base orientation weights
        1e1, 1e1, 1e1,        # Base linear velocity weights
        1e1, 1e1, 1e1         # Base angular velocity weights
    ])

    # Acceleration cost weights (unchanged)
    W_acc: np.ndarray = np.array([
        1e-3, 1e-3, 1e-3,
        1e-3, 1e-3, 1e-3, 
        1e-3, 1e-3, 1e-3,
        1e-3, 1e-3, 1e-3
    ])
    
    # Updated swing cost weights
    W_swing: np.ndarray = np.array([1e-1] * 4)

    # Updated force regularization weights for each foot
    W_cnt_f_reg: np.ndarray = np.array([
        [1e-2, 1e-2, 1e-3],
        [1e-2, 1e-2, 1e-3],
        [1e-2, 1e-2, 1e-3],
        [1e-2, 1e-2, 1e-3],
    ])

    # Feet position constraint stability
    foot_pos_constr_stab: np.ndarray = np.array([1.0e3] * 4)

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
