import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple
from ..config_abstract import MPCOptConfig
from contact_tamp.traj_opt_acados.interface.acados_helper import HPIPM_MODE

@dataclass
class MPCQuadrupedCyclic(MPCOptConfig):
    ### MPC Config
    # Time horizon (s)
    time_horizon : float = 1.
    # Number of optimization nodes
    n_nodes : int = 25
    # Replanning frequency
    replanning_freq : int = 30
    # gain on joint position for torque PD
    Kp : float = 20.
    # Kp : float = 100.
    # gain on joint velocities for torque PD
    Kd : float = 1.5
    # Kd : float = 2.0
    ### Solver Config
    # Recompile solver
    # recompile: bool = True
    recompile: bool = False
    # Solver maximum SQP iterations
    max_iter : int = 1
    # Maximum qp iteration for one SQP step 
    max_qp_iter: int = 6