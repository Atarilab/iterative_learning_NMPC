import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple
from ..config_abstract import MPCOptConfig
from contact_tamp.traj_opt_acados.interface.acados_helper import HPIPM_MODE

@dataclass
class MPCQuadrupedCyclic(MPCOptConfig):
    # MPC Config
    time_horizon : float = 1.
    n_nodes : int = 50
    opt_dt_scale : Tuple[float, float] = (0.5, 1.75)
    replanning_freq : int = 20
    interpolation_mode : str = "quadratic"
    Kp : float = 25
    Kd : float = 3
    use_delay : bool = True
    # Solver config
    max_iter : int = 1
    max_qp_iter: int = 6
    nlp_tol: float = 1e-1
    qp_tol: float = 1e-2
    recompile: bool = True
    use_cython: bool = False
    hpipm_mode: HPIPM_MODE = HPIPM_MODE.speed
    enable_time_opt : bool = False
    enable_impact_dyn : bool = False
    real_time_it : bool = False
    cnt_patch_restriction : bool = True
    opt_peak : bool = True
    warm_start_sol : bool = True
    warm_start_nlp : bool = True
    warm_start_qp : bool = True