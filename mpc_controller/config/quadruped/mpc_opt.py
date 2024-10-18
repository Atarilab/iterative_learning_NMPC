import numpy as np
from dataclasses import dataclass, field
from typing import List
from ..config_abstract import MPCOptConfig
from contact_tamp.traj_opt_acados.interface.acados_helper import HPIPM_MODE

@dataclass
class MPCQuadrupedCyclic(MPCOptConfig):
    time_horizon : float = 0.8
    n_nodes : int = 50
    opt_dt_scale : np.ndarray = field(default_factory=lambda: np.array([0.75, 1.25]))
    replanning_freq : int = 50
    max_iter : int = 5
    qp_iter: int = 10
    nlp_tol: float = 1e-1 # outer loop SQP tolerance
    qp_tol: float = 5e-3 # inner loop interior point method tolerance
    recompile: bool = False
    use_cython: bool = False
    hpipm_mode: HPIPM_MODE = HPIPM_MODE.speed
    interpolation_mode : str = "linear"
    enable_time_opt : bool = False
    real_time_it : bool = False
    opt_cnt_pos : bool = False
    opt_peak : bool = True
    warm_start_sol : bool = True
    warm_start_nlp : bool = True
    warm_start_qp : bool = True