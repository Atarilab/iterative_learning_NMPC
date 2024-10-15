import numpy as np
from dataclasses import dataclass
from typing import List
from ..config_abstract import MPCOptConfig

@dataclass
class MPCQuadrupedCyclic(MPCOptConfig):
    time_horizon : float = 0.2
    n_nodes : int = 20
    opt_dt_scale : np.ndarray = np.array([0.5, 2.5])
    replanning_freq : int = 10
    max_iter : int = 50
    interpolation_mode : str = "linear"
    enable_time_opt : bool = False
    real_time_it : bool = False
    opt_cnt_pos : bool = False
    opt_peak : bool = True
    warm_start_sol : bool = False
    warm_start_nlp : bool = True
    warm_start_qp : bool = True