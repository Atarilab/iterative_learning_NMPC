import numpy as np
from dataclasses import dataclass, field
from typing import List
from ..config_abstract import MPCOptConfig

@dataclass
class MPCQuadrupedCyclic(MPCOptConfig):
    time_horizon : float = 0.8
    n_nodes : int = 50
    opt_dt_scale : np.ndarray = field(default_factory=lambda: np.array([0.75, 1.25]))
    replanning_freq : int = 50
    max_iter : int = 2
    interpolation_mode : str = "linear"
    enable_time_opt : bool = False
    real_time_it : bool = False
    opt_cnt_pos : bool = False
    opt_peak : bool = True
    warm_start_sol : bool = True
    warm_start_nlp : bool = True
    warm_start_qp : bool = True