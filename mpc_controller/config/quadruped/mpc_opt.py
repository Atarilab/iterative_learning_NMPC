import numpy as np
from dataclasses import dataclass, field
from typing import List
from ..config_abstract import MPCOptConfig

@dataclass
class MPCQuadrupedCyclic(MPCOptConfig):
    time_horizon : float = .5
    n_nodes : int = 40
    opt_dt_scale : np.ndarray = field(default_factory=lambda: np.array([0.75, 1.25]))
    replanning_freq : int = 10
    max_iter : int = 5
    interpolation_mode : str = "quadratic"
    enable_time_opt : bool = False
    real_time_it : bool = False
    opt_cnt_pos : bool = False
    opt_peak : bool = True
    warm_start_sol : bool = False
    warm_start_nlp : bool = True
    warm_start_qp : bool = True