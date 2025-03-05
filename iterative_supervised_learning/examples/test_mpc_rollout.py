import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../..')

import argparse
from typing import Tuple, List
import numpy as np
from datetime import datetime
from iterative_supervised_learning.utils.RolloutMPC_rewrite import rollout_mpc

if __name__ == "__main__":
    # rollout_mpc
    early_termination, record_path = rollout_mpc(show_plot=False,
                                                 sim_time = 10.0,
                                                 current_time = 0.0,
                                                 visualize=True,
                                                 save_data=True)
    print(record_path)
    print(early_termination)