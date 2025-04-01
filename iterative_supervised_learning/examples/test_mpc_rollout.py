import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../..')

import argparse
from typing import Tuple, List
import numpy as np
from datetime import datetime
from iterative_supervised_learning.utils.RolloutMPC_shift_phase_percentage import rollout_mpc_phase_percentage_shift

if __name__ == "__main__":
    # rollout_mpc
    early_termination, record_path = rollout_mpc_phase_percentage_shift(show_plot=True,
                                                 sim_time = 10.0,
                                                 current_time = 0.0,
                                                 visualize=True,
                                                 save_data=True,
                                                 record_video = False,
                                                 v_des= [0.30,0,0])
    print(record_path)
    print(early_termination)