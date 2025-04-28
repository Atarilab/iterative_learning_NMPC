import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

import argparse
from typing import Tuple, List
import numpy as np
from datetime import datetime
from DAgger.utils.RolloutMPC_shift_phase_percentage import rollout_mpc_phase_percentage_shift
# from Behavior_Cloning.utils.RolloutMPC_force_perturbation import rollout_mpc_phase_percentage_shift

if __name__ == "__main__":
    # rollout_mpc
    early_termination, record_path = rollout_mpc_phase_percentage_shift(show_plot=True,
                                                 sim_time = 10.0,
                                                 current_time = 0.0,
                                                 visualize=True,
                                                 save_data=False,
                                                 record_video = False,
                                                 v_des= [0.15,0.1,0.0])
    print(record_path)
    print(early_termination)