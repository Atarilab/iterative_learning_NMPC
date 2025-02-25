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
    early_termination, record_path = rollout_mpc(show_plot=False)
    print(record_path)