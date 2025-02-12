import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../..')

import argparse
from typing import Tuple, List
import numpy as np
from datetime import datetime
from iterative_supervised_learning.utils.RolloutMPC import rollout_mpc

# Example usage
if __name__ == "__main__":
    try:
        record_dir = "./data/"  # Define record directory
        if not os.path.exists(record_dir):
            os.makedirs(record_dir)

        record_dir, state_history, base_history, vc_goal_history, cc_goal_history, action_history = rollout_mpc(
            mode="close_loop",
            sim_time=3.0,
            robot_name="go2",
            record_dir=record_dir,
            v_des=[0.3, 0.0, 0.0],
            save_data=True,
            interactive=False,
            record_video=False,
            visualize=True,
            randomize_initial_state=False,
            show_plot=False
        )

        print(f"Recorded data path: {record_dir}")

    except Exception as e:
        print(f"Error in rollout_mpc: {e}")

    # print(state_history[:500,0])
    # print(state_history[500:1000,0])
    for i in range(5):
        print(state_history[i,:])
    print("shape of state space = ", np.shape(state_history))