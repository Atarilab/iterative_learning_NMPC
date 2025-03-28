import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../..')

import numpy as np
from iterative_supervised_learning.utils.RolloutPolicy import rollout_policy

policy_path = "/home/atari/workspace/iterative_supervised_learning/examples/data/behavior_cloning/trot/Mar_28_2025_15_09_14/network/policy_final.pth"
data_MPC_path = "/home/atari/workspace/iterative_supervised_learning/examples/data/behavior_cloning/trot/Mar_28_2025_15_09_14/dataset/experiment/traj_nominal_03_28_2025_15_09_18.npz"
v_des = [0.15,0.0,0.0]

# extract initial states from start time
data_MPC = np.load(data_MPC_path)
# start_time = 0.1
start_time = 0.0
q_MPC = data_MPC["q"]
v_MPC = data_MPC["v"]

q0 = q_MPC[int(start_time * 1000)]
v0 = v_MPC[int(start_time * 1000)]
print("current q0 from MPC recording is = ",q0)
print("current v0 from MPC recording is = ",v0)
initial_state = [q0,v0]

# rollout policy
rollout_policy(policy_path, 
                sim_time=5.0, 
                v_des = v_des, 
                record_video=False,
                norm_policy_input=True,
                save_data=True,
                initial_state = initial_state,
                start_time = start_time,
                data_MPC_path=data_MPC_path)
