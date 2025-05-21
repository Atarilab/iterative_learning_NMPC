import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../..')

import numpy as np
from Behavior_Cloning.utils.RolloutPolicy import rollout_policy
import random
import torch

seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

# this works ok, config: no phase percentage, improved nullspace perturbation, 150000 data points
# policy_path = "/home/atari/workspace/Behavior_Cloning/examples/data/behavior_cloning/trot/Apr_01_2025_16_00_31/network/policy_400.pth"
# data_MPC_path = "/home/atari/workspace/Behavior_Cloning/examples/data/behavior_cloning/trot/Apr_01_2025_16_00_31/dataset/experiment/traj_nominal_04_01_2025_16_00_34.npz"
# v_des = [0.15,0.0,0.0]

# survive for 8 seconds
# policy_path = "/home/atari/workspace/Behavior_Cloning/examples/data/behavior_cloning/trot/Apr_01_2025_17_30_52/network/policy_450.pth"
# data_MPC_path = "/home/atari/workspace/Behavior_Cloning/examples/data/behavior_cloning/trot/Apr_01_2025_17_30_52/dataset/experiment/traj_nominal_04_01_2025_17_30_58.npz"
# v_des = [0.15,0.0,0.0]

# much smaller dataset can also train a working policy that survives 3s
# policy_path = "/home/atari/workspace/Behavior_Cloning/examples/data/behavior_cloning/trot/Apr_02_2025_13_34_41/network/policy_450.pth"
# data_MPC_path = "/home/atari/workspace/Behavior_Cloning/examples/data/behavior_cloning/trot/Apr_02_2025_13_34_41/dataset/experiment/traj_nominal_04_02_2025_13_34_50.npz"
# v_des = [0.15,0.0,0.0]

# force perturbation(best for now, start from 1.0s)
# policy_path = "/home/atari/workspace/Behavior_Cloning/examples/data/behavior_cloning/trot/Apr_16_2025_13_02_09/network/policy_400.pth"
# data_MPC_path = "/home/atari/workspace/Behavior_Cloning/examples/data/behavior_cloning/trot/Apr_16_2025_13_02_09/dataset/experiment/traj_nominal_04_16_2025_13_02_15.npz"
# v_des = [0.15,0.0,0.0]

#######################################################################################################################################################################################
# batch size = 256
policy_path = "/home/atari/workspace/Behavior_Cloning/utils/data/behavior_cloning/trot/May_21_2025_09_18_32/network/policy_150.pth"
data_MPC_path = "/home/atari/workspace/Behavior_Cloning/utils/data/behavior_cloning/trot/May_21_2025_09_18_32/dataset/experiment/traj_nominal_05_21_2025_09_18_34.npz"
v_des = [0.15,0.0,0.0]

# no batch normalization
# policy_path = "/home/atari/workspace/Behavior_Cloning/utils/data/to_test/network/policy_final.pth"
# data_MPC_path = "/home/atari/workspace/Behavior_Cloning/utils/data/behavior_cloning/trot/May_21_2025_09_18_32/dataset/experiment/traj_nominal_05_21_2025_09_18_34.npz"
# v_des = [0.15,0.0,0.0]

# dropout
# policy_path = "/home/atari/workspace/Behavior_Cloning/utils/data/to_test/test2/network/policy_50.pth"
# data_MPC_path = "/home/atari/workspace/Behavior_Cloning/utils/data/behavior_cloning/trot/May_21_2025_09_18_32/dataset/experiment/traj_nominal_05_21_2025_09_18_34.npz"
# v_des = [0.15,0.0,0.0]

# with phase percentage
# policy_path = "/home/atari/workspace/Behavior_Cloning/utils/data/behavior_cloning/trot/May_21_2025_14_23_08/network/policy_350.pth"
# data_MPC_path = "/home/atari/workspace/Behavior_Cloning/utils/data/behavior_cloning/trot/May_21_2025_14_23_08/dataset/experiment/traj_nominal_05_21_2025_14_23_10.npz"
# v_des = [0.15,0.0,0.0]

#######################################################################################################################################################################################
# extract initial states from start time
data_MPC = np.load(data_MPC_path)
start_time = 1.0
# start_time = 0.0
q_MPC = data_MPC["q"]
v_MPC = data_MPC["v"]

q0 = q_MPC[int(start_time * 1000)]
v0 = v_MPC[int(start_time * 1000)]
print("current q0 from MPC recording is = ",q0)
print("current v0 from MPC recording is = ",v0)
initial_state = [q0,v0]
input()

# rollout policy
rollout_policy(policy_path, 
                sim_time=60.0, 
                v_des = v_des, 
                record_video=False,
                norm_policy_input=True,
                visualize=True,
                save_data=False,
                initial_state = initial_state,
                start_time = start_time,
                data_MPC_path=data_MPC_path,
                interactive=True,)
