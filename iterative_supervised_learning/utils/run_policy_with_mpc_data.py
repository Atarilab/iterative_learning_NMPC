import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../..')
import numpy as np
import torch
from iterative_supervised_learning.utils.network import GoalConditionedPolicyNet
from iterative_supervised_learning.utils.database import Database

# define global variables
SIM_DT = 1.0e-3
VIEWER_DT = 1/30.
# with base_wrt_feet
n_state = 44 # state:44 + vc_goal:3
n_state += 3
print("n_state = ",n_state)
n_action = 12

kp = 20.0
kd = 1.5

episode_length = 2000
v_des = [0.3,0.0,0.0]
action_policy_his = []

# read data from mpc file
data_path = "/home/atari/workspace/iterative_supervised_learning/examples/data/kp20_kd1.5.npz"
data = np.load(data_path)
state_his = data["state"]

# initialize policy network
policy_path = "/home/atari/workspace/iterative_supervised_learning/examples/data/behavior_cloning/trot/Mar_07_2025_15_50_55/network/policy_100.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = GoalConditionedPolicyNet(input_size=n_state, output_size=n_action, num_hidden_layer=3,
                                            hidden_dim=512, batch_norm=True)
policy_net.load_state_dict(torch.load(policy_path, map_location=device)['network'])
policy_net.to(device)
policy_net.eval()
print(policy_net)

# initialize input normalization parameters
norm_policy_input = True
mean_std = None
database_path = "/home/atari/workspace/iterative_supervised_learning/examples/data/behavior_cloning/trot/Mar_07_2025_15_50_55/dataset/database_0.hdf5"
if norm_policy_input and database_path:
    db = Database(limit=10000000, norm_input=True)
    db.load_saved_database(database_path)
    # db.calc_input_mean_std()
    mean_std = db.get_database_mean_std()
print(mean_std)

# main loop
for i in range(episode_length):
    # form normalized policy input
    state = state_his[i,:]
    print("unnormalized state is = ")
    print(state)
    print()
    if norm_policy_input and mean_std is not None:
        state_mean, state_std = mean_std[0],mean_std[1]
        # print("state_mean = ", state_mean)
        # print("state_std = ",state_std)
        # input()
        state[1:] = (state[1:] - state_mean[1:]) / state_std[1:]
    print("normalized state is = ")
    print(state)
    print()
    x = np.concatenate([np.array(state), np.array(v_des)])[:n_state]
    print("current policy input is = ", x)
    print("shape of the policy input is = ", np.shape(x))
    x_tensor = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
    
    # get policy output
    y_tensor = policy_net(x_tensor)
    action_policy = y_tensor.detach().cpu().numpy().reshape(-1)
    action_policy_his.append(action_policy)
    print("current policy output is = ")
    print(action_policy)
    # input()

# save action_policy_his into a file
action_policy_his = np.array(action_policy_his)
save_path = "./data/action_policy_history.npz"
np.savez(save_path, action_policy_his=action_policy_his)