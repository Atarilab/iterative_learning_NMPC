import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../..')

import numpy as np
from iterative_supervised_learning.utils.network import GoalConditionedPolicyNet
from iterative_supervised_learning.utils.database import Database

n_state = 44 # with base_wrt_feet
# n_state = 36 # without base_wrt_feet
n_state += 3
n_action = 12

if __name__ == "__main__":
    # initialize some path
    policy_path = "/home/atari/workspace/iterative_supervised_learning/examples/data/behavior_cloning/trot/Mar_04_2025_15_40_12/network/policy_final.pth"
    data_path = "/home/atari/workspace/iterative_supervised_learning/examples/data/simulation_data_03_04_2025_15_59_50.npz"
    database_path = "/home/atari/workspace/iterative_supervised_learning/examples/data/behavior_cloning/trot/Mar_04_2025_15_40_12/dataset/database_0.hdf5"
    norm_policy_input = True
    v_des = [0.3,0.0,0.0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # initialize policy network
    policy_net = GoalConditionedPolicyNet(input_size=n_state, output_size=n_action, num_hidden_layer=3,
                                                   hidden_dim=512, batch_norm=True)
    policy_net.load_state_dict(torch.load(policy_path, map_location=device)['network'])
    policy_net.to(device)
    policy_net.eval()
    print(policy_net)
    
    # initialize database
    db = Database(limit=10000000, norm_input=True)
    db.load_saved_database(database_path)
    mean_std = db.get_database_mean_std()
    state_mean = mean_std[0]
    state_std = mean_std[1]
    print("state_mean = ", state_mean)
    print("state_std = ",state_std)
    input()
    
    # load state from saved file
    data = np.load(data_path)
    state = data["state"]
    # first_state = state[0]
    action = data["action"]
    # first_action = action[0]
    ctrl = data["ctrl"]
    # first_ctrl = ctrl[0]
    
    
    # iterate over state
    for i in range(50,75):
        current_state = state[i]
        current_action = action[i]
        current_ctrl = ctrl[i]
        print("current state is = ",current_state)
        
        if norm_policy_input:
            # normalize state
            current_state[1:] = (current_state[1:] - state_mean[1:]) / state_std[1:]
            print("normalized state = ", current_state)
    
        # get policy input
        x = np.concatenate([np.array(current_state), np.array(v_des)])[:n_state]
        x_tensor = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
        print("shape of policy input is = ",np.shape(x))
        print("policy input = ", x)
    
        # get policy output
        y_tensor = policy_net(x_tensor)
        y = y_tensor.cpu().detach().numpy().reshape(-1)
        print("PD target should be = ",current_action)
        print("inferenced PD target = ", y)
        
        # print("current torque should be  = ", current_ctrl)

        input()