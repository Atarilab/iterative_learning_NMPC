import numpy as np

# MPC path
data_path = "/home/atari/workspace/iterative_supervised_learning/examples/data/simulation_data_03_18_2025_09_54_23.npz"
data_MPC = np.load(data_path)

# extract q,v,ctrl and action
q = data_MPC["q"][:,7:]
v = data_MPC["v"][:,6:]

tau_frflrrrl = data_MPC["ctrl"]
FR_torque = tau_frflrrrl[:,0:3]
FL_torque = tau_frflrrrl[:,3:6]
RR_torque = tau_frflrrrl[:,6:9]
RL_torque = tau_frflrrrl[:,9:]
tau_flfrrlrr = np.concatenate([FL_torque,FR_torque,RL_torque,RR_torque],axis=1)

action = data_MPC["action"]

print("Shape of q:", q.shape)
print("Shape of v:", v.shape)
print("Shape of tau_flfrrlrr:", tau_flfrrlrr.shape)
print("Shape of action:", action.shape)


# print out first k data points
k = 5
for i in range(k):
    print("joint q is  = ")
    print(q[i,:])
    print("joint v is = ")
    print(v[i,:])
    print("applied torque is = ")
    print(tau_flfrrlrr[i,:])
    print("calculated PD target is = ")
    print(action[i,:])
    print()