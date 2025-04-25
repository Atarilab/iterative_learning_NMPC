import numpy as np

data_path = "/home/atari/workspace/iterative_supervised_learning/examples/data/kp20_kd1.5.npz"
data = np.load(data_path)
feet_pos_w = data["feet_pos_w"]
time = data["time"]

for i in range(len(time)):
    print("current time is = ",time[i])
    print("current feet position is = ")
    print(feet_pos_w[i])
    input()