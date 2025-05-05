# TODO this script is to define a safe zone for policy-controlled robot and let MPC takes over

# a working policy
policy_path = "/home/atari/workspace/Behavior_Cloning/examples/data/behavior_cloning/trot/Apr_16_2025_13_02_09/network/policy_400.pth"
data_MPC_path = "/home/atari/workspace/Behavior_Cloning/examples/data/behavior_cloning/trot/Apr_16_2025_13_02_09/dataset/experiment/traj_nominal_04_16_2025_13_02_15.npz"
v_des = [0.15,0.0,0.0]

