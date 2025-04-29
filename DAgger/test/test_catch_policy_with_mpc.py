# This script is to test catching policy with MPC/ aggregate dataset.

import sys
import os
import time
import numpy as np
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

from DAgger.utils.Rollout_combined_controller import rollout_combined_controller
from DAgger.utils.dataset_utils import merge_datasets  # assume you have something similar or I can help make it

def main():
    # ========== Parameters ==========
    robot_name = "go2"
    control_mode = "policy"  # start with policy, MPC catches failures
    policy_path = "/home/atari/workspace/Behavior_Cloning/examples/data/behavior_cloning/trot/Apr_16_2025_13_02_09/network/policy_400.pth"
    reference_mpc_path = "/home/atari/workspace/Behavior_Cloning/examples/data/behavior_cloning/trot/Apr_16_2025_13_02_09/dataset/experiment/traj_nominal_04_16_2025_13_02_15.npz"
    record_dir = "./data/safedagger/"
    sim_time = 20.0
    start_time = 0.0
    v_des = np.array([0.15, 0.0, 0.0])
    save_data = True
    visualize = True

    # Create directory
    os.makedirs(record_dir, exist_ok=True)

    # ========== Step 1: Rollout combined controller ==========
    print("=== Rolling out combined controller (Policy + MPC fallback) ===")
    rollout_combined_controller(
        control_mode=control_mode,
        robot_name=robot_name,
        sim_time=sim_time,
        start_time=start_time,
        v_des=v_des,
        record_video=False,
        visualize=visualize,
        save_data=save_data,
        record_dir=record_dir,
        interactive=False,
        policy_path=policy_path,
        reference_mpc_path=reference_mpc_path,
        nominal_flag=False,  # Not nominal: mixed expert/policy
    )

    # ========== Step 2: Pretraining dataset ==========
    print("\n=== Preparing pretraining dataset (Dataset 0) ===")
    pretrain_dataset_path = os.path.join(record_dir, "pretrain_dataset0.npz")

    # For the first iteration, you can use purely policy data, or mix with nominal
    # Here assuming we already have nominal mpc rollouts collected elsewhere

    # NOTE: If you want to copy nominal rollouts into Dataset 0, you can load and re-save them here.
    # Otherwise, skip.

    # ========== Step 3: Aggregated dataset ==========
    print("\n=== Aggregating collected data (Dataset 1) ===")
    all_traj_files = [
        os.path.join(record_dir, f) for f in os.listdir(record_dir) if f.endswith(".npz")
    ]

    dataset_agg_path = os.path.join(record_dir, "agg_dataset1.npz")

    print(f"Found {len(all_traj_files)} rollouts to aggregate.")

    merge_datasets(all_traj_files, dataset_agg_path)

    print(f"Aggregated dataset saved to {dataset_agg_path}")

    # ========== Step 4: Ready for retraining ==========
    print("\n🎯 Ready for retraining with aggregated dataset!")

if __name__ == "__main__":
    main()
