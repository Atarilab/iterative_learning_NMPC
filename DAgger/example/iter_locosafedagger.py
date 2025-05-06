import os
import hydra
from omegaconf import OmegaConf
from DAgger.utils.data_collection_locosafedagger import DataCollection 
from DAgger.utils.train_locosafedagger import BehavioralCloning
import random
import numpy as np
import torch

# Set random seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def run_data_collection(cfg, policy_path, reference_mpc_path, output_dir):
    cfg.policy_path = policy_path
    cfg.reference_mpc_path = reference_mpc_path
    cfg.data_save_path = output_dir
    dc = DataCollection(cfg)
    dc.run()
    return os.path.join(dc.data_save_path, "dataset/database_0.hdf5")

def run_training(cfg, dataset_path, pretrained_path=None, output_policy_dir=None):
    cfg.database_path = dataset_path
    cfg.pretrained_policy_path = pretrained_path or ""
    bc = BehavioralCloning(cfg)
    bc.run()
    return os.path.join(os.path.dirname(dataset_path), "../network/policy_final.pth")

def run_pipeline(cfg, n_iterations):
    # Start from pretraining dataset
    aggregated_dataset = cfg.pretrain_dataset_path

    for i in range(n_iterations):
        print(f"\n🔁 Starting SafeDAgger Iteration {i}")

        # === Collect new expert data ===
        current_policy = cfg.initial_policy_path if i == 0 else policy_path
        print(f"Current policy path: {current_policy}")
        input()
        
        output_dir = os.path.join(cfg.output_root, f"iter_{i}")
        print(f"Output directory: {output_dir}")
        input()
        
        new_data_path = run_data_collection(cfg, current_policy, cfg.reference_mpc_path, output_dir)

        # === Aggregate ===
        agg_output_path = os.path.join(output_dir, f"agg_dataset_{i+1}.hdf5")
        dc = DataCollection(cfg)
        dc.append_to_dataset(base_dataset_path=aggregated_dataset, output_path=agg_output_path)
        aggregated_dataset = agg_output_path  # use for next iteration

        # === Train new policy ===
        policy_path = run_training(cfg, dataset_path=aggregated_dataset, pretrained_path=current_policy)
        print(f"✅ Policy saved: {policy_path}")

@hydra.main(config_path="../cfgs", config_name="iter_locosafedagger.yaml")
def main(cfg):
    run_pipeline(cfg, cfg.n_iteration)

if __name__ == "__main__":
    main()
