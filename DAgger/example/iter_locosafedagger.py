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

class SafeDAggerPipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.policy_path = cfg.initial_policy_path
        self.aggregated_dataset = cfg.pretrain_dataset_path
        self.base_run_dir = cfg.run_dir

    def run_data_collection(self, iter_index):
        self.cfg.iter_index = iter_index
        self.cfg.policy_path = self.policy_path
        self.cfg.reference_mpc_path = self.cfg.reference_mpc_path  # no-op for clarity

        dc = DataCollection(self.cfg)
        dc.run()

        return os.path.join(self.cfg.data_save_path, "agg_dataset.hdf5")

    def run_training(self):
        # No modification of cfg — use locals only
        training_cfg = OmegaConf.merge(self.cfg, OmegaConf.create({
            "database_path": self.aggregated_dataset,
            "pretrained_policy_path": self.policy_path,
        }))
        bc = BehavioralCloning(training_cfg)
        bc.run()

        trained_policy_path = os.path.join(os.path.dirname(self.aggregated_dataset), "../network/policy_final.pth")
        return trained_policy_path

    def run(self):
        for i in range(self.cfg.n_iteration):
            print(f"\n🔁 Starting SafeDAgger Iteration {i}")

            new_dataset_path = self.run_data_collection(i)
            self.aggregated_dataset = new_dataset_path

            self.policy_path = self.run_training()
            print(f"✅ Policy saved: {self.policy_path}")


@hydra.main(config_path="../cfgs", config_name="iter_locosafedagger.yaml")
def main(cfg):
    pipeline = SafeDAggerPipeline(cfg)
    pipeline.run()

if __name__ == "__main__":
    main()
