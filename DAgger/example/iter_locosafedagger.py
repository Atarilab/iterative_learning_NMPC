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
        
        # initialize iterative parameters
        self.policy_path = None
        self.aggregated_dataset = None
        self.iter_index = 0
        
        # setup the run directory
        self.base_run_dir = cfg.run_dir
        
        # import some fixed parameters
        self.reference_mpc_path = cfg.reference_mpc_path

    def run_data_collection(self, iter_index, goal_index, previous_dataset_path, current_policy_path, goal  = [0.0,0.0,0.0]):
        dc = DataCollection(self.cfg,
                            iter_index,
                            goal_index,
                            previous_dataset_path,
                            current_policy_path,
                            goal = goal)
        dc.run()

        return os.path.join(dc.data_save_path, "agg_dataset.hdf5")

    def run_training(self,
                     previous_policy_path=None,
                     current_dataset_path=None):
        
        bc = BehavioralCloning(self.cfg,
                               previous_policy_path, 
                               current_dataset_path)
        bc.run()

        trained_policy_path = os.path.join(os.path.dirname(self.aggregated_dataset), "../network/policy_final.pth")
        return trained_policy_path

    def run(self):
        goal_list = [[0.15, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [-0.15, 0.0, 0.0]]

        for goal_index, goal in enumerate(goal_list):
            print("\n" + "=" * 50)
            print(f"🚩 Starting training for Goal {goal_index}: {goal}")
            print("=" * 50)

            for iter_index in range(self.cfg.n_iteration):
                print(f"\n🔁 Iteration {iter_index} for Goal {goal_index}")
                
                # === Initialize only once for first goal & iteration ===
                if goal_index == 0 and iter_index == 0:
                    self.policy_path = self.cfg.initial_policy_path
                    self.aggregated_dataset = self.cfg.pretrain_dataset_path
                
                # === Logging current setup ===
                print(f"Current policy: {self.policy_path}")
                print(f"Previous dataset: {self.aggregated_dataset}")
                print(f"Goal: {goal}")
                print(f"Save Path: goal_{goal_index}/iter_{iter_index}")
                input()

                # === Data collection ===
                print("Starting data collection...")
                new_dataset_path = self.run_data_collection(
                    iter_index=iter_index,
                    goal_index=goal_index,
                    previous_dataset_path=self.aggregated_dataset,
                    current_policy_path=self.policy_path,
                    goal=goal
                )
                self.aggregated_dataset = new_dataset_path  # Update for training
                print(f"✅ Aggregated dataset saved: {self.aggregated_dataset}")
                input()

                # === Training ===
                print("Starting training...")
                self.policy_path = self.run_training(
                    previous_policy_path=self.policy_path,
                    current_dataset_path=self.aggregated_dataset
                )
                print(f"✅ Policy saved: {self.policy_path}")
                input()

            # Optionally: store final policy per goal
            print(f"🎉 Finished all iterations for goal {goal_index}")


@hydra.main(config_path="../cfgs", config_name="iter_locosafedagger.yaml")
def main(cfg):
    pipeline = SafeDAggerPipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()