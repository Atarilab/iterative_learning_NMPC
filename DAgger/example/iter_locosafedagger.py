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

    def run(self,goal_index, goal):
        for i in range(self.cfg.n_iteration):
            print(f"\n🔁 Starting SafeDAgger Iteration {i}")
            if i == 0:
                self.policy_path = self.cfg.initial_policy_path
                self.aggregated_dataset = self.cfg.pretrain_dataset_path

            print(f"Current policy path: {self.policy_path}")
            print(f"Previous dataset path: {self.aggregated_dataset}")
            input()
            
            print("Starting data collection...")
            new_dataset_path = self.run_data_collection(iter_index = i,
                                                        goal_index = goal_index,
                                                        previous_dataset_path = self.aggregated_dataset,
                                                        current_policy_path = self.policy_path,
                                                        goal = goal)
            # prepare dataset for training and next iteration
            self.aggregated_dataset = new_dataset_path
            print(f"✅ Aggregated dataset saved: {self.aggregated_dataset}")
            input()
            
            print("Starting training...")
            print(f"Current policy path: {self.policy_path}")
            print(f"Previous dataset path: {self.aggregated_dataset}")
            input()
            self.policy_path = self.run_training(previous_policy_path=self.policy_path,
                                                 current_dataset_path=self.aggregated_dataset)
            print(f"✅ Policy saved: {self.policy_path}")
            input()


@hydra.main(config_path="../cfgs", config_name="iter_locosafedagger.yaml")
def main(cfg):
    pipeline = SafeDAggerPipeline(cfg)
    goal_list = [[0.15,0.0,0.0],
                 [0.0,0.0,0.0],
                 [-0.15,0.0,0.0]]
    for idx, goal in enumerate(goal_list):
        print("\n" + "="*50)
        print(f"🚩 Starting rollout for Goal {idx+1}: {goal}")
        print("="*50)
        pipeline.run(goal)


if __name__ == "__main__":
    main()