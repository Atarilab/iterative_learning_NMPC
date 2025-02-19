# This script is for testing training a Neural Network.
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../..')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

import os
import numpy as np
import random
import hydra
import wandb
from tqdm import tqdm
from datetime import datetime

from iterative_supervised_learning.utils.network import GoalConditionedPolicyNet
from iterative_supervised_learning.utils.database import Database

# Set random seed for reproducibility
# seed = 42
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)

# Login to wandb
wandb.login()


class BehavioralCloning:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Model Parameters
        self.n_state = cfg.n_state + 3
        self.n_action = cfg.n_action
        # only for contact conditioned goals
        self.goal_horizon = cfg.goal_horizon
        self.normalize_policy_input = cfg.normalize_policy_input
        
        # Training properties
        self.batch_size = cfg.batch_size
        self.learning_rate = cfg.learning_rate
        self.n_epoch = cfg.n_epoch
        self.n_train_frac = cfg.n_train_frac
        
        # Loss function
        self.criterion = nn.L1Loss()
        
    def initialize_network(self, input_size = 0, output_size= 0, num_hidden_layer=3, hidden_dim=512, batch_norm=True):
        network = GoalConditionedPolicyNet(
            input_size, output_size, num_hidden_layer, hidden_dim, batch_norm
        ).to(self.device)
        print("Policy Network initialized")
        return network
    
    def train_network(self, network):
        dataset_size = len(self.database)
        train_size = int(self.n_train_frac * dataset_size)
        test_size = dataset_size - train_size
        
        train_data, test_data = torch.utils.data.random_split(self.database, [train_size, test_size])
        train_loader = DataLoader(train_data, self.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_data, self.batch_size, shuffle=True, drop_last=True)
        
        optimizer = torch.optim.Adam(network.parameters(), lr=self.learning_rate)
        
        for epoch in tqdm(range(self.n_epoch)):
            network.train()
            train_losses = []
            
            for x, y in train_loader:
                optimizer.zero_grad()
                x, y = x.to(self.device).float(), y.to(self.device).float()
                
                # print("network input = ",x)
                # # Check if the first column of x is within [0,1]
                # first_entry = x[:, 0]  # Extract the first feature of each sample
                # out_of_bounds = (first_entry < 0) | (first_entry > 1)  # Find out-of-bound values

                # if out_of_bounds.any():
                #     print("Warning! Some first entries are out of bounds:")
                #     print(first_entry[out_of_bounds])  # Print only the values that are out of range
                # else:
                #     print("All first entries of x are within [0,1]")
                    
                # print("shape of x is = ",x.shape)
                # print("to be matched output = ",y)
                # print("shape of y is = ",y.shape)
                
                y_pred = network(x)
                # print("calculated output = ",y_pred)
                # input()
                loss = self.criterion(y_pred, y)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            
            valid_losses = []
            network.eval()
            with torch.no_grad():
                for z, w in test_loader:
                    z, w = z.to(self.device).float(), w.to(self.device).float()
                    w_pred = network(z)
                    test_loss = self.criterion(w_pred, w)
                    valid_losses.append(test_loss.item())
            
            wandb.log({
                'Training Loss': np.mean(train_losses),
                'Validation Loss': np.mean(valid_losses)
            })
            
            if epoch % 10 == 0:
                self.save_network(network, f"policy_{epoch}")
        
        self.save_network(network, "policy_final")
        return network
    
    def save_network(self, network, name):
        save_path = os.path.join(self.network_savepath, f"{name}.pth")
        torch.save({'network': network.state_dict()}, save_path)
        print(f"Network saved at {save_path}")
    
    def run(self):
        # self.vc_input_size = self.n_state + 5
        # self.cc_input_size = self.n_state + (self.goal_horizon * 3 * 4)
        
        self.input_size = self.n_state
        self.output_size = self.n_action
        
        self.network = self.initialize_network(
            input_size=self.input_size, 
            output_size=self.output_size, 
            num_hidden_layer=self.cfg.num_hidden_layer,
            hidden_dim=self.cfg.hidden_dim
        )
       
        # Print model architecture
        print("\n=== Initialized Network Structure ===")
        print(self.network)
        print("num of weights = ", sum(p.numel() for p in self.network.parameters() if p.requires_grad))
        #==================================================================================
        input()
        # self.cc_network = self.initialize_network(
        #     self.cc_input_size, self.output_size, self.cfg.num_hidden_layer, self.cfg.hidden_dim
        # )
        
        self.database = Database(limit=self.cfg.database_size, norm_input=self.normalize_policy_input)
        filename = self.cfg.database_path
        self.database.load_saved_database(filename)
        
        self.network_savepath = os.path.join(os.path.dirname(filename), '../network')
        os.makedirs(self.network_savepath, exist_ok=True)
        
        # wandb.init(project='policy_training', config={'goal_type': 'cc'}, name='cc_training')
        # self.database.set_goal_type('cc')
        # self.cc_network = self.train_network(self.cc_network)
        # wandb.finish()
        
        wandb.init(project='policy_training', config={'goal_type': 'vc'}, name='vc_training')
        self.database.set_goal_type('vc')
        self.network = self.train_network(self.network)
        wandb.finish()


@hydra.main(config_path='cfgs', config_name='bc_config.yaml')
def main(cfg):
    bc = BehavioralCloning(cfg)
    bc.run()

if __name__ == '__main__':
    main()
