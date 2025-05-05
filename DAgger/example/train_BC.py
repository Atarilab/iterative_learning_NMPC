# This script is for testing training a Neural Network.
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../..')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.serialization import add_safe_globals
from omegaconf import OmegaConf

import os
import numpy as np
import random
import hydra
import wandb
from tqdm import tqdm
from datetime import datetime

from DAgger.utils.network import GoalConditionedPolicyNet
from DAgger.utils.database import Database

# Set random seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
# add_safe_globals([GoalConditionedPolicyNet])

# Login to wandb
wandb.login()


class BehavioralCloning:
    def __init__(self, cfg):
        self.cfg = cfg
        
        # define device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Model Parameters
        self.n_state = cfg.n_state + 3
        self.n_action = cfg.n_action
        
        # # only for contact conditioned goals
        # self.goal_horizon = cfg.goal_horizon
        
        # define input normalization
        self.normalize_policy_input = cfg.normalize_policy_input
        self.action_type = cfg.action_type
        
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
        print("whole dataset size = ", dataset_size)
        print("training dataset size = ",train_size)
        print("validation dataset size = ", test_size)
        print("learning rate = ", self.learning_rate)
        print("number of epochs = ", self.n_epoch)
                
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
                y_pred = network(x)
                
                loss = self.criterion(y_pred, y)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
                
                # print("network input = ",x[0])                    
                # print("shape of x is = ",x.shape)
                # print("to be matched output = ",y[0])
                # print("shape of y is = ",y.shape)
                # print("calculated output = ",y_pred[0])
                # print("current training loss = ", loss)
                # input()
            
            valid_losses = []
            network.eval()
            with torch.no_grad():
                for z, w in test_loader:
                    z, w = z.to(self.device).float(), w.to(self.device).float()
                    w_pred = network(z)
                    test_loss = self.criterion(w_pred, w)
                    valid_losses.append(test_loss.item())
            
            # for z, w in test_loader:
            #     z, w = z.to(self.device).float(), w.to(self.device).float()
            #     w_pred = network(z)
            #     test_loss = self.criterion(w_pred, w)
            #     valid_losses.append(test_loss.item())
            
            # wandb log    
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
        payload = {
            'network_state_dict': network.state_dict(),
            'norm_policy_input': None
        }
        if self.normalize_policy_input:
            payload["norm_policy_input"] = self.mean_std
            
        torch.save(payload, save_path)
        print(f"Network saved at {save_path}")

    def run(self):
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
        
        # Load database
        self.database = Database(limit=self.cfg.database_size, norm_input=self.normalize_policy_input)
        filename = self.cfg.database_path
        self.database.load_saved_database(filename)
        self.mean_std = self.database.get_database_mean_std()
        # print(self.mean_std)
        # input()
        
        # Network saving
        self.network_savepath = os.path.join(os.path.dirname(filename), '../network')
        os.makedirs(self.network_savepath, exist_ok=True)
                
        wandb.init(project='policy_training', config={'goal_type': 'vc'}, name='vc_training')
        self.database.set_goal_type('vc')
        self.network = self.train_network(self.network)
        wandb.finish()


@hydra.main(config_path='../cfgs', config_name='train_BC_config.yaml')
def main(cfg):
    bc = BehavioralCloning(cfg)
    bc.run()

if __name__ == '__main__':
    main()