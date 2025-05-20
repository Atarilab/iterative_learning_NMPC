import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GoalConditionedPolicyNet_experimental(nn.Module):
    def __init__(self, input_size:int, output_size:int, num_hidden_layer:int=4, hidden_dim:int=256, batch_norm:bool=False):
        """policy network

        Args:
            input_size (int): input dimension
            output_size (int): output dimension
            num_hidden_layer (int, optional): number of hidden layers. Defaults to 4.
            hidden_dim (int, optional): number of nodes per hidden layer. Defaults to 256.
            batch_norm (bool, optional): bool if 1D batch norm should be performed between layers. Defaults to False.
        """     
           
        super().__init__()
        
        assert num_hidden_layer > 0, 'number of hidden layers should be greater than 0'
        
        print('input_size: ', input_size)
        print('num hidden layers: ', num_hidden_layer)
        print('hidden dim: ', hidden_dim)
        print('batch normalization: ', str(batch_norm))
        print('output_size: ', output_size)
        
        # input layer
        layers = [nn.Linear(input_size, hidden_dim)]
        
        # batch norm
        if batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
                
        # ReLu activation function
        layers.append(nn.ReLU())
            
        # loop for each hidden layer
        for _ in range(1, num_hidden_layer):
            # fully connected layer
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            
            # batch normalization if true
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
                
            # ReLu activation function
            layers.append(nn.ReLU())
            
        # output layer
        layers.append(nn.Linear(hidden_dim, output_size))
        
        # stack all layers into a network
        self.net = nn.Sequential(*layers)
        
        # weight initialization
        self.init_weight()
    
    def init_weight(self):
        """Kaiming Initialization
        """        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Initialize the weight tensors with Xavier/Glorot initialization
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                
                # Initialize biases to zero
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """network forward pass

        Args:
            x (_type_): input

        Returns:
            network output
        """        
        return self.net(x)
