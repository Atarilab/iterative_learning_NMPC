from __future__ import absolute_import
import numpy as np
import h5py
import torch
from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename

class Database():
    def __init__(self, limit:int, norm_input:bool=True,goal_type:str="vc"):
        assert goal_type in ["vc","cc"]
        
        # initialize database parameters
        self.limit = limit
        self.length = 0
        self.start = 0
        self.file = None
        self.mode = None
        
        # initialize database buffer
        self.groups = ["states","vc_goals","actions"]
        self.states = [None for _ in range(self.limit)]
        self.vc_goals = [None for _ in range(self.limit)]
        self.actions = [None for _ in range(self.limit)]
        
        # policy input parameters
        self.norm_input = norm_input
        
        # set goal type
        self.goal_type = goal_type
        
        # Normalization parameters
        self.states_norm = None
        self.states_mean = None
        self.states_std = None
        
        self.vc_goals_norm = None
        self.vc_goals_mean = None
        self.vd_goals_std = None
        
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        if self.norm_input:
            state = self.states[index]
            
            if self.goal_type == "vc":
                goal = self.vc_goals_norm[index]
            else:
                print("cc goals not implemented yet")
        
        else:
            state = self.states[index]
            
            if self.goal_type == "vc":
                goal = self.vc_goals[index]
            else:
                print("cc goals not implemented yet")
        
        action = self.actions[index]
        x = np.hstack((state,goal))
        y = action
        
        return x,y
    
    def append(self,states,actions,vc_goals=None):
        if vc_goals is None:
            raise ValueError('vc_goals cant be empty')
        
        for i in range(len(states)):
            if self.length < self.limit:
                # still have space
                self.length += 1
            
            elif self.length == self.limit:
                # no space, remove the first item
                print("WARNING! Database overflow!!!!!!!!!!!!!")
            
            else:
                raise RuntimeError
            
            # Append data into buffer
            data_index = (self.start + self.length - 1) % self.limit
            self.states[data_index] = states[i]
            self.actions[data_index] = actions[i]
            
            if vc_goals is not None:
                self.vc_goals[data_index] = vc_goals[i]
        
        # update mean std
        self.calc_input_mean_std(vc_goal = vc_goals)
    
    def calc_input_mean_std(self, vc_goal = None):
        # compute state mean std norm
        self.states_mean = np.mean(np.array(self.states[:self.length]),axis=0)
        self.states_std = np.std(np.array(self.states[:self.length]),axis = 0)
        
        states_array = np.array(self.states[:self.length]) 
        states_norm = np.copy(states_array)
        states_norm[:, 1:] = (states_array[:, 1:] - self.states_mean[1:]) / self.states_std[1:]
        self.states_norm = states_norm
        
        # compute vc_goal mean std norm
        if vc_goal is not None:
            self.vc_goals_mean = 0.0
            self.vc_goals_std = 1.0
            self.vc_goals_norm = (np.array(self.vc_goals[:self.length]) - self.vc_goals_mean) / self.vc_goals_std         
        
    def load_saved_database(self,filename:str=None):
        if filename is None:
            Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
            filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file        
        
        if len(filename) == 0:
            raise FileNotFoundError()
        
        with h5py.File(filename, 'r') as hf:
            states = hf['states'][:]
            actions = hf['actions'][:]
            
            try:
                vc_goals = hf['vc_goals'][:]
            except Exception:
                print('Velocity Constained goal not found in database')
                vc_goals = None
            
            # append data into database
            self.append(states, actions, vc_goals=vc_goals)
        
        # calculate normalization parameters
        self.calc_input_mean_std(vc_goal=vc_goals)
    
    def get_database_mean_std(self):
        if self.norm_input:
            if self.goal_type == "vc":
                return [self.states_mean, self.states_std, self.vc_goals_mean, self.vc_goals_std]
            else:
                return None
        else:
            return None