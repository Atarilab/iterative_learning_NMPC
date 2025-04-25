from __future__ import absolute_import
import numpy as np
import h5py
import torch
from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename
    

class Database():
    def __init__(self, limit:int, norm_input:bool=True, goal_type:str='vc'):
        
        assert goal_type in ['vc', 'cc'], 'Goal type can only be vc or cc'
        
        # database parameters
        self.limit = limit        
        self.length = 0
        self.start = 0
        self.file = None
        self.mode = None
        self.traj_ids = [None for _ in range(self.limit)]      # e.g., "nominal", "perturbed_001"
        self.traj_times = [None for _ in range(self.limit)]    # timestep within trajectory

        # database variables
        self.groups = ['states', 'vc_goals', 'cc_goals', 'actions']  # vc: velocity conditioned, cc: contact conditioned
        self.states = [None for _ in range(self.limit)]
        self.vc_goals = [None for _ in range(self.limit)]
        self.cc_goals = [None for _ in range(self.limit)]
        self.actions = [None for _ in range(self.limit)]
        
        # policy input parameters
        self.norm_input = True
        self.set_normalize_input(norm_input)
        
        # set goal type
        self.goal_type = None
        self.set_goal_type(goal_type)
        
        # Normalization parameters
        self.states_norm = None
        self.vc_goals_norm = None
        self.cc_goals_norm = None
        
        self.states_mean = None
        self.states_std = None
        self.vc_goals_mean = None
        self.vc_goals_std = None
        self.cc_goals_mean = None
        self.cc_goals_std = None
        
    def __len__(self):
        """get database length
        """        
        return self.length
    
    def __getitem__(self, index):
        """get item with norm and goal type consideration

        Args:
            index (_type_): index

        Returns:
            x, y: as torch tensors
        """        
        if self.norm_input:
            state = self.states_norm[index]
            
            if self.goal_type =='vc':
                goal = self.vc_goals_norm[index]
            elif self.goal_type == 'cc':
                goal = self.cc_goals_norm[index]
                
        else:
            state = self.states[index]
            
            if self.goal_type =='vc':
                goal = self.vc_goals[index]
            elif self.goal_type == 'cc':
                goal = self.cc_goals[index]
            
        action = self.actions[index]
        
        x = torch.from_numpy(np.hstack((state, goal)))
        y = torch.from_numpy(action)
            
        return x, y
    
    def set_normalize_input(self, value:bool):
        """set if input to policy should be normalized

        Args:
            value (bool): boolean
        """        
        print('__getitem__ of database set to normalized: ' + str(value))
        self.norm_input = value
        
    def set_goal_type(self, value:str):
        """set goal conditioning type

        Args:
            value (str): goal type
        """        
        assert value in ['vc', 'cc'], 'Goal type can only be vc or cc'
        print('goal type of database set to: ' + str(value))
        self.goal_type = value

    def append(self, states, actions, vc_goals=None, cc_goals=None,traj_id=None, times=None):
        """Append data to database

        Args:
            states (_type_): robot state
            actions (_type_): controller action
            vc_goals (_type_, optional): vc goal. Defaults to None.
            cc_goals (_type_, optional): cc goal. Defaults to None.

        Raises:
            ValueError: if no goal is specified
            RuntimeError: index overflow
        """        
        if vc_goals is None and cc_goals is None:
            raise ValueError('both vc_goals and cc_goals cant be empty!')
        
        # loop to save each element to database
        for idx in range(len(states)):
            if self.length < self.limit:
                # We have space, simply increase the length.
                self.length += 1
                
            elif self.length == self.limit:
                # No space, "remove" the first item.
                print('Warning! Database overflow!')
                self.start = (self.start + 1) % self.limit
            else:
                # This should never happen.
                raise RuntimeError()   
            
            # append data into buffer
            data_index = (self.start + self.length - 1) % self.limit
            self.states[data_index] = states[idx]
            self.actions[data_index] = actions[idx]
            
            if vc_goals is not None:
                self.vc_goals[data_index] = vc_goals[idx]
                
            if cc_goals is not None:
                self.cc_goals[data_index] = cc_goals[idx]
                
            # Store metadata
            if traj_id is not None:
                self.traj_ids[data_index] = traj_id[idx] 
            if times is not None:
                self.traj_times[data_index] = times[idx]
        
        # update mean and std
        self.calc_input_mean_std(vc_goal=vc_goals, cc_goal=cc_goals)
    
    def load_saved_database(self, filename:str=None):
        """load database with data saved as h5py file

        Args:
            filename (str, optional): file path. Defaults to None.

        Raises:
            FileNotFoundError: file not found at file path
        """        
        # prompt user to choose filename if not specified
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
            
            try:
                cc_goals = hf['cc_goals'][:]
            except Exception:
                print('Contact Constained goal not found in database') 
                cc_goals = None   
            
            try:
                traj_ids = hf['traj_ids'][:]
                traj_ids = traj_ids.astype(int).tolist()
            except KeyError:
                traj_ids = None

            try:
                traj_times = hf['traj_times'][:]
            except KeyError:
                traj_times = None

            self.append(
                states, actions, vc_goals=vc_goals, cc_goals=cc_goals,
                traj_id=traj_ids, times=traj_times
            )
        
        # calculate normalization parameters
        self.calc_input_mean_std(vc_goal=vc_goals, cc_goal=cc_goals)
    
    def calc_input_mean_std(self, vc_goal=None, cc_goal=None):
        """calculate normalization parameters

        Args:
            vc_goal (_type_, optional): velocity conditioned goals. Defaults to None.
            cc_goal (_type_, optional): contact conditioned goals. Defaults to None.
        """ 
        # Get the current mean and std of the database for each element of the input along time. 
        # Not of all the inputs for one time step!
        # States
        # current implementation put phase percentage as the first entry of state, so it must be constraint within 0-1
        
        # Compute mean and std across all states
        self.states_mean = np.mean(np.array(self.states[:self.length]), axis=0)
        self.states_std = np.std(np.array(self.states[:self.length]), axis=0)
        
        # Normalize all but the first column (first feature)
        states_array = np.array(self.states[:self.length])  # Convert to array for easier slicing
        states_norm = np.copy(states_array)  # Create a copy to preserve structure
        
        # Normalize only columns 1-44 (excluding the first column)
        states_norm[:, 1:] = (states_array[:, 1:] - self.states_mean[1:]) / self.states_std[1:]
        
        # The first column remains unchanged
        self.states_norm = states_norm  # here we only set states_norm to normalized state
        # print("shape of self.states is = ", np.shape(self.states[:self.length]))
        # print("shape of states_mean is = ",np.shape(self.states_mean))
        # print("shape of states_norm is = ",np.shape(self.states_norm))
        
        # print(np.array(self.states[:self.length])[:1,:])
        # input()
        # print(self.states_norm[:1,:])
        
        
        # vc goal will not be normalized as phi is already constrainted betweeon 0 - 1
        # desired velocities are also always constant throughout the rollout process
        if vc_goal is not None:
            self.vc_goals_mean = 0.0
            self.vc_goals_std = 1.0
            self.vc_goals_norm = (np.array(self.vc_goals[:self.length]) - self.vc_goals_mean) / self.vc_goals_std   
            # print("normalized vc_goal is = ",self.vc_goals_norm)
            
        # cc goal
        # if None not in np.array(self.cc_goals[:self.length]):
        if cc_goal is not None:
            self.cc_goals_mean = np.mean(np.array(self.cc_goals[:self.length]), axis=0)
            self.cc_goals_std = np.std(np.array(self.cc_goals[:self.length]), axis=0)
            self.cc_goals_norm = (np.array(self.cc_goals[:self.length]) - self.cc_goals_mean) / self.cc_goals_std       
        
    def get_database_mean_std(self):
        """get current database normalization parameters

        Returns:
            [states_mean, states_std, goal_mean, goal_std]
        """        
        
        if self.norm_input:
            if self.goal_type == 'vc':
                return [self.states_mean, self.states_std, self.vc_goals_mean, self.vc_goals_std]
            elif self.goal_type == 'cc':
                return [self.states_mean, self.states_std, self.cc_goals_mean, self.cc_goals_std]
            else:
                return None
        else:
            return None
    
    def save_as_npz(self, filename: str):
        """Save the in-memory database to a compressed .npz file."""
        np.savez(
            filename,
            states=np.array(self.states[:self.length]),
            vc_goals=np.array(self.vc_goals[:self.length]),
            cc_goals=np.array(self.cc_goals[:self.length]),
            actions=np.array(self.actions[:self.length])
        )
        print(f"Saved OOD dataset to: {filename}")
    
    def load_from_npz(self, filename: str):
        """
        Load database contents from a compressed .npz file.
        This method assigns the data, sets the length, and computes normalization.
        """
        data = np.load(filename)

        # Check required fields
        required_fields = ["states", "vc_goals", "cc_goals", "actions"]
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing field '{field}' in NPZ file.")

        # Assign data
        self.states = data["states"]
        self.vc_goals = data["vc_goals"]
        self.cc_goals = data["cc_goals"]
        self.actions = data["actions"]
        self.length = len(self.states)

        # Ensure fields are truncated to length if needed (e.g., if stored as longer buffers)
        self.states = self.states[:self.length]
        self.vc_goals = self.vc_goals[:self.length]
        self.cc_goals = self.cc_goals[:self.length]
        self.actions = self.actions[:self.length]

        # Compute normalization
        self.calc_input_mean_std(vc_goal=self.vc_goals, cc_goal=self.cc_goals)

        print(f"Successfully loaded {self.length} samples from {filename}")

