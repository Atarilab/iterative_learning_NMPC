import numpy as np
import os
from typing import List, Tuple
from mj_pin.simulator import Simulator
from mj_pin.utils import mj_frame_pos

class SteppingStonesBase:
    DEFAULT_FILE_NAME = "stepping_stones.npz"
    
    def __init__(self,
                 grid_size: Tuple[int, int] = (9, 9),
                 spacing: Tuple[float, float] = (0.19, 0.13),
                 size_ratio: Tuple[float, float] = (0.65, 0.65),
                 randomize_pos_ratio: float = 0.,
                 randomize_height_ratio: float = 0.,
                 N_to_remove: int = 0,
                 shape : str = "cylinder",
                 height : float = 0.1,
                 **kwargs) -> None:
        """
        Define stepping stones locations on a grid. 

        Args:
            - grid_size (Tuple[int, int], optional): Number of stepping stones node (x, y).
            - spacing (Tuple[float, float], optional): Spacing of the center of the stones (x, y).
            - size_ratio (Tuple[float, float], optional): Size ratio of the stepping 
            stone and the spacing.
            size_ratio[0] * spacing and size_ratio[1] * spacing. Defaults to False.
            - randomize_pos (float, optional): Randomize stepping stone location within it's area 
            without collision. Ratio to the max displacement. Defaults to 0, no displacement.
            - randomize_height_ratio (float, optional): Randomize height between [(1-ratio)*h, (1+ratio)*h].
            - N_to_remove (int, optional): Number of stones to remove.
        """
        self.grid_size = grid_size
        self.randomize_pos_ratio = randomize_pos_ratio
        self.spacing = list(spacing)
        self.size_ratio = list(size_ratio)
        self.randomize_height_ratio = randomize_height_ratio
        self.N_to_remove = N_to_remove
        self.shape = shape
        self.height = height

        self.I = self.grid_size[0]
        self.J = self.grid_size[1]
        self.N = self.I * self.J
        self.id_to_remove = np.array([], dtype=np.int32)
        self.id_kept = np.arange(self.N)
        self.init_stones()
        
    def init_stones(self) -> None:
        """ 
        Init stones positions.
        """
        self._init_center_location()  
        self._init_size()
        self._randomize_height()
        self._randomize_center_location()
        
    def _init_center_location(self) -> None:
        """
        Initialize the center locations of the stepping stones.
        """        
        ix = np.arange(self.I) - self.I // 2
        iy = np.arange(self.J) - self.J // 2
        z = np.full(((self.N, 1)), self.height)

        nodes_xy = np.dstack(np.meshgrid(ix, iy)).reshape(-1, 2)
        stepping_stones_xy = nodes_xy * np.array([self.spacing])
        self.positions = np.hstack((stepping_stones_xy, z))

    def _randomize_height(self) -> None:
        """
        Randomize the height of the stones.
        """
        self.positions[:, -1] = self.height + (np.random.rand(self.N) - 0.5) * 2 * self.randomize_height_ratio * self.height
        
    def _init_size(self) -> None:
        """
        Init the size of the stepping stones.
        """
        size_ratio = np.random.uniform(
            low=self.size_ratio[0],
            high=self.size_ratio[1],
            size=self.N
            )
        self.size = size_ratio * min(self.spacing)
        
    def _randomize_center_location(self) -> None:
        """
        Randomize the center of the stepping stones locations.
        """
        max_displacement_x = (self.spacing[0] - self.size) / 2.
        max_displacement_y = (self.spacing[1] - self.size) / 2.
        
        dx = np.random.uniform(-1., 1., self.N) * max_displacement_x * self.randomize_pos_ratio
        dy = np.random.uniform(-1., 1., self.N) * max_displacement_y * self.randomize_pos_ratio

        self.positions[:, 0] += dx
        self.positions[:, 1] += dy
        
    def remove_random(self, N_to_remove: int = -1, keep: list[int] = []) -> None:
        """
        Randomly remove stepping stones.
        
        Args:
            N_to_remove (int): Number of box to remove.
            keep (list[int]): id of the stepping stones to keep
        """
        # 0 probability for id in keep
        probs = np.ones((self.N,))
        probs[keep] = 0.
        probs /= np.sum(probs)
        
        if N_to_remove == -1:
            N_to_remove = self.N_to_remove
        
        self.id_to_remove = np.random.choice(self.N, N_to_remove, replace=False, p=probs)
        self.id_kept = np.setdiff1d(np.arange(self.N), self.id_to_remove)
            
    def get_closest(self, positions_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the indices and positions of the stepping stones closest to 
        each position in <position_xyz>. 

        Args:
            positions_xyz (np.ndarray): array of N 3D positions [N, 3].
        """
        
        # Squared distance
        diffs = self.positions[:, np.newaxis, :] - positions_xyz[np.newaxis, :, :]
        d_squared = np.sum(diffs**2, axis=-1)

        # Find the indices of the closest points
        closest_indices = np.argmin(d_squared, axis=0)
        
        # Extract the closest points from stepping stones
        closest_points = self.positions[closest_indices]

        return closest_indices, closest_points
    
            
    def get_closest_xy(self, positions_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the indices and positions of the stepping stones closest to 
        each position in <position_xyz>. 

        Args:
            positions_xy (np.ndarray): array of N 3D positions [N, 3].
        """
        
        # Squared distance
        diffs = self.positions[:, np.newaxis, :2] - positions_xy[np.newaxis, :, :2]
        d_squared = np.sum(diffs**2, axis=-1)

        # Find the indices of the closest points
        closest_indices = np.argmin(d_squared, axis=0)
        
        # Extract the closest points from stepping stones
        closest_points = self.positions[closest_indices]

        return closest_indices, closest_points
    
    def set_start_position(self, start_pos: np.array) -> np.ndarray:
        """
        Set closest x, y of stepping stones of the start positions
        to x, y of start positions.

        Args:
            start_pos (np.array): Start positions. Shape [N, 3].
        Returns:
            np.ndarray: stepping stones closest to start positions.
        """
        id_closest_to_start, _ = self.get_closest_xy(start_pos[:, :2])
        self.positions[id_closest_to_start, :2] = start_pos[:, :2] 
        self.positions[id_closest_to_start, 2] = start_pos[:, -1] - 0.015
        self.size[id_closest_to_start] = (self.size_ratio[0] + self.size_ratio[1]) / 2. * min(self.spacing)

        return id_closest_to_start
    
    def pick_random(self, positions_xyz: np.ndarray, d_min : float, d_max : float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pick random stepping stones around given positions at a distance between d_min and d_max.

        Args:
            positions_xyz (np.ndarray): array of N 3D positions [N, 3].
            d_min (float): minimum distance to consider for picking stones.
            d_max (float): maximum distance to consider for picking stones.

        Returns:
            Tuple[np.ndarray, np.ndarray]: id [N], positions [N, 3]
        """
        # Squared distance
        diffs = self.positions[:, np.newaxis, :] - positions_xyz[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diffs**2, axis=-1))

        # Init
        N = len(positions_xyz)
        chosen_indices = np.zeros(N, dtype=np.int32)
        chosen_positions = np.zeros_like(positions_xyz, dtype=np.float32)
        
        for i in range(N):
            # Filter based on d_max
            within_d_min = dist[:, i] >= d_min
            within_d_max = dist[:, i] < d_max
            # Get valid indices
            valid_indices = np.where(np.logical_and(within_d_max, within_d_min))
            
            if len(valid_indices[0]) == 0:
                return None, None

            id = np.random.choice(valid_indices[0], replace=False)
            pos = self.positions[id]
            
            chosen_indices[i] = id
            chosen_positions[i] = pos

        return chosen_indices, chosen_positions

    def save(self, save_dir: str) -> None:
        """
        Save the environment's state as a npz file.
        """
        save_path = os.path.join(save_dir, SteppingStonesBase.DEFAULT_FILE_NAME)
        np.savez(
            save_path,
            grid_size=self.grid_size,
            randomize_pos_ratio=self.randomize_pos_ratio,
            spacing=self.spacing,
            size_ratio=self.size_ratio,
            randomize_height_ratio=self.randomize_height_ratio, 
            shape=self.shape,
            height=self.height,
            positions=self.positions,
            size=self.size,
            id_to_remove=self.id_to_remove
            )
        
    @staticmethod
    def load(env_dir: str) -> 'SteppingStonesBase':
        """
        Load the environment's state from a npz file.
        """
        path = os.path.join(env_dir, SteppingStonesBase.DEFAULT_FILE_NAME)
        data = np.load(path)
        env = SteppingStonesBase(grid_size=tuple(data['grid_size']),
                                spacing=tuple(data['spacing']),
                                size_ratio=tuple(data['size_ratio']),
                                randomize_pos_ratio=data['randomize_pos_ratio'],
                                randomize_height_ratio=data['randomize_height_ratio'],
                                shape=data['shape'],
                                height=data['height'])
        
        env.positions = data['positions']
        env.size = data['size']
        env.id_to_remove = data['id_to_remove']
        
        return env
    
class MjSteppingStones(SteppingStonesBase):
    GREY_RBGA = (0.3, 0.3, 0.3, 1.)
    RED_RGBA = (168/255, 0., 17/255, 1.)
    GREEN_RGBA = (15/255, 130/255, 35/255, 1.)
    
    def set_random_goal(self, sim : Simulator, feet_frames : List[str], d_min : float = 0.) -> np.ndarray:
        """
        Set a random goal for each of the feet at a minimum distance d_min.
        """
        # Set the stepping stones at the feet position
        sim._init_model_data()
        base_pos = sim.q0[:3]
        feet_pos = np.array(
            [mj_frame_pos(sim.mj_model, sim.mj_data, foot)
             for foot
             in feet_frames])
        feet_spacing = feet_pos - base_pos
        feet_spacing[:, -1] = 0.
        
        # Set goal stepping stones so that the displacement
        # of the feet to the nominal configuration
        # on the goal is < max_displacement_feet
        max_displacement_feet = 0.07
        max = self.N * 10
        i = 0
        id_goal = None
        
        while id_goal is None:
            _, center_goal = self.pick_random(base_pos[None, :], d_min, 3*d_min)
            center_feet_goal = center_goal + feet_spacing
            id_goal, _ = self.pick_random(center_feet_goal, 0., max_displacement_feet)
            self.init_stones()
            
            i+=1
            if i > max:
                raise ValueError("Invalid stones setup: goal cannot be found.")
        
        return id_goal
    
    def setup(self,
              sim : Simulator,
              feet_frames : List[str],
              remove_random : int = 0,
              randomize_state : bool = False,
              random_goal : bool = True,
              ) -> None:
        """
        Add stepping stones to a simulator.
        """
        # Initial state above the stepping stones
        sim.set_initial_state()
        q0 = sim.q0
        v0 = sim.v0
        margin_z = 0.015
        q0[2] += self.height + margin_z
        
        # Set the goal indices
        id_goal = []
        if random_goal:
            x_length = self.grid_size[0] * self.spacing[0]
            y_length = self.grid_size[1] * self.spacing[1]
            min_length = min(x_length, y_length)
            d_min = min_length / 3.
            id_goal = self.set_random_goal(sim, feet_frames, d_min).tolist()
            
        # Randomize initial state
        if randomize_state:
            std_pos = 1e-2
            std_quat = 3e-2
            std_q = 5e-2
            std_v = 5e-2
            q0[:3] += np.random.randn(3) * std_pos
            q0[3:7] += np.random.randn(4) * std_quat
            q0[3:7] /= np.linalg.norm(q0[3:7])
            q0[7:] += np.random.randn(len(q0) - 7) * std_q
            v0 += np.random.randn(len(v0)) * std_v
        sim.set_initial_state(q0, v0)
        
        # Set the stepping stones at the feet position
        sim._init_model_data()
        feet_pos = np.array(
            [mj_frame_pos(sim.mj_model, sim.mj_data, foot)
             for foot
             in feet_frames])
        id_start = self.set_start_position(feet_pos).tolist()
        
        # Remove random
        if remove_random > 0:
            id_keep = id_start + id_goal
            self.remove_random(remove_random, id_keep)

        # Add all stepping stones to the simulator
        euler = np.zeros(3)
        sim.edit.reset()
        for id, (pos, s) in enumerate(zip(self.positions, self.size)):
            if id in self.id_to_remove:
                continue
            
            if id in id_goal:
                rgba = MjSteppingStones.GREEN_RGBA
            elif id in id_start:
                rgba = MjSteppingStones.RED_RGBA
            else:
                rgba = MjSteppingStones.GREY_RBGA
                
            h = pos[2]
            pos[2] /= 2.
            if self.shape == "cylinder":
                sim.edit.add_cylinder(
                    pos,
                    s,
                    h,
                    euler,
                    rgba,
                )
                
            else:
                size = [s/2., s/2., pos[2]]
                sim.edit.add_box(
                    pos,
                    size,
                    euler,
                    rgba,
                )
                
        return id_start, id_goal

        
if __name__ == "__main__":
    from mj_pin.utils import get_robot_description
    from mj_pin.simulator import Simulator
    

    robot_description = get_robot_description("go2")
    feet_frames = ["FL", "FR", "RL", "RR"]
    sim = Simulator(robot_description.xml_scene_path)
    stones = MjSteppingStones(
        grid_size=(30, 30),
        spacing=(0.19, 0.19),
        height=0.2,
        size_ratio=(0.45, 0.45),
        randomize_pos_ratio=1.,
        randomize_height_ratio=0.0,
        shape="box")
    stones.setup(sim, feet_frames, 10, randomize_state=True, random_goal=False)
    sim.run()
    