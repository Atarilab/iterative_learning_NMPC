import yaml
import os
from dataclasses import dataclass

@dataclass
class ConfigBase:
    def __post_init__(self):
        self.file_name = "config.yaml"
    
    def _path_from_dir(self, file_dir: str) -> str:
        os.makedirs(file_dir, exist_ok=True)
        file_path = os.path.join(file_dir, self.file_name)
        return file_path
    
    def save(self, file_dir: str) -> None:
        """
        Save the experiment configuration to a file.
        """
        file_path = self._path_from_dir(file_dir)
        with open(file_path, 'w') as f:
            yaml.safe_dump(self.__dict__, f)

    def load(self, file_dir: str) -> 'ConfigBase':
        """
        Load the experiment configuration from a file.
        """
        file_path = self._path_from_dir(file_dir)
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        
        for k, v in data.items():
            setattr(self, k, v)
        
        return self
    
@dataclass  
class SearchConfig(ConfigBase):
    xml_path : str = "",
    feet_frames_mj : list[str] = [],
    grid_size : tuple[int, int] = (10, 10),
    spacing : tuple[float, float] = (0.19, 0.19),
    size_ratio : tuple[float, float] = (0.45, 0.45),
    randomize_pos_ratio : float = 0.75,
    randomize_height_ratio : float = 0.1,
    n_remove : int = 50,
    height : float = 0.2,
    shape : str = "box",
    max_step_size : float = 0.31,
    max_foot_displacement : float = 0.265,
    
    def __post_init__(self):
        self.file_name = "search_config.yaml"

@dataclass
class RunConfig(ConfigBase):
    xml_path : str = "",
    n_run : int = 1,
    n_cores : int = 1,
    sim_dt : float = 1.0e-3,
    collision : bool = True,
    feet_frames_mj : list[str] = [],
    
    def __post_init__(self):
        self.file_name = "run_config.yaml"