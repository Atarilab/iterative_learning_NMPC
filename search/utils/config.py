import yaml
import os
from dataclasses import dataclass

@dataclass
class ConfigBase:
    def __post_init__(self):
        self.file_name = "config.yaml"
    
    def _path_from_dir(self, file_dir: str) -> str:
        os.makedirs(file_dir, exist_ok=True)
        print(file_dir, self.file_name)
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