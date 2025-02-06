import yaml
import os
from dataclasses import dataclass

@dataclass
class ConfigBase:
    def __post_init__(self):
        self.file_name = "config.yaml",
    
    @staticmethod
    def _path_from_dir(file_dir: str) -> str:
        os.makedirs(file_dir, exist_ok=True)
        file_path = os.path.join(file_dir,file_path)
        return file_path
    
    def save(self, file_dir: str) -> None:
        """
        Save the experiment configuration to a file.
        """
        file_path = self._path_from_dir(file_dir)
        with open(file_path, 'w') as f:
            yaml.dump(self.__dict__, f)

    @staticmethod
    def load(file_dir: str) -> 'ConfigBase':
        """
        Load the experiment configuration from a file.
        """
        file_path = ConfigBase._path_from_dir(file_dir)
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        return ConfigBase(**data)