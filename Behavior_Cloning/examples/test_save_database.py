import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../..')

import pickle
import hydra
import h5py
import numpy as np
from datetime import datetime
from Behavior_Cloning.utils.database import Database

class DatasetCollector:
    def __init__(self, cfg, experiment_dir):
        self.cfg = cfg
        self.experiment_dir = experiment_dir
        self.gaits = cfg.gaits if hasattr(cfg, 'gaits') else ['default']
        self.database = Database(limit=cfg.database_size,norm_input=True)  # Assumes Database has an append method and required attributes
        self.data_save_path = self._prepare_save_path()

    def _prepare_save_path(self):
        current_time = datetime.now().strftime("%b_%d_%Y_%H_%M_%S")
        base_path = os.path.join(self.cfg.data_save_path, "behavior_cloning", '_'.join(self.gaits))
        if self.cfg.suffix:
            base_path += f"_{self.cfg.suffix}"
        return os.path.join(base_path, current_time, "dataset")

    def load_data_from_npz_files(self):
        total_file_data_len = 0
        file_count = 0

        for file_name in os.listdir(self.experiment_dir):
            file_path = os.path.join(self.experiment_dir, file_name)
            if file_name.endswith(".npz") and os.path.isfile(file_path):
                print(f"Loading data from: {file_path}")
                data = np.load(file_path)
                
                length = data["state"].shape[0]
                total_file_data_len += length
                file_count += 1

                self.database.append(
                    states=data["state"],
                    vc_goals=data["vc_goals"],
                    cc_goals=data["cc_goals"],
                    actions=data["action"]
                )

        print(f"\n✅ Loaded {file_count} files from {self.experiment_dir}")
        print(f"🔢 Total samples from .npz files: {total_file_data_len}")
        print(f"📦 Total samples in database:     {len(self.database)}")

        if total_file_data_len != len(self.database):
            print("⚠️  Mismatch between loaded samples and database entries!")
        else:
            print("✅ Data length check passed.")


    def save_dataset(self, iteration):
        os.makedirs(self.data_save_path, exist_ok=True)

        data_len = len(self.database)
        with h5py.File(os.path.join(self.data_save_path, f"database_{iteration}.hdf5"), 'w') as hf:
            hf.create_dataset('states', data=self.database.states[:data_len])
            hf.create_dataset('vc_goals', data=self.database.vc_goals[:data_len])
            hf.create_dataset('cc_goals', data=self.database.cc_goals[:data_len])
            hf.create_dataset('actions', data=self.database.actions[:data_len])

        config_path = os.path.join(self.data_save_path, "config.pkl")
        if not os.path.exists(config_path):
            with open(config_path, "wb") as f:
                pickle.dump(self.cfg, f)

        print(f"Dataset saved at iteration {iteration + 1}")

# Example usage:
@hydra.main(config_path='cfgs', config_name='data_collection_config.yaml',version_base="1.1")
def main(cfg):
    collector = DatasetCollector(cfg, experiment_dir="/home/atari/workspace/Behavior_Cloning/examples/data/behavior_cloning/trot/Mar_27_2025_11_53_51/dataset/experiment")
    collector.load_data_from_npz_files()
    collector.save_dataset(iteration=0)

if __name__ == "__main__":
    main()
