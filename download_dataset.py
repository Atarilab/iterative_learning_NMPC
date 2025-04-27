# this dataset download is optional, but recommended if you want to run policy rollout
from huggingface_hub import snapshot_download
import os

# Your repo name
repo_id = "Chiniklas/behavior_cloning_data"  # <-- Change if your repo name is different

# Target directory where you want to save it
target_dir = "Behavior_Cloning/examples/data"

# Make sure the folder exists
os.makedirs(target_dir, exist_ok=True)

# Download the dataset
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=target_dir,
    local_dir_use_symlinks=False,  # Actually copy files, not just symlinks
)

print(f"✅ Dataset downloaded successfully to: {target_dir}")
