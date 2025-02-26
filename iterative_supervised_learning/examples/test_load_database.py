import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../..')
import numpy as np
import torch
from iterative_supervised_learning.utils.database import Database  # Ensure this imports the correct Database class

# Path to the saved database (Update this if needed)
database_path = "/home/atari/workspace/iterative_supervised_learning/examples/data/behavior_cloning/trot/Feb_26_2025_10_35_39/dataset/database_0.hdf5"

# Initialize Database with a suitable limit
db = Database(limit=10000000,norm_input=False)  # Ensure the limit is large enough to load the full dataset

# Load the saved database
print(f"Loading database from: {database_path}")
db.load_saved_database(database_path)

# Check database size
print(f"Database loaded successfully. Total stored samples: {len(db)}")

# Retrieve and print a few samples
num_samples_to_print = 15
for i in range(min(num_samples_to_print, len(db))):
    x, y = db[i]
    print(f"\nSample {i+1}:")
    print(f"Input (state + goal): {x.shape}, Data: {x}")
    print(f"Output (action): {y.shape}, Data: {y}")

# Check normalization parameters
mean_std = db.get_database_mean_std()
if mean_std:
    print("\nDatabase Normalization Parameters:")
    print(f"States Mean: {mean_std[0]}")
    print(f"States Std: {mean_std[1]}")
    print(f"Goal Mean: {mean_std[2]}")
    print(f"Goal Std: {mean_std[3]}")
else:
    print("\nNormalization is disabled.")

# check for database length
print("current database length = ",db.length)