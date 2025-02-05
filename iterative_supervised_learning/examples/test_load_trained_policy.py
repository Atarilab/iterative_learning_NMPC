import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../..')

import numpy as np
from iterative_supervised_learning.utils.network import GoalConditionedPolicyNet

# Function to test loading a trained network
def initialize_network(input_size, output_size, num_hidden_layer=3, hidden_dim=512, batch_norm=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = GoalConditionedPolicyNet(
        input_size, output_size, num_hidden_layer, hidden_dim, batch_norm
    ).to(device)
    print("Policy Network initialized")
    return network
    
def test_load_network(policy_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    checkpoint = torch.load(policy_path, map_location=device)
    state_dict = checkpoint["network"]

    # Extract saved model architecture
    print("\n=== Saved Model Keys ===")
    print(state_dict.keys())

    # Infer input/output sizes from state_dict
    inferred_input_size = state_dict["net.0.weight"].shape[1]  # First layer input size
    inferred_output_size = state_dict["net.4.weight"].shape[0]  # Last layer output size

    n_state = 39
    n_action = 12
    # Print architecture information
    print("\n=== Model Metadata ===")
    print(f"Inferred Input Size: {inferred_input_size}")
    print(f"Inferred Output Size: {inferred_output_size}")

    # Create a model with the inferred architecture
    policy_net = initialize_network(input_size=n_state, 
                                          output_size=n_action,
                                          num_hidden_layer=2,  # Adjust based on saved model
                                          hidden_dim=256,      # Adjust based on saved model
                                          batch_norm=True)    # Adjust based on saved model
    policy_net.eval()
    policy_net.to(device)

    # Try loading state dict
    try:
        policy_net.load_state_dict(state_dict, strict=False)
        print("\n✅ Model Loaded Successfully! ✅")
    except RuntimeError as e:
        print("\n❌ Model Loading Failed! ❌")
        print(str(e))

    # Print current model structure
    print("\n=== Current Model Architecture ===")
    print(policy_net)

    # Test forward pass
    test_input = torch.randn(1, inferred_input_size).to(device)
    try:
        with torch.no_grad():
            output = policy_net(test_input)
        print("\n✅ Forward Pass Successful!")
        print("Output shape:", output.shape)
    except Exception as e:
        print("\n❌ Forward Pass Failed!")
        print(str(e))

# Run test
if __name__ == "__main__":
    # policy_path = "/home/atari/workspace/iterative_supervised_learning/examples/data/behavior_cloning/trot/Feb_04_2025_15_05_24/network/policy_final.pth"  # Change this
    policy_path = "/home/atari/workspace/iterative_supervised_learning/examples/data/behavior_cloning/trot/Feb_04_2025_15_05_24/network/policy_140.pth"
    if not os.path.exists(policy_path):
        print(f"❌ Policy file not found: {policy_path}")
        sys.exit(1)
    
    test_load_network(policy_path)
