import os
import numpy as np
import matplotlib.pyplot as plt

def compare_and_visualize_data(file1: str, file2: str, save_dir: str = "./comparison_plots"):
    """
    Compare data from two simulation files and visualize the differences.

    Args:
        file1 (str): Path to the first .npz file (e.g., recorded data from MPC).
        file2 (str): Path to the second .npz file (e.g., replayed data from the DummyController).
        save_dir (str): Directory to save the comparison plots.
    """
    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Load data from files
    data1 = np.load(file1)
    data2 = np.load(file2)

    # Validate required keys in both files
    required_keys = ["q", "v", "ctrl", "time"]
    for key in required_keys:
        if key not in data1 or key not in data2:
            raise KeyError(f"Both files must contain '{key}' data.")

    # Extract data
    q1, v1, ctrl1, time1 = data1["q"], data1["v"], data1["ctrl"], data1["time"]
    q2, v2, ctrl2, time2 = data2["q"], data2["v"], data2["ctrl"], data2["time"]

    # Ensure the time vectors are the same length
    min_length = min(len(time1), len(time2))
    time1, time2 = time1[:min_length], time2[:min_length]
    q1, q2 = q1[:min_length], q2[:min_length]
    v1, v2 = v1[:min_length], v2[:min_length]
    ctrl1, ctrl2 = ctrl1[:min_length], ctrl2[:min_length]

    # Calculate differences
    q_diff = np.linalg.norm(q1 - q2, axis=1)
    v_diff = np.linalg.norm(v1 - v2, axis=1)
    ctrl_diff = np.linalg.norm(ctrl1 - ctrl2, axis=1)

    # Visualization
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(time1, q_diff, label="Position Difference (q)", color='b')
    plt.xlabel("Time (s)")
    plt.ylabel("Norm Difference")
    plt.title("Position Difference Over Time")
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(time1, v_diff, label="Velocity Difference (v)", color='g')
    plt.xlabel("Time (s)")
    plt.ylabel("Norm Difference")
    plt.title("Velocity Difference Over Time")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(time1, ctrl_diff, label="Control Difference (ctrl)", color='r')
    plt.xlabel("Time (s)")
    plt.ylabel("Norm Difference")
    plt.title("Control Input Difference Over Time")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/comparison_plot.png")
    plt.show()

    print(f"Comparison plot saved to {save_dir}/comparison_plot.png")

    # Additional statistics
    print("\nSummary of Differences:")
    print(f"Mean Position Difference: {np.mean(q_diff):.6f}")
    print(f"Mean Velocity Difference: {np.mean(v_diff):.6f}")
    print(f"Mean Control Difference: {np.mean(ctrl_diff):.6f}")

# Example usage:

if __name__ == "__main__":
    compare_and_visualize_data(
        file1="/home/atari/workspace/iterative_supervised_learning/utils/data/simulation_data_02_21_2025_15_49_00.npz",
        file2="/home/atari/workspace/iterative_supervised_learning/examples/data/simulation_data.npz"
    )
