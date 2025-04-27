import numpy as np
import matplotlib.pyplot as plt

def plot_feet_positions_multiple(feet_positions, num_samples=20):
    """
    Plots the (x,y) positions of the four legs over multiple samples in a single plot.
    
    Parameters:
    - feet_positions: A numpy array of shape (N, 12), where N is the number of samples.
                      Each sample contains (x,y,z) coordinates for 4 legs in the format:
                      [x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4].
    - num_samples: The number of samples to plot on the same graph.
    """
    feet_positions = np.array(feet_positions)
    
    # Ensure we don't exceed the available samples
    num_samples = min(num_samples, len(feet_positions))
    
    # Define colors and markers for each leg
    colors = ['r', 'g', 'b', 'm']  # Colors for 4 legs
    markers = ['o', 's', 'D', 'x']  # Different markers for clarity
    
    plt.figure(figsize=(8, 6))
    
    # Plot each sample
    for sample_idx in range(num_samples):
        feet_pos_all = feet_positions[sample_idx]
        
        # Extract (x, y) coordinates for each leg
        x_coords = feet_pos_all[::3]  # [x1, x2, x3, x4]
        y_coords = feet_pos_all[1::3]  # [y1, y2, y3, y4]
        
        # Plot each leg's position with connecting lines
        for i in range(4):
            plt.plot(x_coords[i], y_coords[i], marker=markers[i], color=colors[i], 
                     label=f'Leg {i+1}' if sample_idx == 0 else "", 
                     markersize=8, linestyle='-')
    
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title(f'Feet Positions of 4 Legs (First {num_samples} Samples)')
    plt.axhline(0, color='gray', linewidth=0.5)
    plt.axvline(0, color='gray', linewidth=0.5)
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_base_wrt_feet(base_wrt_feet, num_samples=20):
    """
    Plots the (x,y) positions of the base relative to the feet over multiple samples.
    
    Parameters:
    - base_wrt_feet: A numpy array of shape (N, 8), where N is the number of samples.
                     Each sample contains (x,y) coordinates for 4 legs relative to the base:
                     [bx1, by1, bx2, by2, bx3, by3, bx4, by4].
    - num_samples: The number of samples to plot on the same graph.
    """
    base_wrt_feet = np.array(base_wrt_feet)
    
    # Ensure we don't exceed the available samples
    num_samples = min(num_samples, len(base_wrt_feet))
    
    # Define colors and markers for each leg
    colors = ['r', 'g', 'b', 'm']  # Colors for 4 legs
    markers = ['o', 's', 'D', 'x']  # Different markers for clarity
    
    plt.figure(figsize=(8, 6))
    
    # Plot each sample
    for sample_idx in range(num_samples):
        base_pos_all = base_wrt_feet[sample_idx]
        
        # Extract (x, y) coordinates for each leg relative to the base
        bx_coords = base_pos_all[::2]  # [bx1, bx2, bx3, bx4]
        by_coords = base_pos_all[1::2]  # [by1, by2, by3, by4]
        
        # Plot base relative positions with dashed lines
        for i in range(4):
            plt.plot(bx_coords[i], by_coords[i], marker=markers[i], color=colors[i], 
                     label=f'Leg {i+1} (Base)' if sample_idx == 0 else "", 
                     markersize=8, linestyle='--')
    
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title(f'Base Relative Positions of 4 Legs (First {num_samples} Samples)')
    plt.axhline(0, color='gray', linewidth=0.5)
    plt.axvline(0, color='gray', linewidth=0.5)
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    file_path = "/home/atari/workspace/Behavior_Cloning/examples/data/simulation_data_02_27_2025_13_16_19.npz"
    data = np.load(file_path)
    feet_positions = data["feet_pos_w"]
    base_wrt_feet = data["base_wrt_feet"]
    print("First 5 feet position samples:")
    print(feet_positions[:5])
    
    print("first 5 base_wrt_feet samples")
    print(base_wrt_feet[:5])
    
    # Plot the first 20 samples on the same plot
    plot_feet_positions_multiple(feet_positions, num_samples=4000)
    plot_base_wrt_feet(base_wrt_feet,num_samples=4000)

