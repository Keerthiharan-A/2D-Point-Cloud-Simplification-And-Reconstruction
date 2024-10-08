import matplotlib.pyplot as plt
import numpy as np

def load_points_from_file(file_path):
    # Load points from an .xy file
    points = np.loadtxt(file_path)
    return points

def plot_data(points):
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))

    # Plot original points
    ax1.scatter(points[:, 0], points[:, 1], color='blue', label='Point Set')
    ax1.set_title('Original Points')
    ax1.set_aspect('equal')
    ax1.legend()

    plt.tight_layout()
    plt.show()

points = np.loadtxt(r'/home/user/Documents/Minu/2D Denoising/2D-Point-Cloud-Simplification-And-Reconstruction/2D_Dataset/bird/BandNoise/bird-17-7.5-2_denoised_10iters.xy')
plot_data(points)