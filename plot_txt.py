import matplotlib.pyplot as plt
import numpy as np

def plot_noise_band(points):
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))

    # Plot original points
    ax1.scatter(points[:, 0], points[:, 1], color='blue', label='Point Set')
    ax1.set_title('Original Points')
    ax1.set_aspect('equal')
    ax1.legend()

    plt.tight_layout()
    plt.show()

points = np.loadtxt(r'C:\Users\KEERTHIHARAN\OneDrive\Desktop\butterfly_2percent.txt')
plot_noise_band(points)