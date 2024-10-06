import numpy as np
import matplotlib.pyplot as plt

def generate_band_noise(points, max_r, nl):
    noise_points = []

    # Generate noise points by distributing them across the band area for each point
    for point in points:
        # Randomly choose a radius for the current point
        r = np.random.uniform(0, max_r)
        # r = np.random.normal(loc=max_r / 2, scale=max_r / 4)
        # r = np.clip(r, 0, max_r) 
        # Calculate points per circle proportional to the maximum radius
        points_per_circle = int(nl * r)
        
        for _ in range(points_per_circle):  # Generate specified number of points per circle
            angle = np.random.uniform(0, 2 * np.pi)  # Random angle around the circle
            # Introduce variability in the radius for each point
            noise_radius = np.random.uniform(0, r)  # Random distance for this noise point
            new_x = point[0] + noise_radius * np.cos(angle)
            new_y = point[1] + noise_radius * np.sin(angle)
            noise_points.append([new_x, new_y])

    # Convert noise_points list to a numpy array
    noise_points = np.array(noise_points)

    # Ensure that noise_points is a 2D array, even if no points are generated
    if noise_points.size == 0:
        noise_points = np.empty((0, 2))

    return noise_points

def plot_noise_band(points, noise_points):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot original points
    ax1.scatter(points[:, 0], points[:, 1], color='blue', label='Original Points', s=10)  # Increase marker size
    ax1.set_title('Original Points')
    ax1.set_aspect('equal')
    ax1.legend()

    # Plot noisy points
    ax2.scatter(noise_points[:, 0], noise_points[:, 1], color='red', alpha=0.6, label='Noise Points', s=10)  # Smaller marker size
    ax2.set_title('Noisy Points')
    ax2.set_aspect('equal')
    ax2.legend()

    plt.tight_layout()
    plt.show()

def read_points_from_file(file_path):
    return np.loadtxt(file_path)

if __name__ == "__main__":
    # Read input points from file
    input_file = r'C:\Users\KEERTHIHARAN\Downloads\Figure10.txt'  # Replace this with the path to your input file
    points = read_points_from_file(input_file)

    # Parameters
    max_r = 20  # Maximum radius for noise generation
    nl = 2        # Base number of noise points to generate per unit of radius

    # Generate noise band
    noise_points = generate_band_noise(points, max_r, nl)

    # Plot original points and noise band
    plot_noise_band(points, noise_points)

    # Optionally save the noise points to a file
    # np.savetxt("output_noise_points.txt", noise_points, fmt='%.6f', delimiter=' ')