import numpy as np
import matplotlib.pyplot as plt

def bounding_box(points):
    """Calculate the bounding box of the input points."""
    min_x = np.min(points[:, 0])
    max_x = np.max(points[:, 0])
    min_y = np.min(points[:, 1])
    max_y = np.max(points[:, 1])
    return min_x, max_x, min_y, max_y

def generate_gaussian_noise(points, noise_percentage, width, height):
    """Add Gaussian noise to a specified percentage of the points."""
    # Calculate number of points to add noise to
    num_points = points.shape[0]
    num_noisy_points = int(noise_percentage * num_points)
    
    # Randomly choose indices for noisy points
    noisy_indices = np.random.choice(num_points, num_noisy_points, replace=False)
    
    # Calculate standard deviations for x and y based on bounding box size
    std_dev_x = 0.04 * width
    std_dev_y = 0.04 * height
    
    # Add Gaussian noise only to the selected noisy points
    noisy_points = points.copy()
    noisy_points[noisy_indices, 0] += np.random.normal(0, std_dev_x, size=num_noisy_points)
    noisy_points[noisy_indices, 1] += np.random.normal(0, std_dev_y, size=num_noisy_points)

    return noisy_points

def generate_outliers(num_outliers, min_x, max_x, min_y, max_y):
    outliers_x = np.random.uniform(min_x, max_x, num_outliers)
    outliers_y = np.random.uniform(min_y, max_y, num_outliers)
    outliers = np.column_stack((outliers_x, outliers_y))
    return outliers

def plot_points_separately(original_points, noisy_points):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plot original points
    axs[0].scatter(original_points[:, 0], original_points[:, 1], color='blue', label='Original Points')
    axs[0].set_title('Original Points')
    axs[0].set_xlabel('X')
    axs[0].set_ylabel('Y')
    axs[0].legend()
    axs[0].set_aspect('equal', adjustable='box')
    
    # Plot noisy points
    axs[1].scatter(noisy_points[:, 0], noisy_points[:, 1], color='green', alpha=0.6, label='Noisy Points')
    axs[1].set_title('Points After Adding Noise')
    axs[1].set_xlabel('X')
    axs[1].set_ylabel('Y')
    axs[1].legend()
    axs[1].set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.show()

def read_points_from_file(file_path):
    """Read input points from a file."""
    return np.loadtxt(file_path)

if __name__ == "__main__":
    # Load your point data from a file
    input_file = r'C:\Users\KEERTHIHARAN\Downloads\Figure1.txt'  # Replace this with your input file path
    points = read_points_from_file(input_file)
    
    # Calculate the bounding box and its width and height
    min_x, max_x, min_y, max_y = bounding_box(points)
    width = max_x - min_x
    height = max_y - min_y
    
    noise_percentage = 0.1  # 20% of points will have noise
    # num_outliers = 50       # Number of outliers to generate

    # Generate Gaussian noise (on a specified percentage of points)
    noisy_points = generate_gaussian_noise(points, noise_percentage, width, height)

    # Plot the original and noisy points separately
    plot_points_separately(points, noisy_points)
    
    # np.savetxt("output_noisy_points.txt", noisy_points, fmt='%.6f', delimiter=' ')