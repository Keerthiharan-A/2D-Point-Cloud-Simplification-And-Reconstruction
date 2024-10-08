import numpy as np
import matplotlib.pyplot as plt

class PointNoiseGenerator:
    def __init__(self, points):
        self.points = points
        self.min_x, self.max_x, self.min_y, self.max_y = self.bounding_box()
        self.width = self.max_x - self.min_x
        self.height = self.max_y - self.min_y

    def bounding_box(self):
        """Calculate the bounding box of the input points."""
        min_x = np.min(self.points[:, 0])
        max_x = np.max(self.points[:, 0])
        min_y = np.min(self.points[:, 1])
        max_y = np.max(self.points[:, 1])
        return min_x, max_x, min_y, max_y

    def generate_distorted_noise(self, noise_percentage):
        """Add Gaussian noise to a specified percentage of the points."""
        num_points = self.points.shape[0]

       # num_noisy_points = int(noise_percentage * num_points)
        num_noisy_points = len(self.points[:, 0])
        
        # Randomly choose indices for noisy points
        noisy_indices = np.random.choice(num_points, num_noisy_points, replace=False)
  
        # Calculate standard deviations for x and y based on bounding box size
        # std_dev_x = 0.04 * self.width
        # std_dev_y = 0.04 * self.height

        std_dev_x = noise_percentage * self.width
        std_dev_y = noise_percentage * self.height
        
        # Add Gaussian noise only to the selected  points
        # noisy_points = self.points.copy()
        # noisy_points[noisy_indices, 0] += np.random.normal(0, std_dev_x, size=num_noisy_points)
        # noisy_points[noisy_indices, 1] += np.random.normal(0, std_dev_y, size=num_noisy_points)

        # Add Gaussian noise only to all the points
        noisy_points = self.points.copy()
        noisy_points[:, 0] += np.random.normal(0, std_dev_x, size=num_noisy_points)
        noisy_points[:, 1] += np.random.normal(0, std_dev_y, size=num_noisy_points)

        return noisy_points

    def generate_outliers(self, num_outliers):
        outliers_x = np.random.uniform(self.min_x, self.max_x, num_outliers)
        outliers_y = np.random.uniform(self.min_y, self.max_y, num_outliers)
        outliers = np.column_stack((outliers_x, outliers_y))
        return outliers

    def plot_points(self, noisy_points):
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Plot original points
        axs[0].scatter(self.points[:, 0], self.points[:, 1], color='blue', label='Original Points')
        axs[0].set_title('Original Points')
        axs[0].set_xlabel('X')
        axs[0].set_ylabel('Y')
        axs[0].legend()
        axs[0].set_aspect('equal', adjustable='box')
        
        # Plot noisy points
        axs[1].scatter(noisy_points[:, 0], noisy_points[:, 1], color='green', label='Noisy Points')
        axs[1].set_title('Points After Adding Noise')
        axs[1].set_xlabel('X')
        axs[1].set_ylabel('Y')
        axs[1].legend()
        axs[1].set_aspect('equal', adjustable='box')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def read_points(file_path):
        """Read input points from a file."""
        return np.loadtxt(file_path)

# Example usage
if __name__ == "__main__":

    input_file = r'/home/user/Documents/Minu/2D Denoising/2D-Point-Cloud-Simplification-And-Reconstruction/2D_Dataset/apple/apple-1.xy'  # Replace this with your input file path
    points = PointNoiseGenerator.read_points(input_file)
    
    generator = PointNoiseGenerator(points)
    noise_percentage = 0.01  # 10% of points will have noise

    # Generate Gaussian noise (on a specified percentage of points)
    noisy_points = generator.generate_distorted_noise(noise_percentage)

    # Plot the original and noisy points separately
    generator.plot_points(noisy_points)