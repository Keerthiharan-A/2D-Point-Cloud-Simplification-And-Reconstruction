import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

class IdNoise:

    def __init__(self, file_path):
        self.point_set = self.load_xy_data(file_path)

    @staticmethod
    def load_xy_data(file_path):
        """Load point set from a .xy file."""
        return np.loadtxt(file_path)

    def compute_distance_and_counts(point_set):
        """Compute the average distance to closest neighbors and the counts."""
        tree = cKDTree(point_set)

        distances, _ = tree.query(point_set, k=2)  # k=2 to get the closest neighbor
        closest_distances = distances[:, 1]
        #self.plot_distances(closest_distances)
        # Calculate the average distance
        average_distance = np.mean(closest_distances) + 2 * np.std(closest_distances)

        # Plot the distances
        #self.plot_distances(closest_distances)

        counts = []
        for point in point_set:
            count = np.sum(np.linalg.norm(point_set - point, axis=1) <= average_distance)
            counts.append(count)

        counts = np.array(counts)

        average_count = np.mean(counts)
        std_dev_count = np.std(counts)

        return np.mean(closest_distances), np.std(closest_distances), average_count, std_dev_count

    @staticmethod
    def plot_distances(closest_distances):
        """Plot the distribution of distances to closest neighbors."""
        plt.figure(figsize=(10, 6))
        plt.hist(closest_distances, bins=30, color='skyblue', edgecolor='black')
        plt.title('Distribution of Distances to Closest Neighbors')
        plt.xlabel('Distance')
        plt.ylabel('Frequency')
        plt.grid()
        plt.show()

    def get_classification(self):
        """Classify the point set as Clean or Noisy based on the average count."""
        

    @staticmethod
    def plot_frequency(counts):
        """Plot the frequency of counts of points within average distance."""
        plt.figure(figsize=(10, 6))
        plt.hist(counts, bins=np.arange(counts.min(), counts.max() + 1) - 0.5,
                 color='skyblue', edgecolor='black')
        plt.title('Frequency Plot of Counts of Points Within Average Distance')
        plt.xlabel('Count of Points')
        plt.ylabel('Frequency')
        plt.grid()
        plt.xticks(np.arange(counts.min(), counts.max() + 1))
        plt.show()

# Example usage
# file_path = r'/home/user/Documents/Minu/2D Denoising/2D-Point-Cloud-Simplification-And-Reconstruction-main/2D_Dataset/watch/DistortedNoise/watch-1-0.1.xy'  # Replace with your .xy file path'  # Replace with your .xy file path
# id_noise = IdNoise(file_path)

# average_count, std_dev_count, counts = id_noise.compute_average_distance_and_counts()


# print(f"Average count of points in surrounding circle: {average_count:.4f}")
# print(f"Standard deviation of counts: {std_dev_count:.4f}")

# #Plot the frequency of counts
# IdNoise.plot_frequency(counts)

# #Check classification
# classification = id_noise.get_classification()
# print(f"The classification of the point set is: {classification}")