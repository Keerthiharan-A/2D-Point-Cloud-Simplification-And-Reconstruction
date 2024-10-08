import numpy as np
from scipy.spatial import cKDTree
from Identifying_Noise import IdNoise 
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import statsmodels.api as sm

class Denoising:

    def __init__(self, file_path,iterations=5):
        self.file_path = file_path
        self.point_set = IdNoise.load_xy_data(file_path)
        self.id_noise = IdNoise(file_path)  # Create an instance of IdNoise
        self.tri = Delaunay(self.point_set) # Finding global DT
        self.neighbors = self.find_neighbors()
        self.iterations = iterations  # Number of denoising iterations


    def classify_noise(self):
        """Classify the point set and apply denoising if necessary."""
        classification = self.id_noise.get_classification()
        print(f"The classification of the point set is: {classification}")
        
        if classification == "Noisy":
            noise_type = self.discern_noise()
            return noise_type
        else:
            # print("Point set is already clean. No denoising required.")
            # return self.point_set  # Return the original point set if clean
            return "Clean"

    def discern_noise(self):
        """Placeholder for the denoising logic."""
        #delaunay = Delaunay(self.point_set)
        flower_points = self.identify_flower_structures()
        # self.plot_delaunay_with_flowers(triangles, flower_points)
        return "Distorted" if len(flower_points) < 0.045*len(self.point_set) else "Band" # Modify this line to return the actual denoised point set
        #return "Distorted" if len(flower_points) < 50 else "Band" # Modify this line to return the actual denoised point set

    def find_neighbors(self):

        neighbors = []

        for _ in range(len(self.point_set)):
            neighbors.append([])

        for simplex in self.tri.simplices:
            for point_idx in simplex:
                for point_idx1 in simplex:
                    dist = np.linalg.norm(self.point_set[point_idx] - self.point_set[point_idx1])
                    if point_idx1 != point_idx and (point_idx1,dist) not in neighbors[point_idx]:                        
                        neighbors[point_idx].append((point_idx1,dist))

        return neighbors
       
    
    def check_flower_structure(self, point_idx):
        big, small = 0, 1e7

        for _, dist in self.neighbors[point_idx]:
            big, small = max(dist,big), min(dist, small)

        # Return true if the largest distance is less than twice the smallest distance
        return big < 2 * small

    def identify_flower_structures(self):
        flower_points = []
        # Check each point for the flower structure
        for i in range(len(self.point_set)):
            if self.check_flower_structure(i):
                flower_points.append(i)
        self.plot_delaunay_with_flowers(flower_points)
        print(len(flower_points))
        return flower_points

    def plot_delaunay_with_flowers(self, flower_points):
        points = self.point_set
        plt.triplot(points[:, 0], points[:, 1], self.tri.simplices, color='gray', linestyle='-', alpha=0.5)
        plt.plot(points[:, 0], points[:, 1], 'o', color='blue')

        # Highlight flower-like structure points
        for idx in flower_points:
            plt.plot(points[idx, 0], points[idx, 1], 'o', color='red', markersize=10, label='Flower Structure' if idx == flower_points[0] else "")

        plt.legend()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Delaunay Triangulation with Flower Structure Points')
        plt.show()

    @staticmethod
    def plot_points(points):
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))

        ax1.scatter(points[:, 0], points[:, 1], color='blue', label='Point Set')
        ax1.set_title('Denoised Points')
        ax1.set_aspect('equal')
        ax1.legend()

        plt.tight_layout()
        plt.show()

    def denoise_point_set(self):
        noise_type = self.classify_noise()
        if noise_type == "Clean":
            return self.point_set
        elif noise_type == "Band":
            # Denoise for Band
            # for each point 
            for iteration in range(self.iterations):
                denoised_points = []
                for point_idx in range(len(self.point_set)):
                    denoised_point = self.weighted_least_squares_and_projection(point_idx)
                    denoised_points.append(denoised_point)
                
                self.point_set = np.array(denoised_points)  # Update point set for next iteration
                print(f"Iteration {iteration + 1} completed")
            
            # Save the final denoised points after all iterations
            self.save_to_xy_file(self.point_set, self.file_path.replace('.xy', f'_denoised_{self.iterations}iters.xy'))
            return self.point_set
           
        else:
            
            return 0
    def weighted_least_squares_and_projection(self, point_idx):
        neighbor_indices = [neighbor[0] for neighbor in self.neighbors[point_idx]]
        distances = [neighbor[1] for neighbor in self.neighbors[point_idx]]
        
        # Calculate Q1, Q3, and IQR
        q1, q3 = np.percentile(distances, [25, 75])
        iqr = q3 - q1
        threshold = q3 + 1.1 * iqr
        
        # Filter neighbors based on distance threshold
        filtered_neighbors = [(idx, dist) for idx, dist in zip(neighbor_indices, distances) if dist <= threshold]
        if len(filtered_neighbors) < 2:
            return self.point_set[point_idx]

        # Prepare data for WLS
        neighbor_indices = [idx for idx, _ in filtered_neighbors]
        distances = [dist for _, dist in filtered_neighbors]
        neighbor_points = self.point_set[neighbor_indices]

        X = sm.add_constant(neighbor_points[:, 0])
        y = neighbor_points[:, 1]
        weights = 1 / np.array(distances)

        model = sm.WLS(y, X, weights=weights)
        results = model.fit()
        intercept, slope = results.params
        # Calculate the closest point on the line to the original point
        original_point = self.point_set[point_idx]
        if np.isinf(slope):  # Special case for vertical line
            proj_x = neighbor_points[:, 0].mean()
            proj_y = original_point[1]
        else:
            proj_x = (original_point[0] + slope * (original_point[1] - intercept)) / (slope**2 + 1)
            proj_y = slope * proj_x + intercept

        return np.array([proj_x, proj_y])
    def save_to_xy_file(self, points, file_path):
        """Save points to an .xy file."""
        np.savetxt(file_path, points, fmt='%.6f')
        print(f"Denoised points saved to {file_path}")

file_path = r'/Users/minureghunath/Documents/Research/2D Denoising/2D-Point-Cloud-Simplification-And-Reconstruction/2D_Dataset/donkey/BandNoise/donkey1-7.5-2.xy'  # Replace with your .xy file path
denoising = Denoising(file_path, iterations=10)
print(denoising.denoise_point_set())