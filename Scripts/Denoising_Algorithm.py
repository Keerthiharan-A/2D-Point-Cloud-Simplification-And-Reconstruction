import numpy as np
from scipy.spatial import cKDTree
from Identifying_Noise import IdNoise 
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import statsmodels.api as sm

class Denoising:

    def __init__(self, file_path):
        self.point_set = IdNoise.load_xy_data(file_path)
        self.id_noise = IdNoise(file_path)  # Create an instance of IdNoise
        self.tri = Delaunay(self.point_set) # Finding global DT
        self.neighbors = self.find_neighbors()

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
        return "Distorted" if len(flower_points) < 0.075*len(self.point_set) else "Band" # Modify this line to return the actual denoised point set
    
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
            return 0
        else:
            
            return 0

file_path = r'/home/user/Documents/Minu/2D Denoising/2D-Point-Cloud-Simplification-And-Reconstruction-main/2D_Dataset/apple/BandNoise/apple-1-7.5-2.xy'  # Replace with your .xy file path
denoising = Denoising(file_path)
print(denoising.denoise_point_set())