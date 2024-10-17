import numpy as np
from scipy.spatial import cKDTree
from Identifying_Noise import IdNoise 
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import statsmodels.api as sm
#from PointCloudVisualizer import PointVisualizerApp
from ViewInputOutput import DualPointVisualizerApp
from scipy.spatial.distance import cdist
import os

class PointSet():

    def __init__(self, point_set):
        self.point_set = point_set
        self.tri = Delaunay(point_set) # Finding global DT
        self.neighbors = self.find_neighbors()
        self.flower_points = self.identify_flower_structures()
        self.neigh1, self.neigh2 = self.count_neighbours()
        self.length = self.calculate_length()

    def calculate_length(self):
        """Calculate the bounding box of the input points."""
        min_x = np.min(self.point_set[:, 0])
        max_x = np.max(self.point_set[:, 0])
        min_y = np.min(self.point_set[:, 1])
        max_y = np.max(self.point_set[:, 1])
        return max(max_x-min_x, max_y-min_y)

    def find_neighbors(self):

        neighbors = []

        for _ in range(len(self.point_set)):
            neighbors.append([])

        for simplex in self.tri.simplices:
            for point_idx in simplex:
                for point_idx1 in simplex:
                    dist = np.linalg.norm(self.point_set[point_idx] - self.point_set[point_idx1])
                    if point_idx1 != point_idx and (point_idx1,dist) not in neighbors[point_idx]:                        
                        neighbors[point_idx].append((point_idx1, dist))

        return neighbors
    
    def check_flower_structure(self, point_idx):
        big, small = 0, 1e7

        for _, dist in self.neighbors[point_idx]:
            big, small = max(dist,big), min(dist, small)

        # Return true if the largest distance is less than twice the smallest distance
        return big < 5 * small
    
    def identify_flower_structures(self):
        flower_points = []
        # Check each point for the flower structure
        for i in range(len(self.point_set)):
            if self.check_flower_structure(i):
                flower_points.append(i)
        self.plot_delaunay_with_flowers(flower_points)
        return set(flower_points)
    
    def count_neighbours(self):
        less5, more5 = 0, 0
        for i in self.flower_points:
            if len(self.neighbors[i]) < 5:
                less5 += 1
            else:
                more5 += 1

        return less5, more5
    
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

class Denoising:

    def __init__(self, noisy_file_path, iterations = 20):
        self.point_set = IdNoise.load_xy_data(noisy_file_path)
        self.gt_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(noisy_file_path))), 'gt.xy')
        self.ground_truth = IdNoise.load_xy_data(self.gt_file_path)
        self.id_noise = IdNoise(noisy_file_path)  # Create an instance of IdNoise
        self.tri = Delaunay(self.point_set) # Finding global DT
        self.neighbors = self.find_neighbors()
        self.file_path = noisy_file_path
        # self.points = PointSet(point_set)
        self.iterations = 35  # Number of denoising iterations
        self.chamfer_distance()

    def chamfer_distance(self):
    # Compute all pairwise distances between points in ground_truth and point_set
        distances_AB = cdist(self.ground_truth, self.point_set, metric='euclidean')
        distances_BA = cdist(self.point_set, self.ground_truth, metric='euclidean')

    # Compute the average of the minimum distances from each point in one set to the other
        chamfer_AB = np.mean(np.min(distances_AB, axis=1))
        chamfer_BA = np.mean(np.min(distances_BA, axis=1))

    # Chamfer distance is the average of these two directed distances
        chamfer_dist = (chamfer_AB + chamfer_BA) / 2.0
        print("Chamfer distance between Ground Truth and Denoised points:", chamfer_dist)

    
    def rmse(self):
        # Compute the Euclidean distances and RMSE
        mean_squared_error = np.sqrt(np.mean(np.sum((self.ground_truth - self.point_set) ** 2, axis=1)))
        print("RMSE between Ground Truth and Denoised points : ", mean_squared_error)

    def classify_noise(self):
        """Classify the point set and apply denoising if necessary."""
        classification = self.id_noise.get_classification()
       #print(f"The classification of the point set is: {classification}")
        points = PointSet(self.point_set)
        print("# flower points with # neighbors < 5: ", points.neigh1)
        print("# flower points with # neighbors >= 5: ", points.neigh2)
        bounding_box_to_flower = points.length/(len(points.flower_points))
        print("Bounding box length to # points: ", points.length/(len(points.flower_points)))
        #print("ratio: " ,bounding_box_to_flower )
        if bounding_box_to_flower <=2:
           return "Band"
        elif  bounding_box_to_flower <=10:
           return "Distorted"
        # if classification == "Noisy":
        #     noise_type = self.discern_noise()
        #     return noise_type
        else:
            # print("Point set is already clean. No denoising required.")
            # return self.point_set  # Return the original point set if clean
           return "Clean"

    def discern_noise(self):
        """Placeholder for the denoising logic."""
        #delaunay = Delaunay(self.point_set)
        flower_points = self.identify_flower_structures()
        # self.plot_delaunay_with_flowers(flower_points)
        return "Distorted" if len(flower_points) < 0.6*len(self.point_set) else "Band" # Modify this line to return the actual denoised point set

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
        return big < 5 * small

    def identify_flower_structures(self):
        flower_points = []
        # Check each point for the flower structure
        for i in range(len(self.point_set)):
            if self.check_flower_structure(i):
                flower_points.append(i)
        # self.plot_delaunay_with_flowers(flower_points)
        print("Total # flower points: ", len(flower_points))
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
        if noise_type == "Band":
            print("band noise")
            for iteration in range(self.iterations):
                denoised_points = []
                for point_idx in range(len(self.point_set)):
                    denoised_point = self.weighted_least_squares_and_projection(point_idx)
                    denoised_points.append(denoised_point)
                
                self.point_set = np.array(denoised_points)  # Update point set for next iteration
                #self.point_set = PointSet(np.array(denoised_points))
                #self.chamfer_distance()
            
            # Save the final denoised points after all iterations
            denoised_file_path = os.path.join('Denoised_output', self.file_path.replace('.xy', f'_denoised_{self.iterations}iters.xy'))
            os.makedirs(os.path.dirname(denoised_file_path), exist_ok=True)
            self.save_to_xy_file(self.point_set, denoised_file_path)
            # app = DualPointVisualizerApp(self.file_path, denoised_file_path)
            # app.open_windows()  # This will correctly initialize and run the main loop   

            # Visualize the denoised file with PointVisualizerApp
                    # after few iterations, check for flower structure
           # Identify flower points and get their indices
            denoised_points_iter = PointSet(self.point_set)
            flower_points_set = denoised_points_iter.flower_points

            denoised_points = self.point_set
            # Loop over all points in the point set
            for idx, point in enumerate(self.point_set):
                if idx in flower_points_set:
                    for idx1, _ in denoised_points_iter.neighbors[idx]:
                    # Apply denoising method to flower points
                        denoised_points[idx1] = self.weighted_least_squares_and_projection(idx1)

            self.point_set = np.array(denoised_points) 
            denoised_file_path =  self.file_path.replace('.xy', f'_flower_denoised_.xy')
            os.makedirs(os.path.dirname(denoised_file_path), exist_ok=True)
            self.save_to_xy_file(self.point_set, denoised_file_path)
            app = DualPointVisualizerApp(self.file_path, denoised_file_path)
            app.open_windows()  # This will correctly initialize and run the main loop
            #self.chamfer_distance()
           
        else:
            print("distorted noise")
            self.rmse()
            for iteration in range(self.iterations):
                denoised_points = []
                for point_idx in range(len(self.point_set)):
                    denoised_point = self.weighted_least_squares_and_projection(point_idx)
                    denoised_points.append(denoised_point)
                
                self.point_set = np.array(denoised_points)  # Update point set for next iteration
                #self.chamfer_distance()
                # print(f"Iteration {iteration + 1} completed")
            
            # Save the final denoised points after all iterations
            denoised_file_path = os.path.join('Denoised_output', self.file_path.replace('.xy', f'_denoised_2iters.xy'))
            os.makedirs(os.path.dirname(denoised_file_path), exist_ok=True)
            self.save_to_xy_file(self.point_set, denoised_file_path)
            denoised_points_iter = PointSet(self.point_set)
            flower_points_set = denoised_points_iter.flower_points

            denoised_points = self.point_set
            # Loop over all points in the point set
            for idx, point in enumerate(self.point_set):
                if idx in flower_points_set:
                    for idx1, _ in denoised_points_iter.neighbors[idx]:
                    # Apply denoising method to flower points
                        denoised_points[idx1] = self.weighted_least_squares_and_projection(idx1)

            self.point_set = np.array(denoised_points) 
            denoised_file_path = os.path.join('Denoised_output', self.file_path.replace('.xy', '_flower_denoised.xy'))
            os.makedirs(os.path.dirname(denoised_file_path), exist_ok=True)
            self.save_to_xy_file(self.point_set, denoised_file_path)
            app = DualPointVisualizerApp(self.file_path, denoised_file_path)
            app.open_windows()  # This will correctly initialize and run the main loop   
            #self.chamfer_distance()
            #self.rmse()

    def weighted_least_squares_and_projection(self, point_idx):
        neighbor_indices = [neighbor[0] for neighbor in self.neighbors[point_idx]]
        distances = [neighbor[1] for neighbor in self.neighbors[point_idx]]
        
        # Calculate Q1, Q3, and IQR
        q1, q3 = np.percentile(distances, [25, 75])
        iqr = q3 - q1
        threshold = q3
        #threshold = 10000
        
        # Filter neighbors based on distance threshold
        #filtered_neighbors = [(idx, dist) for idx, dist in zip(neighbor_indices, distances) if dist > threshold]
        #print(len(filtered_neighbors))
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

noisy_file_path = r'/home/user/Documents/Minu/2D Denoising/2D-Point-Cloud-Simplification-And-Reconstruction/2D_Dataset/teddy/BandNoise/teddy_vary_band.xy'  # Replace with your .xy file path
gt_file_path = r'/home/user/Documents/Minu/2D Denoising/2D-Point-Cloud-Simplification-And-Reconstruction/2D_Dataset/swordfishes/swordfishes.xy'
denoising = Denoising(noisy_file_path, gt_file_path)
denoising.denoise_point_set()

# Idea for teh new discrenment : if (# flower pts)/(bounding box width or length) is <= 2 --> Band noise, else if <10 --> Distorted Noise, else  clean
# Drawbacks : If the points in some animal with legs then the # flower points for a clean set itself is high because the pts in between the legs forms flower points, so samples should be suitably chosen. 
# Advantages : This can plausibly work for sparsley sampled points