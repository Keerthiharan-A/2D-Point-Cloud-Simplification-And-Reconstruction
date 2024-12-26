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
import sys
import pickle
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Utils.Create_features import Features

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
        #self.plot_delaunay_with_flowers(flower_points)
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

    def __init__(self, noisy_file_path, iterations = 20, gt_file_path = None):
        self.point_path = noisy_file_path
        self.point_set = IdNoise.load_xy_data(noisy_file_path)
        self.gt_file_path = gt_file_path
        self.gt = IdNoise.load_xy_data(self.gt_file_path)
        self.scaled_point_set = self.min_max_scaling(self.point_set)
        self.scaled_gt = self.min_max_scaling(self.gt)
        self.tri = Delaunay(self.point_set) # Finding global DT
        self.neighbors = self.find_neighbors()
        self.flower_points = self.identify_flower_structures()

        self.file_path = noisy_file_path
        # self.points = PointSet(point_set)
        self.iterations = iterations  # Number of denoising iterations

    def chamfer_distance(self):
        # Compute all pairwise distances between points in ground_truth and point_set
        distances_AB = cdist(self.gt, self.point_set, metric='euclidean')
        distances_BA = cdist(self.point_set, self.gt, metric='euclidean')
        # Compute the average of the minimum distances from each point in one set to the other
        chamfer_AB = np.mean(np.min(distances_AB, axis=1))
        chamfer_BA = np.mean(np.min(distances_BA, axis=1))
        # Chamfer distance is the average of these two directed distances
        chamfer_dist = (chamfer_AB + chamfer_BA) / 2.0
        print("Chamfer distance is:", chamfer_dist)
        return chamfer_dist
    
    def min_max_scaling(self, pointset):
        """Scale the points to the range [0, 1]."""
        min_vals = np.min(pointset, axis=0)
        max_vals = np.max(pointset, axis=0)
        min_vals = np.min(pointset, axis=0)
        max_vals = np.max(pointset, axis=0)
        
        # Apply Min-Max scaling
        scaled_points = (pointset - min_vals) / (max_vals - min_vals)
        scaled_points = (pointset - min_vals) / (max_vals - min_vals)
        
        return scaled_points
    
    def rmse(self):
        # Compute the Euclidean distances and RMSE
        mean_squared_error = np.sqrt(np.mean(np.sum((self.ground_truth - self.point_set) ** 2, axis=1)))
        print("RMSE between Ground Truth and Denoised points : ", mean_squared_error)

    def classify_noise(self):
        """Classify the point set and apply denoising if necessary."""
        features_object = Features(self.point_path)
        features = np.array(features_object.get_features())
        #print(features)
        with open(r"best_random_forest_model.pkl", "rb") as input_model:
            classifier_model = pickle.load(input_model)
        label = classifier_model.predict(features.reshape(1, -1))
        #print(classifier_model.predict_proba(features.reshape(1, -1)))
        if label == 0:
            return "Clean"
        elif label == 1:
            return "Distorted"
        else:
            return "Band"
        
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
        #self.plot_delaunay_with_flowers(flower_points)
        print("Total # flower points: ", len(flower_points))
        return flower_points

    def plot_delaunay_with_flowers(self, flower_points):
        points = self.point_set
        plt.triplot(points[:, 0], points[:, 1], self.tri.simplices, color='gray', linestyle='-', alpha=0.7,linewidth=3)
        plt.plot(points[:, 0], points[:, 1], 'o', color='blue', markersize=14)

        # Highlight flower-like structure points
        for idx in flower_points:
            plt.plot(points[idx, 0], points[idx, 1], 'o', color='red', markersize=14, label='Flower Structure' if idx == flower_points[0] else "")

        #plt.legend()
        #plt.xlabel('X')
        #plt.ylabel('Y')
        #plt.title('Delaunay Triangulation with Flower Structure Points')
        plt.gca().invert_xaxis()   
        plt.gca().invert_yaxis()
        plt.axis('off')
  
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

    def clustering(self, n_clusters=4):
        curvatures = []

        for i, point in enumerate(self.point_set):
            neighbors = self.neighbors[i]
            if len(neighbors) < 3:
                curvatures.append(0)
                continue
            neighbor_points = np.array([self.point_set[neighbor[0]] for neighbor in neighbors])
            pca = PCA(n_components=2)
            pca.fit(neighbor_points)
            curvature = pca.explained_variance_ratio_[1] / pca.explained_variance_ratio_[0]
            curvatures.append(curvature)

        curvatures = np.array(curvatures).reshape(-1, 1)
        x_coords = np.array([point[0] for point in self.point_set])
        y_coords = np.array([point[1] for point in self.point_set])

        mask = (curvatures > 0.3).flatten()
        filtered_x = x_coords[mask]
        filtered_y = y_coords[mask]
        filtered_curvatures = curvatures[mask]
        default_color = 'gray'
        # Plotting the filtered points
        plt.figure(figsize=(10, 8))
        plt.scatter(x_coords, y_coords, c=default_color, s=5, edgecolors='k', label='Other points')

        # Plot only the points with curvature > 0.5 and apply heatmap coloring
        scatter = plt.scatter(filtered_x, filtered_y, c=filtered_curvatures, cmap='viridis', s=50, edgecolors='k', label='Curvature > 0.3')

        # Add colorbar for the filtered points
        plt.colorbar(scatter, label='Curvature')

        # Title and labels
        plt.title('All Points with Heatmap on Curvature > 0.3')
        plt.xlabel('X-coordinate')
        plt.ylabel('Y-coordinate')
        plt.legend()
        plt.show()
        return mask

    def denoise_point_set(self):
        #noise_type = "Distorted"
        noise_type = self.classify_noise()
        cd_old = self.chamfer_distance()
        #self.clustering()
        if noise_type == "Band":
            print("band noise")
            for iteration in range(self.iterations):
                denoised_points = []
                for point_idx in range(len(self.point_set)):
                    #denoised_point = self.weighted_least_squares_and_projection(point_idx)
                    denoised_point = self.wls_with_normal(point_idx)
                    denoised_points.append(denoised_point)

                self.point_set = np.array(denoised_points)
                print(f"Iteration {iteration+1} completed.")
                cd_new = self.chamfer_distance()
                # Computing DT again
                self.tri = Delaunay(self.point_set) # Finding global DT
                self.neighbors = self.find_neighbors()
                if cd_old < cd_new:
                    cd_old = cd_new
                    break
                cd_old = cd_new

            denoised_file_path1 =  self.file_path.replace('.xy', f'_denoised_.xy')
            self.save_to_xy_file(self.point_set, denoised_file_path1)

            mask = self.clustering()
            print("After curvature:")

            for iteration in range(self.iterations):
                denoised_points = []
                for point_idx, point in enumerate(self.point_set):
                    if mask[point_idx]:
                        denoised_point = self.quadratic_wls(point_idx)
                        denoised_points.append(denoised_point)
                    else:
                        denoised_points.append(point)
                print(f"Iteration {iteration+1} completed.")
                self.point_set = np.array(denoised_points)
                cd_new = self.chamfer_distance()
                # Computing DT again
                self.tri = Delaunay(self.point_set) # Finding global DT
                self.neighbors = self.find_neighbors()
                if cd_old < cd_new:
                    break
                cd_old = cd_new

            denoised_file_path =  self.file_path.replace('.xy', f'_cluster_denoised_.xy')
            #os.makedirs(os.path.dirname(denoised_file_path), exist_ok=True)
            self.save_to_xy_file(self.point_set, denoised_file_path)
            #print(denoised_file_path)
            app = DualPointVisualizerApp(denoised_file_path1, denoised_file_path)
            app.open_windows()
            #self.chamfer_distance()
           
        elif noise_type == "Distorted":
            print("distorted noise")
            for iteration in range(self.iterations):
                denoised_points = []
                for point_idx in range(len(self.point_set)):
                    #denoised_point = self.weighted_least_squares_and_projection(point_idx)
                    denoised_point = self.wls_with_normal(point_idx)
                    denoised_points.append(denoised_point)

                #print(f"Average Mean squared error {error/len(self.scaled_point_set)}, for iteration {iteration + 1}")
                print(f"Iteration {iteration+1} completed.")
                self.scaled_point_set = np.array(denoised_points)
                #self.chamfer_distance()
                # print(f"Iteration {iteration + 1} completed")
            
            # Save the final denoised points after all iterations
            #denoised_file_path = os.path.join('Denoised_output', self.file_path.replace('.xy', f'_denoised_2iters.xy'))
            #os.makedirs(os.path.dirname(denoised_file_path), exist_ok=True)
            #self.save_to_xy_file(self.scaled_point_set, denoised_file_path)
            #computing the flower points again and doing MLS only on those points
            denoised_points_iter = PointSet(self.scaled_point_set)
            flower_points_set = denoised_points_iter.flower_points

            denoised_points = self.scaled_point_set
            # Loop over all points in the point set
            for idx, point in enumerate(self.scaled_point_set):
                 if idx in flower_points_set:
                    for idx1, _ in denoised_points_iter.neighbors[idx]:
            #         # Apply denoising method to flower points
            #             #denoised_points[idx1] = self.weighted_least_squares_and_projection(idx1)
                         denoised_points[idx1] = self.wls_with_normal(idx1)

            self.scaled_point_set = np.array(denoised_points) 
            denoised_file_path = self.file_path.replace('.xy', '_flower_denoised.xy')
            os.makedirs(os.path.dirname(denoised_file_path), exist_ok=True)
            self.save_to_xy_file(self.scaled_point_set, denoised_file_path)
            app = DualPointVisualizerApp(self.file_path, denoised_file_path)
            app.open_windows()  
            #self.chamfer_distance()
            #self.rmse()
        else:
            print("The given point set is clean")

    # def weighted_linear_regression(self, X, y, weights):

    #     def loss(params):
    #         m, c = params
    #         residuals = y - (m * X + c)
    #         return np.sum(weights * residuals**2)
        
    #     def constraint(params):
    #         m, c = params
    #         return m + c - 1

    #     initial_guess = [0, 0]
    #     constr = {'type': 'eq', 'fun': constraint}

    #     result = minimize(loss, initial_guess, constraints=constr)
    #     m_opt, c_opt = result.x

    #     return m_opt, c_opt

    def quadratic_wls(self, point_idx): 
        neighbor_indices = [neighbor[0] for neighbor in self.neighbors[point_idx]]
        distances = [neighbor[1] for neighbor in self.neighbors[point_idx]]
        
        q1, q3 = np.percentile(distances, [25, 75])
        iqr = q3 - q1
        threshold = q3 + 1.5 * iqr
        
        filtered_neighbors = [(idx, dist) for idx, dist in zip(neighbor_indices, distances) if dist <= threshold]
        if len(filtered_neighbors) < 3:
            return self.point_set[point_idx]
        
        neighbor_indices = [idx for idx, _ in filtered_neighbors]
        distances = [dist for _, dist in filtered_neighbors]
        neighbor_points = self.point_set[neighbor_indices]

        X = neighbor_points[:, 0]
        y = neighbor_points[:, 1]
        weights = 1 / np.array(distances)

        X_quad = np.column_stack((X**2, X, np.ones_like(X)))
        model = sm.WLS(y, X_quad, weights=weights)
        results = model.fit()
        a, b, c = results.params

        original_point = self.point_set[point_idx]
        proj_x = original_point[0]
        proj_y = a * proj_x**2 + b * proj_x + c

        return np.array([proj_x, proj_y])

    def weighted_least_squares_and_projection(self, point_idx):
        neighbor_indices = [neighbor[0] for neighbor in self.neighbors[point_idx]]
        distances = [neighbor[1] for neighbor in self.neighbors[point_idx]]
        
        # Calculate Q1, Q3, and IQR
        q1, q3 = np.percentile(distances, [25, 75])
        iqr = q3 - q1
        threshold = q3 + 1.5*iqr
        
        filtered_neighbors = [(idx, dist) for idx, dist in zip(neighbor_indices, distances) if dist <= threshold]
        if len(filtered_neighbors) < 2:
            return self.point_set[point_idx]

        # Prepare data for WLS
        neighbor_indices = [idx for idx, _ in filtered_neighbors]
        distances = [dist for _, dist in filtered_neighbors]
        neighbor_points = self.point_set[neighbor_indices]

        X = neighbor_points[:, 0]
        y = neighbor_points[:, 1]
        weights = 1 / np.array(distances)
        X = sm.add_constant(X)
        model = sm.WLS(y, X, weights=weights)
        results = model.fit()
        # print(results.params, end=" ")
        # print(X, end=" ")
        # print(y)
        if len(results.params) == 1:
            slope = float('inf')
            intercept = results.params[0]
        else:
            intercept, slope = results.params

        #slope, intercept = self.weighted_linear_regression(X,y,weights)
        #print(f"Subject to constraints, Slope is {slope}, Intercept is {intercept}")
        # Calculate the closest point on the line to the original point
        original_point = self.point_set[point_idx]
        if np.isinf(slope):  # Special case for vertical line
            proj_x = neighbor_points[:, 0].mean()
            proj_y = original_point[1]
        else:
            proj_x = (original_point[0] + slope * (original_point[1] - intercept)) / (slope**2 + 1)
            proj_y = slope * proj_x + intercept

        return np.array([proj_x, proj_y])

    def wls_with_normal(self, point_idx):
        neighbor_indices = [neighbor[0] for neighbor in self.neighbors[point_idx]]
        distances = [neighbor[1] for neighbor in self.neighbors[point_idx]]

        q1, q3 = np.percentile(distances, [25, 75])
        iqr = q3 - q1
        threshold = q3 + 1.5 * iqr
        filtered_neighbors = [(idx, dist) for idx, dist in zip(neighbor_indices, distances) if dist <= threshold]

        if len(filtered_neighbors) < 2:
            return self.point_set[point_idx]

        neighbor_indices = [idx for idx, _ in filtered_neighbors]
        distances = [dist for _, dist in filtered_neighbors]
        neighbor_points = self.point_set[neighbor_indices]
        weights = 1 / np.array(distances)

        centroid = np.average(neighbor_points, axis=0, weights=weights)
        
        # Covariance matrix for the neighbors
        relative_positions = neighbor_points - centroid
        cov_matrix = np.cov(relative_positions.T, aweights=weights)

        # Eigen decomposition to find the direction of least variance
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        normal_direction = eigenvectors[:, 0]
        
        original_point = self.point_set[point_idx]
        displacement = np.dot(original_point - centroid, normal_direction)
        updated_point = original_point - displacement * normal_direction
        return updated_point

    def save_to_xy_file(self, points, file_path):
        """Save points to an .xy file."""
        np.savetxt(file_path, points, fmt='%.6f')
        print(f"Denoised points saved to {file_path}")

noisy_file_path = r'Feature_data/Monitor0.xy'  # Replace with your .xy file path
gt_file_path = r'Feature_data/mullets/BandNoise/mullets.xy'
denoising = Denoising(noisy_file_path, 15, gt_file_path)
denoising.denoise_point_set()

# Running commands
# conda activate denoising
# python Scripts\Denoising_Algorithm.py