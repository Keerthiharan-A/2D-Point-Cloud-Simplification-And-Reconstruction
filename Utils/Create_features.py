import numpy as np
from scipy.spatial import Delaunay
import os
import sys

# Add parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Scripts.Identifying_Noise import IdNoise

class Features:
    """Class to extract features from a point set."""
    
    def __init__(self, file_path):
        self.point_set = self.load_xy_data(file_path)
        if self.point_set.size == 0:  # Check if the point set is empty
            raise ValueError(f"The file {file_path} is empty. Skipping this file.")
        
        self.scaled_point_set = self.min_max_scaling()
        self.tri = Delaunay(self.scaled_point_set)
        self.neighbors = self.find_neighbors()
        self.flower_points = self.identify_flower_structures()
        self.neigh1, self.neigh2 = self.count_neighbours()
        self.mean_closest_distance, self.std_closest_distance, self.average_count, self.std_count = IdNoise.compute_distance_and_counts(self.scaled_point_set)
    
    def get_features(self):
        """Return the extracted features."""
        return (len(self.scaled_point_set), len(self.flower_points)/len(self.scaled_point_set), self.neigh1, 
                self.neigh2, self.mean_closest_distance, self.std_closest_distance, 
                self.std_count)
    
    def load_xy_data(self, file_path):
        """Load point set from a .xy file."""
        try:
            data = np.loadtxt(file_path)
            if data.size == 0:  # Check if the loaded data is empty
                print(f"Warning: The file {file_path} is empty.")
                return np.array([])  # Return an empty array if the file is empty
            return data
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return np.array([])  # Return an empty array if loading fails
    
    def min_max_scaling(self):
        """Scale the points to the range [0, 1]."""
        min_vals = np.min(self.point_set, axis=0)
        max_vals = np.max(self.point_set, axis=0)
        scaled_points = (self.point_set - min_vals) / (max_vals - min_vals)
        
        return scaled_points
    
    def find_neighbors(self):
        """Find neighbors for each point in the point set."""
        neighbors = [[] for _ in range(len(self.scaled_point_set))]

        for simplex in self.tri.simplices:
            for point_idx in simplex:
                for point_idx1 in simplex:
                    if point_idx1 != point_idx:
                        dist = np.linalg.norm(self.scaled_point_set[point_idx] - self.scaled_point_set[point_idx1])
                        if (point_idx1, dist) not in neighbors[point_idx]:                        
                            neighbors[point_idx].append((point_idx1, dist))

        return neighbors
    
    def check_flower_structure(self, point_idx):
        """Check if a point has a flower structure."""
        big, small = 0, 1e7
        for _, dist in self.neighbors[point_idx]:
            big = max(dist, big)
            small = min(dist, small)

        return big < 5 * small
    
    def identify_flower_structures(self):
        """Identify flower structures in the point set."""
        flower_points = {i for i in range(len(self.scaled_point_set)) if self.check_flower_structure(i)}
        return flower_points
    
    def count_neighbours(self):
        """Count points with less than and more than 5 neighbors."""
        more5,fmore5 = 0, 0
        for i in self.scaled_point_set:
            if i in self.flower_points:
                if len(self.neighbors[i]) >= 5:
                    fmore5 += 1
            if len(self.neighbors[i]) >= 5:
                more5 += 1

        return more5/len(self.scaled_point_set), fmore5/len(self.flower_points)