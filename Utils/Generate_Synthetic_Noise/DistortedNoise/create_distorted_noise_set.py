import os
import numpy as np
from generateDistortedNoise import PointNoiseGenerator  # Ensure this is the correct import

def save_noise_points(file_path, points):
    """Saves the noisy points to a specified file."""
    np.savetxt(file_path, points, fmt='%.6f', delimiter=' ')

def process_dataset(dataset_dir, noise_levels):
    """Processes each folder in the dataset to add noise to point sets."""
    # Iterate through each folder in the dataset directory
    for object_folder in os.listdir(dataset_dir):
        object_folder_path = os.path.join(dataset_dir, object_folder)
        
        if os.path.isdir(object_folder_path):
            # Create the DistortedNoise folder inside the object folder
            distorted_noise_folder = os.path.join(object_folder_path, 'DistortedNoise')
            os.makedirs(distorted_noise_folder, exist_ok=True)

            # Find the .xy file in the current object folder
            xy_file = None
            
            for file in os.listdir(object_folder_path):
                if file.endswith('.xy'):
                    xy_file = file
                    break  # We only need the first .xy file found
            
            if xy_file:
                # Read points from the .xy file
                points = PointNoiseGenerator.read_points(os.path.join(object_folder_path, xy_file))
                generator = PointNoiseGenerator(points)
                
                # Generate noise for each specified noise level
                for nl in noise_levels:
                    noise_points = generator.generate_distorted_noise(nl)
                    
                    # Create the new filename with the format filename-rvalue.xy
                    new_file_name = f"{os.path.splitext(xy_file)[0]}-{nl}.xy"
                    new_file_path = os.path.join(distorted_noise_folder, new_file_name)
                    
                    # Save the noisy point set to the new file
                    save_noise_points(new_file_path, noise_points)

if __name__ == "__main__":
    # Parameters
    dataset_dir = '/home/user/Documents/Minu/2D Denoising/2D-Point-Cloud-Simplification-And-Reconstruction/New_Data'  # Path to the dataset directory
    noise_levels = [0.005, 0.01, 0.015, 0.02]  # Values of noise levels to apply

    # Process the dataset and generate noise
    process_dataset(dataset_dir, noise_levels)

    print("Noise generation and file saving complete.")