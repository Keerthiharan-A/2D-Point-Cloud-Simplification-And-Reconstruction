import os
import shutil
import numpy as np
from generateBandNoise import generate_band_noise  # Importing the noise generation function

def save_noise_points(file_path, points):
    np.savetxt(file_path, points, fmt='%.6f', delimiter=' ')

def process_dataset(dataset_dir, output_dir, r_values, rl_values):
    # Create the BandNoise folder if it doesn't exist inside each object folder
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through each folder in the dataset directory
    for object_folder in os.listdir(dataset_dir):
        object_folder_path = os.path.join(dataset_dir, object_folder)
        
        if os.path.isdir(object_folder_path):
            # Create BandNoise folder inside the object folder if it doesn't exist
            band_noise_folder = os.path.join(object_folder_path, 'BandNoise')
            os.makedirs(band_noise_folder, exist_ok=True)
            
            # Get the first .xy, .png, and .edg files
            xy_file = None
            png_file = None
            edg_file = None
            
            for file in os.listdir(object_folder_path):
                if file.endswith('.xy') and xy_file is None:
                    xy_file = file
                if file.endswith('.png') and png_file is None:
                    png_file = file
                if file.endswith('.edg') and edg_file is None:
                    edg_file = file
            
            if xy_file and png_file and edg_file:
                # Copy the .xy, .png, and .edg files to BandNoise folder
                shutil.copy(os.path.join(object_folder_path, xy_file), os.path.join(band_noise_folder, xy_file))
                shutil.copy(os.path.join(object_folder_path, png_file), os.path.join(band_noise_folder, png_file))
                shutil.copy(os.path.join(object_folder_path, edg_file), os.path.join(band_noise_folder, edg_file))
                
                # Read the original points from the .xy file
                points = np.loadtxt(os.path.join(band_noise_folder, xy_file))
                
                # Generate noise for all combinations of r and rl
                for r, rl in zip(r_values, rl_values):
                    noise_points = generate_band_noise(points, r, rl)
                    
                    # Create the new filename with format filename-rvalue-rlvalue.xy
                    new_file_name = f"{os.path.splitext(xy_file)[0]}-{r}-{rl}.xy"
                    new_file_path = os.path.join(band_noise_folder, new_file_name)
                    
                    # Save the noisy point set to the new file
                    save_noise_points(new_file_path, noise_points)

if __name__ == "__main__":
    # Parameters
    dataset_dir = 'dataset'  # Path to dataset directory
    output_dir = 'BandNoise'  # Output folder to store files and noisy points within each object folder
    r_values = [7.5, 10, 12.5]  # Values of r (circle radius)
    rl_values = [2, 2.5, 3]  # Values of rl (noise level)

    # Process the dataset and generate noise
    process_dataset(dataset_dir, output_dir, r_values, rl_values)

    print("Noise generation and file saving complete.")