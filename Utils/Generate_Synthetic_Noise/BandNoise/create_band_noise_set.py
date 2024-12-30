import os
import shutil
import numpy as np
from generateBandNoise import generate_band_noise  # Importing the noise generation function

def save_noise_points(file_path, points):
    np.savetxt(file_path, points, fmt='%.6f', delimiter=' ')

def rename_txt_to_xy(folder_path):
    """
    Rename all .txt files in a folder to .xy.
    """
    for file in os.listdir(folder_path):
        if file.endswith('.txt'):
            base_name = os.path.splitext(file)[0]
            new_name = f"{base_name}.xy"
            old_path = os.path.join(folder_path, file)
            new_path = os.path.join(folder_path, new_name)
            os.rename(old_path, new_path)

def process_dataset(dataset_dir, r_values, rl_values):
    """
    Process the dataset by generating band noise for each .xy file.
    If the folder contains .txt files, rename them to .xy before processing.
    """
    # Iterate through each folder in the dataset directory
    band_noise_folder = os.path.join(dataset_dir, 'BandNoise')
    os.makedirs(band_noise_folder, exist_ok=True)

    # Iterate through all files in the main directory
    for file in os.listdir(dataset_dir):
        if file.endswith('.xy'):
            xy_file = file

            # Read the original points from the .xy file
            points = np.loadtxt(os.path.join(dataset_dir, xy_file))

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
    dataset_dir = r'D:\2D-Point-Cloud-Simplification-And-Reconstruction\NonManifold_data\multiple_components\mc1'  # Path to dataset directory
    r_values = [7.5, 10, 12.5]  # Values of r (circle radius)
    rl_values = [2, 2, 5]  # Values of rl (noise level)

    # Process the dataset and generate noise
    process_dataset(dataset_dir, r_values, rl_values)

    print("Noise generation and file saving complete.")
