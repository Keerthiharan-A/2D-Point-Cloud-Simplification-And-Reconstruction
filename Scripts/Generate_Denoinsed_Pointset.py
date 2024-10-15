import os
import numpy as np
from Denoising_Algorithm import Denoising

import os

import os

def process_directory(input_directory):
    # Loop through each folder in the input directory
    for folder in os.listdir(input_directory):
        main_img = os.path.join(input_directory, folder)
        
        if os.path.isdir(main_img):
            for noise_type in os.listdir(main_img):
                noise_dir = os.path.join(main_img, noise_type)
                
                if os.path.isdir(noise_dir):
                    for noisy_file in [f for f in os.listdir(noise_dir) if f.endswith('.xy')]:
                        file_path = os.path.join(noise_dir, noisy_file)
                        print(f"Processing file: {noisy_file}")
                        denoising = Denoising(file_path)
                        denoising.denoise_point_set()
                        print(f"Completed processing for: {noisy_file}")

def main():
    # Define input and output directories
    input_directory = '/home/user/Documents/Minu/2D Denoising/2D-Point-Cloud-Simplification-And-Reconstruction/2D_Dataset'  # Replace with your input directory

    # Process the directory
    process_directory(input_directory)

if __name__ == "__main__":
    main()