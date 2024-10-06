import os
import numpy as np
from Denoising_Algorithm import Denoising

def process_directory(input_directory, output_directory):
    os.makedirs(output_directory, exist_ok=True)

    for filename in os.listdir(input_directory):
        if filename.endswith('.xy'):
            file_path = os.path.join(input_directory, filename)
            print(f"Processing file: {file_path}")

            # Create a Denoising object
            denoising = Denoising(file_path)

            # Classify and denoise the point set
            denoised_point_set = denoising.classify_and_denoise()

            # Save the denoised output
            output_file_path = os.path.join(output_directory, f"denoised_{filename}")
            np.savetxt(output_file_path, denoised_point_set)
            print(f"Completed processing for: {filename}")

def main():
    # Define input and output directories
    input_directory = ''  # Replace with your input directory
    output_directory = ''  # Replace with your output directory

    # Process the directory
    process_directory(input_directory, output_directory)

if __name__ == "__main__":
    main()