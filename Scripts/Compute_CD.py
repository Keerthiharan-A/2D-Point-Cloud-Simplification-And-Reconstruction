import os
import csv
from Denoising_Algorithm import Denoising

folder_path = "Feature_data"

# Create a CSV file to save the results
csv_file_path = "chamfer_distances.csv"

# Initialize the CSV file with headers
with open(csv_file_path, mode='w', newline='') as csvfile:
    fieldnames = ['File Name', 'Noise Type', 'Noise Level', 'Noisy_CD', 'Denoised_CD', 'No of Iterations']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

# Iterate through the files in the folder and its subfolders
for root, dirs, files in os.walk(folder_path):
    for filename in files:
        noisy_file_path = os.path.join(root, filename)
        
        if filename.endswith('.xy'):  # Process only .xy files
            noise_type = None
            noise_level = None
            ground_truth_file_path = None
            
            # Check if the file is a band noise file
            if filename.endswith(('-7.5-2.xy', '-10-2.xy', '-12.5-5.xy')):
                noise_type = 'Band Noise'
                if filename.endswith('-7.5-2.xy'):
                    noise_level = '(7.5,2)'
                    ground_truth_file_path = os.path.join(root, filename.rsplit('-7.5-2.xy', 1)[0] + '.xy')
                elif filename.endswith('-10-2.xy'):
                    noise_level = '(10,2)'
                    ground_truth_file_path = os.path.join(root, filename.rsplit('-10-2.xy', 1)[0] + '.xy')
                elif filename.endswith('-12.5-5.xy'):
                    noise_level = '(12.5,5)'
                    ground_truth_file_path = os.path.join(root, filename.rsplit('-12.5-5.xy', 1)[0] + '.xy')

            # Check if the file is a distorted noise file
            elif filename.endswith(('-0.01.xy', '-0.005.xy', '-0.02.xy', '-0.015.xy')):
                noise_type = 'Distorted Noise'
                if filename.endswith('-0.01.xy'):
                    noise_level = '1%'
                    ground_truth_file_path = os.path.join(root, filename.rsplit('-0.01.xy', 1)[0] + '.xy')
                elif filename.endswith('-0.005.xy'):
                    noise_level = '0.5%'
                    ground_truth_file_path = os.path.join(root, filename.rsplit('-0.005.xy', 1)[0] + '.xy')
                elif filename.endswith('-0.02.xy'):
                    noise_level = '2%'
                    ground_truth_file_path = os.path.join(root, filename.rsplit('-0.02.xy', 1)[0] + '.xy')
                elif filename.endswith('-0.015.xy'):
                    noise_level = '1.5%'
                    ground_truth_file_path = os.path.join(root, filename.rsplit('-0.015.xy', 1)[0] + '.xy')

            # Skip files that do not match known noise suffixes
            if noise_type is None or ground_truth_file_path is None:
                continue

            # Check if the ground truth file exists
            if os.path.exists(ground_truth_file_path):
                # Create an instance of the Denoising class
                denoising = Denoising(noisy_file_path, 50, ground_truth_file_path)
                
                # Compute the Chamfer Distance (CD)
                noisy_cd, denoised_cd, no_of_iters = denoising.denoise_point_set()
                
                # Write the result to the CSV file
                with open(csv_file_path, mode='a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writerow({
                        'File Name': filename,
                        'Noise Type': noise_type,
                        'Noise Level': noise_level,
                        'Noisy_CD': noisy_cd,
                        'Denoised_CD': denoised_cd,
                        'No of Iterations' : no_of_iters
                    })
                print(f"File: {filename}, Noisy_CD: {noisy_cd}, Denoised_CD: {denoised_cd}")

print(f"Results saved to {csv_file_path}")