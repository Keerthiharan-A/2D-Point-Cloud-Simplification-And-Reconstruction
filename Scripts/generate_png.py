import os
import numpy as np
import matplotlib.pyplot as plt

# Specify the directory containing .xy files
input_folder = '/home/user/Documents/Minu/2D Denoising/2D-Point-Cloud-Simplification-And-Reconstruction/2D_Dataset/apple/BandNoise'
output_folder = '/home/user/Documents/Minu/2D Denoising/2D-Point-Cloud-Simplification-And-Reconstruction/2D_Dataset/apple/Band_Images'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop through all files in the input directory
for filename in os.listdir(input_folder):
    if filename.endswith('.xy'):
        # Construct the full file path
        file_path = os.path.join(input_folder, filename)
        
        # Load the data
        data = np.loadtxt(file_path)
        x = data[:, 0]
        
        # Create the plot
        plt.figure(figsize=(50,50))
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
        ax1.scatter(data[:, 0], data[:, 1], color='blue', label='Point Set')
        # ax1.set_title('Original Points')
        # ax1.set_aspect('equal')
        # ax1.legend()

        # plt.plot(x, y, label=filename)
        # # plt.xlabel('X')
        # # plt.ylabel('Y')
        #plt.title(f'Data from {filename}')
        #plt.legend()
        plt.subplots_adjust(left=0,right=1,top=1,bottom=0)
        
        # Save the plot as a .png file in the output directory
        # plt.get_current_fig_manager().full_screen_toggle()
        png_filename = os.path.splitext(filename)[0] + '.png'
        output_path = os.path.join(output_folder, png_filename)
        plt.savefig(output_path)
        plt.close()

print("Conversion complete. PNG files are saved in:", output_folder)