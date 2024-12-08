import os
import matplotlib.pyplot as plt

dataset_path = r'D:\2D-Point-Cloud-Simplification-And-Reconstruction\2D_Dataset\apple'

def load_gt_points(file_path):
    with open(file_path, 'r') as file:
        points = []
        for line in file:
            x, y = map(float, line.split())
            points.append((x, y))
        return points

### Plot - 1
fig, axes = plt.subplots(2, 4, figsize=(12, 6))  # 2 rows, 4 columns
axes = axes.flatten()
gt_file_path = os.path.join(dataset_path, 'gt.xy')
gt_points = load_gt_points(gt_file_path)

x_gt, y_gt = zip(*gt_points)

axes[0].scatter(x_gt, y_gt, color='black', s=10)
axes[0].set_title('Ground Truth')
# axes[0].set_xlabel('X')
# axes[0].set_ylabel('Y')

current_index = 1

noise_types = ['BandNoise', 'DistortedNoise']
for noise_type in noise_types:
    noise_dir = os.path.join(dataset_path, noise_type)
    
    # Collect all the BandNoise files and sort them by noise level
    if noise_type == 'BandNoise':
        noise_files = []
        for file_name in os.listdir(noise_dir):
            if file_name.endswith('.xy'):
                noise_level_str = file_name.split('-')[3].replace('.xy', '')
                noise_level = float(noise_level_str)
                radius = file_name.split('-')[2].replace('.xy', '')
                noise_files.append((file_name, noise_level, radius))
        
        # Sort BandNoise files by noise level in increasing order
        noise_files.sort(key=lambda x: (x[1], x[2]))

        # Plot 'BandNoise' in sorted order
        for file_name, noise_level, radius in noise_files:
            file_path = os.path.join(noise_dir, file_name)
            noisy_points = load_gt_points(file_path)
            
            x_noisy, y_noisy = zip(*noisy_points)
            if current_index < len(axes):
                axes[current_index].scatter(x_noisy, y_noisy, color='red', s=10)
                axes[current_index].set_title(f'{noise_type}, nl = {(noise_level)}% , r = {radius}')
                axes[current_index].set_xlabel('')
                axes[current_index].set_ylabel('')
                current_index += 1
            else:
                print("Not enough subplots for all noise levels!")
    
    # Handle DistortedNoise without sorting
    elif noise_type == 'DistortedNoise':
        for file_name in os.listdir(noise_dir):
            if file_name.endswith('.xy'):
                noise_level_str = file_name.split('-')[2].replace('.xy', '')
                noise_level = float(noise_level_str)
                file_path = os.path.join(noise_dir, file_name)
                
                noisy_points = load_gt_points(file_path)
                
                x_noisy, y_noisy = zip(*noisy_points)
                if current_index < len(axes):
                    axes[current_index].scatter(x_noisy, y_noisy, color='blue', s=10)
                    axes[current_index].set_title(f'{noise_type}, nl = {(noise_level * 100)}%')
                    axes[current_index].set_xlabel('')
                    axes[current_index].set_ylabel('')
                    current_index += 1
                else:
                    print("Not enough subplots for all noise levels!")

plt.tight_layout()
output_path = os.path.join(dataset_path, 'noise_point_sets.png')
plt.savefig(output_path)
plt.show()

print(f"Plot saved to {output_path}")

fig, axes = plt.subplots(1, 3, figsize=(12, 4))  # 1 row, 3 columns
axes = axes.flatten()

# Plot Ground Truth
gt_file_path = os.path.join(dataset_path, 'gt.xy')
gt_points = load_gt_points(gt_file_path)
x_gt, y_gt = zip(*gt_points)

axes[0].scatter(x_gt, y_gt, color='black', s=10)
axes[0].set_title('Ground Truth')
# axes[0].set_xlabel('X')
# axes[0].set_ylabel('Y')

# Plot Distorted Noise with noise level 1.5%
distorted_file_path = None
distorted_noise_level = 0.015  # 1.5%

# Find the distorted file with 1.5% noise level
noise_dir = os.path.join(dataset_path, 'DistortedNoise')
for file_name in os.listdir(noise_dir):
    if file_name.endswith(f'{distorted_noise_level}.xy'):
        distorted_file_path = os.path.join(noise_dir, file_name)
        break

if distorted_file_path:
    noisy_points = load_gt_points(distorted_file_path)
    x_noisy, y_noisy = zip(*noisy_points)
    axes[1].scatter(x_noisy, y_noisy, color='blue', s=10)
    axes[1].set_title(f'Distorted Noise - {int(distorted_noise_level * 100)}%')
    # axes[1].set_xlabel('X')
    # axes[1].set_ylabel('Y')
else:
    print("Distorted noise file with 1.5% not found.")

# Plot Band Noise with noise level 5% and radius 12.5
band_file_path = None
band_noise_level = 5  # 5%
radius = 12.5  # Radius 12.5

# Find the band file with the desired noise level and radius
noise_dir = os.path.join(dataset_path, 'BandNoise')
for file_name in os.listdir(noise_dir):
    if file_name.endswith(f'{radius}-{band_noise_level}.xy'):
        band_file_path = os.path.join(noise_dir, file_name)
        break

if band_file_path:
    noisy_points = load_gt_points(band_file_path)
    x_noisy, y_noisy = zip(*noisy_points)
    axes[2].scatter(x_noisy, y_noisy, color='red', s=10)
    axes[2].set_title(f'Band Noise - {int(band_noise_level)}% - Radius {radius}')
    # axes[2].set_xlabel('X')
    # axes[2].set_ylabel('Y')
else:
    print("Band noise file with 5% and radius 12.5 not found.")

plt.tight_layout()
output_path = os.path.join(dataset_path, 'selected_noise_point_sets.png')
plt.savefig(output_path)
plt.show()

print(f"Plot saved to {output_path}")