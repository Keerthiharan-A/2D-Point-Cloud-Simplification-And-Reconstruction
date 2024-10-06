import os
import shutil
import glob

# Paths
inputs_dir = 'inputs'
groundtruths_dir = 'groundtruths'
png_dir = 'data-sources/GroundTruthImages'
output_dir = 'dataset'

# Create dataset directory if not exists
os.makedirs(output_dir, exist_ok=True)

# Process all txt files in the inputs folder
for txt_file in glob.glob(os.path.join(inputs_dir, "*.txt")):
    # Extract object name from the file name (ignoring numbers and other separators)
    file_name = os.path.basename(txt_file)  # e.g., apple-2.png.txt
    object_name = ''.join([char for char in file_name if not char.isdigit() and char != '-' and char != '.txt' and char != '.png']).strip()
    
    # Create a folder for the object inside the dataset
    object_folder = os.path.join(output_dir, object_name)
    os.makedirs(object_folder, exist_ok=True)
    
    # Rename txt file to .xy format and move to dataset folder
    new_txt_file = os.path.join(object_folder, f"{file_name.replace('.png.txt', '')}.xy")
    shutil.copyfile(txt_file, new_txt_file)
    
    # Copy the corresponding .edg file from the groundtruths folder
    edg_file = os.path.join(groundtruths_dir, file_name + '1.edg')  # .edg file is like apple-2.png.txt1.edg
    if os.path.exists(edg_file):
        shutil.copyfile(edg_file, os.path.join(object_folder, os.path.basename(edg_file)))
    
    # Find and copy the corresponding .png file from data-sources/GroundTruthImages
    png_file = None
    for root, dirs, files in os.walk(png_dir):
        for file in files:
            if file_name.replace('.png.txt', '') in file and file.endswith('.png'):
                png_file = os.path.join(root, file)
                break
    
    if png_file:
        shutil.copyfile(png_file, os.path.join(object_folder, os.path.basename(png_file)))

print("Dataset extraction and organization complete.")