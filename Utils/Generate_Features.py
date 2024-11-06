import os
import pandas as pd
from Create_features import Features

def label_file(file_name):
    """Label the file based on its name."""
    if any(file_name.endswith(suffix + '.xy') for suffix in ['-7.5-2', '-10-2', '-12.5-5']):
        return 'Band'
    elif any(file_name.endswith(suffix + '.xy') for suffix in ['-0.01', '-0.015', '-0.02', '-0.005','-0.003','-0.01','-0.03']):
        return 'Distorted'
    else:
        return 'Clean'

def process_xy_files(directory):
    """Process all .xy files in the specified directory and its subdirectories."""
    feature_list = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.xy'):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")

                try:
                    feature_extractor = Features(file_path)
                    features = feature_extractor.get_features()
                    
                    label = label_file(file)
                    
                    feature_list.append({
                        'file_name': file,
                        'num_points': features[0],
                        'frac_flower_points': features[1],
                        'more5_neighbor_flower': features[2],
                        'more5_neighbor_ovr': features[3],
                        'mean_closest_distance': features[4],
                        'std_closest_distance': features[5],
                        'mean_count': features[6],
                        'std_count': features[7],
                        'label': label,
                    })
                except ValueError as e:
                    print(e)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    features_df = pd.DataFrame(feature_list)
    features_df.drop_duplicates(inplace=True)

    features_df.to_csv('features_output.csv', index=False)
    print("Features saved to features_output.csv")

if __name__ == "__main__":
    root_directory = r"/home/user/Documents/Minu/2D Denoising/2D-Point-Cloud-Simplification-And-Reconstruction/Feature_data"
    process_xy_files(root_directory)