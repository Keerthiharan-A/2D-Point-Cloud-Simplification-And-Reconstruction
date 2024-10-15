import numpy as np
import os

def load_txt(file_path):
    """Load point set from a .xy file."""
    return np.loadtxt(file_path)

def save_points(file_path, points):
    """Saves the noisy points to a specified file."""
    np.savetxt(file_path, points, fmt='%.6f', delimiter=' ')

if __name__ == "__main__":
    input_folder = '/home/user/Documents/Minu/2D Denoising/2D-Point-Cloud-Simplification-And-Reconstruction/benchmark_results'  # Specify the input folder path
    for folder in os.listdir(input_folder):
        folder_path = os.path.join(input_folder, folder)
        xy_file = None
        points = None
        # Load the .xy file
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if file.endswith(".xy"):
                points = load_txt(file_path)
                xy_file = file_path
                break

        if points is None:
            print(f"No .xy file found in {folder_path}")
            continue

        # Load the .edg file and process it
        for file in os.listdir(folder_path):
            if file.endswith(".edg"):
                edg_path = os.path.join(folder_path, file)
                edges = load_txt(edg_path).astype(int)
                
                # Set up new file path for output
                new_file_name = f"{os.path.splitext(file)[0]}-denoised.xy"
                new_file_path = os.path.join(folder_path, new_file_name)
                
                # Initialize data and visibility list
                data = []
                vis = [False] * points.shape[0]
                
                # Collect unique points based on edges
                for edg in edges:
                    if not vis[edg[0]]:
                        data.append(points[edg[0]])
                        vis[edg[0]] = True
                    if not vis[edg[1]]:
                        data.append(points[edg[1]])
                        vis[edg[1]] = True
                
                # Save the points to the new file
                save_points(new_file_path, np.array(data))
                print(f"Processed and saved: {new_file_path}")