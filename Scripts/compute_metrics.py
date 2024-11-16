from Identifying_Noise import IdNoise
from scipy.spatial.distance import cdist
import numpy as np
import os
import pandas as pd

def CD(gt_path, denoised_path):
    ground_truth = IdNoise.load_xy_data(gt_path)
    point_set = IdNoise.load_xy_data(denoised_path)
    point_set = point_set[point_set[:, 0].argsort()]
    distances_AB = cdist(ground_truth, point_set, metric='euclidean')
    distances_BA = cdist(point_set, ground_truth, metric='euclidean')
    chamfer_AB = np.mean(np.min(distances_AB, axis=1))
    chamfer_BA = np.mean(np.min(distances_BA, axis=1))
    chamfer_dist = (chamfer_AB + chamfer_BA) / 2.0
    return chamfer_dist

def RMSE(gt_path, denoised_path):
    ground_truth = IdNoise.load_xy_data(gt_path)
    point_set = IdNoise.load_xy_data(denoised_path)    
    if ground_truth.shape != point_set.shape:
        return float('nan')
    point_set = point_set[point_set[:, 0].argsort()]
    mean_squared_error = np.sqrt(np.mean(np.sum((ground_truth - point_set) ** 2, axis=1)))
    return mean_squared_error

shapes = ["fish", "guitar", "hammer", "teddy"]

main_path = r"D:\2D-Point-Cloud-Simplification-And-Reconstruction\benchmark_results"
metrics = []
for shape in shapes:
    folder_path = os.path.join(main_path, shape)
    gt_path = os.path.join(folder_path, "gt.xy")
    for file in os.listdir(folder_path):
        if file == "gt.xy" or file.endswith("-0.02.xy") or file.endswith("-0.01.xy") or file.endswith("-0.005.xy") or file.endswith("-0.015.xy"):
            continue
        file_path = os.path.join(folder_path, file)
        chamfer, root_mean = CD(gt_path, file_path), RMSE(gt_path, file_path)
        metrics.append({'Name' : file, 'CD' : chamfer, 'RMSE' : root_mean})

df = pd.DataFrame(metrics)
df.to_csv("Rest_quant_metrics.csv", index=False)