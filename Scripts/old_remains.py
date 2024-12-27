# Save the final denoised points after all iterations
            #denoised_file_path = os.path.join('Denoised_output', self.file_path.replace('.xy', f'_denoised_2iters.xy'))
            #os.makedirs(os.path.dirname(denoised_file_path), exist_ok=True)
            #self.save_to_xy_file(self.scaled_point_set, denoised_file_path)
            #computing the flower points again and doing MLS only on those points

# elif noise_type == "Distorted":
#             print("distorted noise")
#             for iteration in range(self.iterations):
#                 denoised_points = []
#                 for point_idx in range(len(self.point_set)):
#                     #denoised_point = self.weighted_least_squares_and_projection(point_idx)
#                     denoised_point = self.wls_with_normal(point_idx)
#                     denoised_points.append(denoised_point)

#                 #print(f"Average Mean squared error {error/len(self.scaled_point_set)}, for iteration {iteration + 1}")
#                 print(f"Iteration {iteration+1} completed.")
#                 self.scaled_point_set = np.array(denoised_points)
#                 #self.chamfer_distance()
#                 # print(f"Iteration {iteration + 1} completed")
            
            
#             denoised_points_iter = PointSet(self.scaled_point_set)
#             flower_points_set = denoised_points_iter.flower_points

#             denoised_points = self.scaled_point_set
#             # Loop over all points in the point set
#             for idx, point in enumerate(self.scaled_point_set):
#                  if idx in flower_points_set:
#                     for idx1, _ in denoised_points_iter.neighbors[idx]:
#             #         # Apply denoising method to flower points
#             #             #denoised_points[idx1] = self.weighted_least_squares_and_projection(idx1)
#                          denoised_points[idx1] = self.wls_with_normal(idx1)

#             self.scaled_point_set = np.array(denoised_points) 
#             denoised_file_path = self.file_path.replace('.xy', '_flower_denoised.xy')
#             os.makedirs(os.path.dirname(denoised_file_path), exist_ok=True)
#             self.save_to_xy_file(self.scaled_point_set, denoised_file_path)
#             app = DualPointVisualizerApp(self.file_path, denoised_file_path)
#             app.open_windows()  
#             #self.chamfer_distance()
#             #self.rmse()
# denoised_file_path1 =  self.file_path.replace('.xy', f'_denoised_.xy')
            #self.save_to_xy_file(self.point_set, denoised_file_path1)