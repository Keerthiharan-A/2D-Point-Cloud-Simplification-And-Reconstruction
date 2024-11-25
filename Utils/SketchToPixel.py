import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the sketch image
img = cv2.imread('/home/user/Documents/Minu/test_prgms/2D_visualization/bear_sketch_less.jpg', cv2.IMREAD_GRAYSCALE)

# Find all non-white pixels (assuming white is 255)
point_set = np.column_stack(np.where(img < 255))

# Visualize the point set
# plt.scatter(point_set[:, 1], point_set[:, 0], s=1, color='black')
# plt.gca().invert_yaxis()  # Match image coordinate system
# plt.title('Point Set from Sketch')
# plt.show()

# Save the point set to a file
np.savetxt('bear_points.xy', point_set, fmt='%d')
