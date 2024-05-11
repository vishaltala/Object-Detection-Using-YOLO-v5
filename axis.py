import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Given points
points = np.array([
    [275, 374], [300, 314], [58, 153], [190, 117], [78, 39], [140, 44], [50, 144], [202, 135], 
    [300, 278], [130, 256], [38, 91], [253, 348], [263, 340], [73, 51], [178, 100], [137, 265], 
    [30, 114], [205, 141], [161, 73], [155, 70], [75, 180], [271, 235], [52, 79], [282, 327], 
    [80, 46], [285, 255], [54, 37], [164, 302], [226, 365], [47, 60], [59, 37], [107, 43], 
    [173, 316], [145, 51], [135, 37], [256, 347], [99, 213], [66, 166], [94, 51], [279, 245], 
    [215, 370], [254, 347], [226, 169], [231, 176], [41, 130], [60, 155], [46, 86], [68, 43], 
    [216, 155], [263, 358]
])

# Fit PCA
pca = PCA(n_components=2)
pca.fit(points)

# First principal component
first_pc = pca.components_[0]

# Center of all points (mean)
center_point = np.mean(points, axis=0)

# Plotting the points and the principal component
plt.figure(figsize=(8, 6))
plt.scatter(points[:, 0], points[:, 1], alpha=0.7)
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 3 * np.sqrt(length)  # Scale vector by 3 times the sqrt of its eigenvalue for visibility
    plt.quiver(center_point[0], center_point[1], v[0], v[1], angles='xy', scale_units='xy', scale=1, color='r')

plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('PCA on Points')
plt.axis('equal')
plt.grid(True)
plt.show()

# Print PCA components and explained variance
print("Principal components:\n", pca.components_)
print("Explained variance ratio:\n", pca.explained_variance_ratio_)
