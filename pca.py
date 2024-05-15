import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def apply_watershed(image, margin=10):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply binary thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Noise removal using morphological opening
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Sure background area using dilation
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Finding sure foreground area using distance transform
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0
    
    # Apply watershed
    markers = cv2.watershed(image, markers)
    
    # Create a mask to ignore boundaries near the edges
    height, width = image.shape[:2]
    
    # Drawing only valid boundaries away from the edge
    for y in range(margin, height - margin):
        for x in range(margin, width - margin):
            if markers[y, x] == -1:
                cv2.circle(image, (x, y), 2, [0, 255, 0], -1)  # Mark boundary in green, change 2 to desired thickness
    
    # Extracting border points, excluding edges near the image boundary
    height, width = image.shape[:2]
    margin = 10  # Margin to exclude near the image edges
    border_points = np.column_stack(np.where(markers == -1))
    filtered_border_points = [(pt[1], height - pt[0] - 1) for pt in border_points if margin < pt[0] < height - margin and margin < pt[1] < width - margin]
    
    return image, filtered_border_points

# Load the image from your specified path
def read_path():
    with open("file_path.txt", "r") as file:
        return file.read()
    
image_path = read_path()
image = cv2.imread(image_path)

# Apply watershed and get border points
result, border_points = apply_watershed(image)

# Define Path for saving the file
parts = image_path.split('/')
border_parts = parts[:-3]
border_parts.append('border_1.jpg')
output_path = '/'.join(border_parts)

cv2.imwrite(output_path, result)
print("Image saved at:", output_path)

# Swap the x and y coordinates using slicing
points = np.array(border_points)

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
    v = vector * np.sqrt(length)  # Scale vector with the sqrt of its eigenvalue for visibility
    print(v)
    plt.quiver(center_point[0], center_point[1], v[0], v[1], angles='xy', scale_units='xy', scale=1, color='r')

plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('PCA')
plt.axis('equal')
plt.grid(True)

pca_parts = parts[:-3]
pca_parts.append('pca_1.jpg')
output_path = '/'.join(pca_parts)

# Save the figure to a file and close the plot
plt.savefig(output_path, format='jpg', dpi=300)
plt.close()

# Delete the file after reading
#os.remove("file_path.txt")