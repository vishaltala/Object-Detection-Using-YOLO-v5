import cv2
import numpy as np
from sklearn.decomposition import PCA

def draw_principal_axis(image, points):
    # Apply PCA to find the direction of maximum variance
    pca = PCA(n_components=2)
    pca.fit(points)

    # First principal component
    center = np.mean(points, axis=0)
    first_pc = pca.components_[0] * np.sqrt(pca.explained_variance_[0]) * 100  # Scale for visibility

    # Calculate the start and end points for the principal axis
    start_point = tuple(np.int32(center + first_pc))
    end_point = tuple(np.int32(center - first_pc))

    # Draw the principal axis
    cv2.arrowedLine(image, start_point, end_point, (0, 0, 255), 5)  # Red arrow line

    return image

# Load the image from specified path
image_path = '/Users/vishaltala/Desktop/Thesis/Codes/Object-Detection-Using-YOLO-v5/yolov5/runs/detect/exp/crops/Box/photo_1.jpg'
image = cv2.imread(image_path)

# Provided coordinates
points = np.array([
    [506, 542], [34, 279], [514, 539], [563, 204], [36, 137], [553, 154], [266, 43], [79, 84],
    [574, 276], [129, 573], [27, 409], [183, 55], [473, 35], [397, 565], [391, 34], [583, 396],
    [534, 81], [436, 30], [512, 540], [425, 31], [41, 357], [23, 402], [530, 69], [370, 35],
    [25, 404], [288, 579], [44, 380], [40, 127], [57, 441], [324, 575], [139, 64], [581, 457],
    [275, 42], [34, 294], [155, 581], [165, 582], [570, 249], [562, 197], [240, 46], [140, 64],
    [585, 73], [528, 66], [294, 40], [31, 192], [583, 430], [108, 72], [39, 346], [54, 437],
    [389, 34], [370, 569], [170, 58], [60, 462], [519, 47], [180, 56], [487, 547], [420, 34],
    [305, 39], [32, 421], [573, 273], [43, 366], [31, 189], [335, 37], [582, 453], [355, 35],
    [381, 34], [31, 170], [569, 242], [546, 123], [566, 224], [69, 501], [475, 550], [401, 33],
    [108, 561], [31, 164], [193, 53], [52, 423], [187, 582], [578, 316], [361, 35], [68, 92],
    [366, 570], [213, 50], [159, 60], [272, 42], [105, 559], [565, 213], [127, 67], [438, 558],
    [267, 43], [557, 175], [330, 574], [477, 35], [34, 283], [224, 582], [37, 319], [91, 546],
    [505, 38], [69, 91], [184, 583], [503, 543]
])

# Draw the principal axis on the image
result_with_axis = draw_principal_axis(image, points)

# Using OpenCV to display the image
cv2.imshow('Image with Principal Axis', result_with_axis)
cv2.waitKey(0)
cv2.destroyAllWindows()
