import cv2
import numpy as np

# Load the image
image_path = "/Users/vishaltala/Desktop/Thesis/Codes/Object-Detection-Using-YOLO-v5/yolov5/runs/detect/exp/photo_1.jpg"
image = cv2.imread(image_path)

# Check if image is loaded
if image is None:
    print("Error: Image not found.")
else:
    # Origin coordinates
    x, y = 622, 555

    # Calculate the end point of the first arrow (100 degrees from x-axis)
    length = 200
    angle_radians = np.radians(100)
    x_end = int(x + length * np.cos(angle_radians))
    y_end = int(y - length * np.sin(angle_radians))  # negative because y coordinates are inverted in images

    # Draw the first arrow
    cv2.arrowedLine(image, (x, y), (x_end, y_end), (0, 255, 0), thickness=2, tipLength=0.3)

    # Calculate the end point of the second arrow (perpendicular i.e., 100 + 90 = 190 degrees)
    angle_radians_perp = np.radians(190)
    x_end_perp = int(x + length * np.cos(angle_radians_perp))
    y_end_perp = int(y - length * np.sin(angle_radians_perp))

    # Draw the second arrow
    cv2.arrowedLine(image, (x, y), (x_end_perp, y_end_perp), (255, 0, 0), thickness=2, tipLength=0.3)

    # Save the modified image
    output_path = "/Users/vishaltala/Desktop/modified_photo_1.jpg"
    cv2.imwrite(output_path, image)
    print("Image saved at:", output_path)
