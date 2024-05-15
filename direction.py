import cv2
import numpy as np

def draw_arrows(image, origin, length1, angle1, length2, angle2):
    # Convert angles from degrees to radians
    angle1_rad = np.deg2rad(angle1)
    angle2_rad = np.deg2rad(angle2)
    
    # Calculate the end points for each arrow
    end1 = (int(origin[0] + length1 * np.cos(angle1_rad)), int(origin[1] - length1 * np.sin(angle1_rad)))
    end2 = (int(origin[0] + length2 * np.cos(angle2_rad)), int(origin[1] - length2 * np.sin(angle2_rad)))
    
    # Draw the arrows
    cv2.arrowedLine(image, origin, end1, (255, 0, 0), 3)  # Blue arrow
    cv2.arrowedLine(image, origin, end2, (0, 255, 0), 3)  # Green arrow
    
    return image

# Load an existing image
image_path = 'path_to_your_image.jpg'  # Replace with your image path
image = cv2.imread(image_path)

# If the image is not found, raise an error
if image is None:
    raise FileNotFoundError("The specified image path does not exist or the image could not be loaded.")

# Define parameters
origin = (250, 250)  # Modify as needed
length1 = 150
angle1 = 45  # Degrees
length2 = 100
angle2 = 120  # Degrees

# Draw the arrows on the image
image_with_arrows = draw_arrows(image, origin, length1, angle1, length2, angle2)

# Save the image to a desired path
save_path = 'path_to_save_image.jpg'  # Replace with your desired save path
cv2.imwrite(save_path, image_with_arrows)

print(f"Image saved successfully at {save_path}")
