import cv2
import numpy as np
import math
from scipy.spatial.transform import Rotation as R

def draw_arrows(image, origin, length1, angle1, length2, angle2):
    
    # Convert angles from degrees to radians
    angle1_rad = np.deg2rad(angle1)
    angle2_rad = np.deg2rad(angle2)
    
    # Step 1: Convert RPY angles from degrees to radians
    rpy_degrees = [180, 0, angle1]
    rpy_radians = np.deg2rad(rpy_degrees)

    # Step 2: Convert RPY (roll, pitch, yaw) to a rotation matrix
    rotation_matrix = R.from_euler('xyz', rpy_radians).as_matrix()

    # Step 3: Convert the rotation matrix to a rotational vector (axis-angle representation)
    rot_vec = R.from_matrix(rotation_matrix).as_rotvec()

    # Step 4: Round the rotational vector to four decimal places
    rot_vec_rounded = np.round(rot_vec, 3)

    # Calculate the end points for each arrow
    end1 = (int(origin[0] + length1 * np.cos(angle1_rad)), int(origin[1] - length1 * np.sin(angle1_rad)))
    end2 = (int(origin[0] + length2 * np.cos(angle2_rad)), int(origin[1] - length2 * np.sin(angle2_rad)))

    origin_for_robot = (int((origin[0]*2560)/1920), int((origin[1]*1472)/1080))
    
    with open("txt_file/direction.txt", "w") as file:
        file.write(f"Origin for Robot: {origin_for_robot}\n")
        file.write(f"Origin: {origin}\n")
        #file.write(f"Ends 1: {end1}\n")
        #file.write(f"Ends 2: {end2}\n")
        #file.write(f"Length 1: {length1}\n")
        #file.write(f"Length 2: {length2}\n")    
        file.write(f"Angle 1: {angle1}\n")
        file.write(f"Angle 2: {angle2}\n")
        file.write(f"Rotational Vector in Rad: {rot_vec_rounded}\n")

    # Convert origin to integer
    origin = (int(origin[0]), int(origin[1]))

    # Draw the arrows
    cv2.arrowedLine(image, origin, end1, (0, 0, 255), 3)  # Red arrow
    cv2.arrowedLine(image, origin, end2, (0, 255, 0), 3)  # Green arrow
    
    return image

# Load all Image Path
def image_path():
    with open("/Users/vishaltala/Desktop/Thesis/Codes/Object-Detection-Using-YOLO-v5/txt_file/detect_img_path.txt", "r") as file:
        return file.read()

# Load an existing image
image_path = image_path()
image = cv2.imread(image_path)

# If the image is not found, raise an error
if image is None:
    raise FileNotFoundError("The specified image path does not exist or the image could not be loaded.")

# Calculate centre of detected Object
def label_path():
    with open("/Users/vishaltala/Desktop/Thesis/Codes/Object-Detection-Using-YOLO-v5/txt_file/label_path.txt", "r") as file:
        return file.read()
    
# Function to read the file and calculate the origin
def calculate_origin(file_path):
    try:
        # Open the file and read the content
        with open(file_path, 'r') as file:
            content = file.read().strip()
        
        # Split the content into a list of numbers
        numbers = list(map(int, content.split()))
        
        if len(numbers) != 5:
            raise ValueError("File does not contain exactly 5 numbers")

        # Calculate x and y
        x = (numbers[1] + numbers[3]) / 2
        y = (numbers[2] + numbers[4]) / 2

        # Create the origin tuple
        origin = (x, y)
        
        return origin

    except Exception as e:
        print(f"An error occurred: {e}")

# Calculate the origin using the function
origin = calculate_origin(label_path())

def calculate_vectors(file_path):
    try:
        # Open the file and read the content
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        # Process each line to extract vectors
        vectors = []
        for line in lines:
            # Remove any leading/trailing whitespace and brackets, then split into numbers
            line = line.strip().strip('[]')
            if line:
                vector = line.split()  # Split the line by whitespace
                if len(vector) == 2:
                    vectors.append(list(map(float, vector)))

        if len(vectors) != 2:
            raise ValueError("File does not contain exactly 2 vectors")

        # Calculate lengths and angles for both vectors
        x1, y1 = vectors[0]
        x2, y2 = vectors[1]

        length1 = math.sqrt(x1**2 + y1**2)
        angle1 = math.degrees(math.atan2(y1, x1))


        length2 = math.sqrt(x2**2 + y2**2)
        angle2 = math.degrees(math.atan2(y2, x2))

        return length1, angle1, length2, angle2

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None, None

# Calculate lengths and angles using the function
path = '/Users/vishaltala/Desktop/Thesis/Codes/Object-Detection-Using-YOLO-v5/txt_file/vectors.txt'
length1, angle1, length2, angle2 = calculate_vectors(path)

# Draw the arrows on the image
image_with_arrows = draw_arrows(image, origin, length1, angle1, length2, angle2)

# Define Path for saving the file
parts = image_path.split('/')
border_parts = parts[:-1]
border_parts.append('direction_1.jpg')
output_path = '/'.join(border_parts)

# Save the image to a desired path
cv2.imwrite(output_path, image_with_arrows)

print(f"Image with detected direction saved successfully at {output_path}")
