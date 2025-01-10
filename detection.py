import cv2
import os
import subprocess
import time

def take_photo_and_run_detection(camera_index=0, save_directory='photos', detection_script='yolov5/detect.py', weights='yolov5/my_model.pt'):
    # Initialize the camera
    cap = cv2.VideoCapture(camera_index)#, cv2.CAP_DSHOW)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1920)
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
    if not cap.isOpened():
        print("Error: Unable to open camera")
        return

    # Ensure save directory exists
    os.makedirs(save_directory, exist_ok=True)

    photo_count = 0

    while True:
        # Wait for 5 seconds before taking a photo
        time.sleep(5)

        # Capture a frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame")
            break

        # Display the captured frame
        cv2.imshow('Frame', frame)

        # Increment photo count and save the frame
        photo_count += 1
        photo_path = os.path.join(save_directory, f'photo_{photo_count}.jpg')
        cv2.imwrite(photo_path, frame)
        print(f"Photo {photo_count} saved at {photo_path}")

        # Run the detection script on the saved photo
        run_detection(detection_script, weights, photo_path)

        # Break the loop after capturing and processing the photo
        break

    # Release the camera and close any open windows
    cap.release()
    cv2.destroyAllWindows()

def run_detection(detection_script, weights, photo_path):
    detection_command = [
        'python', detection_script,
        '--weights', weights,
        '--img', '416',
        '--save-txt', '--save-crop',
        '--conf', '0.90',
        '--source', photo_path
    ]
    try:
        subprocess.run(detection_command, check=True)
        print(f"Detection completed for {photo_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error running detection: {e}")

def get_label_path(label_file_path='txt_file/label_path.txt'):
    try:
        with open(label_file_path, 'r') as file:
            return file.read().strip()
    except Exception as e:
        print(f"Error reading label path: {e}")
        return None

def calculate_origin(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read().strip()

        numbers = list(map(int, content.split()))

        if len(numbers) != 5:
            raise ValueError("File does not contain exactly 5 numbers")

        x = (numbers[1] + numbers[3]) / 2
        y = (numbers[2] + numbers[4]) / 2

        print(x, y)
        return (x, y)

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def convert_origin_for_robot(origin, image_resolution=(1920, 1080), robot_resolution=(2560, 1472)):
    try:
        x_robot = int((origin[0] * robot_resolution[0]) / image_resolution[0])
        y_robot = int((origin[1] * robot_resolution[1]) / image_resolution[1])
        print(x_robot, y_robot)
        return (x_robot, y_robot)
    except Exception as e:
        print(f"Error converting origin for robot: {e}")
        return None

def save_center_point(center_point, save_path='txt_file/center_point.txt'):
    try:
        with open(save_path, 'w') as file:
            file.write(f"{center_point}")
    except Exception as e:
        print(f"Error saving center point: {e}")

def main():
    take_photo_and_run_detection()

    label_path = get_label_path()
    if label_path:
        origin = calculate_origin(label_path)
        if origin:
            origin_for_robot = convert_origin_for_robot(origin)
            if origin_for_robot:
                save_center_point(origin_for_robot)

if __name__ == "__main__":
    main()
