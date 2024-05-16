import cv2
import os
import subprocess

def take_photos_and_run_detection(camera_index=0, save_directory='photos', detection_script='yolov5/detect.py', weights='yolov5/my_model.pt'):
    # Create a VideoCapture object
    cap = cv2.VideoCapture(camera_index)

    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Error: Unable to open camera")
        return

    # Create the save directory if it doesn't exist
    os.makedirs(save_directory, exist_ok=True)

    photo_count = 0

    # Infinite loop to capture photos
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Unable to capture frame")
            break

        # Display the captured frame
        cv2.imshow('Frame', frame)

        # Press 's' to capture a photo
        key = cv2.waitKey(1)
        if key == ord('s'):
            photo_count += 1
            # Save the captured frame
            photo_path = os.path.join(save_directory, f'photo_{photo_count}.jpg')
            cv2.imwrite(photo_path, frame)
            print(f"Photo {photo_count} saved at {photo_path}")

            # Run the detection script on the captured photo
            detection_command = [
                'python', detection_script,
                '--weights', weights,
                '--img', '416',
                '--save-txt', '--save-crop',
                '--conf', '0.90',
                '--source', photo_path
            ]
            try:
                # Call the detection script with subprocess
                subprocess.run(detection_command, check=True)
                print(f"Detection completed for {photo_path}")
            except subprocess.CalledProcessError as e:
                print(f"Error running detection: {e}")

            # Break the loop after capturing and processing the photo
            break

        # Press 'q' to quit capturing photos
        elif key == ord('q'):
            break

    # Release the VideoCapture object
    cap.release()
    cv2.destroyAllWindows()

# Call the function to capture photos and run detection
take_photos_and_run_detection(camera_index=0, save_directory='photos')
