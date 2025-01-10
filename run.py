import cv2

# Open a connection to the camera (0 is typically the default camera)
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open camera.")
else:
    while True:
        # Capture frame-by-frame
        ret, frame = camera.read()
        
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        # Display the resulting frame
        cv2.imshow('Camera Feed', frame)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the camera and close all OpenCV windows
camera.release()
cv2.destroyAllWindows()
