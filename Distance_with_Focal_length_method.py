import cv2
import numpy as np
from ultralytics import YOLO  

# Load YOLOv8 model
model = YOLO("C:/Users/LakiBitz/Desktop/UnoCardDetection/runs/detect/train/weights/best.pt")

# Camera calibration results
camera_matrix = np.array([
    [1.06150525e+03, 0.00000000e+00, 9.61646225e+02],
    [0.00000000e+00, 1.06613057e+03, 5.21156498e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
])

dist_coeffs = np.array([[0.14101295, -0.30574447, 0.00225758, 0.0032847, 0.06745202]])

# Known real-world width of the target object
REAL_WIDTH = 5.7  # cm

# Correction factor to improve accuracy
CORRECTION_FACTOR = 0.6

# Define the camera's horizontal field of view
HORIZONTAL_FOV = 60 

# Capture video from the webcam
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Undistort the frame using the camera matrix and distortion coefficients
    undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs)

    # Use YOLO to detect objects in the undistorted frame
    results = model.predict(source=undistorted_frame, save=False, conf=0.25)

    # Parse the results and draw bounding boxes
    for result in results:
        if len(result.boxes) > 0:
            for box in result.boxes:
                # Extract bounding box coordinates
                x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()
                class_name = model.names[int(box.cls.cpu().numpy())]

                # Convert coordinates to integer
                x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

                # Calculate bounding box width (in pixels)
                perceived_width = x_max - x_min

                # Calculate distance using the distance formula with a correction factor
                if perceived_width > 0:  # Prevent division by zero
                    focal_length = (camera_matrix[0][0] + camera_matrix[1][1]) / 2  # Average of fx and fy
                    distance = (REAL_WIDTH * focal_length) / perceived_width
                    distance *= CORRECTION_FACTOR  # Apply correction factor
                    distance = round(distance, 1)

                    # Calculate the center point of the bounding box
                    object_center_x = (x_min + x_max) / 2

                    # Calculate the angle within the camera's field of view
                    frame_width = undistorted_frame.shape[1]
                    angle_from_center = ((object_center_x - frame_width / 2) / (frame_width / 2)) * (HORIZONTAL_FOV / 2)
                    angle_from_center = round(angle_from_center*2, 2)

                    # Display the bounding box, distance, and angle on the frame
                    cv2.rectangle(undistorted_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(undistorted_frame, f'{class_name}: {distance} cm, {angle_from_center} deg', (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the undistorted frame with the detection, distance estimation, and angle
    cv2.imshow('YOLOv8 Object Detection with Distance and Angle Estimation', undistorted_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
