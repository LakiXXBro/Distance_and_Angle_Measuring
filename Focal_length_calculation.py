import cv2
from ultralytics import YOLO
model = YOLO("runs/detect/train/weights/best.pt")

# Known parameters for calibration
KNOWN_DISTANCE = 30  # cm, distance from the camera to the Uno Card
KNOWN_WIDTH = 5.7  # cm, actual width of the UNO card

# Start video capture
cap = cv2.VideoCapture(0)

print("Press 'c' to capture an image for focal length calculation or 'x' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break
    cv2.imshow('Video Feed', frame)

    # Wait for user input to capture an image
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        # Use YOLO to detect the Uno card in the captured frame
        results = model.predict(frame, conf=0.7)

        # Extract detection information from results
        if len(results) > 0:
            detection = results[0]
            # Extract bounding box coordinates from the first detected object
            x_min, y_min, x_max, y_max = detection.boxes.xyxy[0]

            # Calculate bounding box width in pixels
            perceived_width = int(x_max - x_min)

            # Calculate the focal length
            if perceived_width > 0:
                focal_length = (KNOWN_DISTANCE * perceived_width) / KNOWN_WIDTH
                print(f"Focal Length (in pixels): {focal_length}")
            else:
                print("Error: Perceived width is zero or negative.")
        else:
            print("No object detected, please try again.")

    elif key == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()
