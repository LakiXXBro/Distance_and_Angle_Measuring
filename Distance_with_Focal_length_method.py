import cv2
from ultralytics import YOLO 

# Load YOLOv8 model
model = YOLO("C:/Users/LakiBitz/Desktop/UnoUno/pythonProject2/runs/detect/train28/weights/best.pt")

# Known real-world width of the target object
REAL_WIDTH = 9  # cm

# Correction factor to improve accuracy
CORRECTION_FACTOR = 1

# Capture video from the webcam
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Use YOLO to detect objects in the raw frame
    results = model.predict(source=frame, save=False, conf=0.25)

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
                    focal_length = 535  # Estimated focal length, adjust based on your camera
                    distance = (REAL_WIDTH * focal_length) / perceived_width
                    distance *= CORRECTION_FACTOR  # Apply correction factor
                    distance = round(distance, 2)

                    # Display the bounding box and distance on the frame
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(frame, f'{class_name}: {distance} cm', (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the raw frame with the detection and distance estimation
    cv2.imshow('YOLOv8 Object Detection with Distance Estimation', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
