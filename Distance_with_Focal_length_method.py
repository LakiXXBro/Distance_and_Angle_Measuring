import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("C:/Users/LakiBitz/Desktop/UnoCardDetection/runs/detect/train/weights/best.pt")

# Capture video from the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Use YOLO to detect objects in the frame
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

                # Draw bounding box and label
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, class_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the video feed with detections
    cv2.imshow('YOLOv8 Object Detection', frame)

    # Press 'x' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
