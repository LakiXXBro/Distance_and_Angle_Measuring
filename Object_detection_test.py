from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("runs/detect/train/weights/best.pt")
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = camera.read()

    if not ret:
        print("Error: Could not read frame from camera.")
        break

    # Perform object detection on the current frame
    results = model.predict(frame, show=False, conf=0.7)
    annotated_frame = results[0].plot()
    cv2.imshow("YOLO Object Detection", annotated_frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break
camera.release()
cv2.destroyAllWindows()
