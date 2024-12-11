import cv2
import numpy as np
from ultralytics import YOLO
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# Load YOLOv8 model
model = YOLO("C:/Users/LakiBitz/Desktop/UnoCardDetection/runs/detect/train/weights/best.pt")

# Camera calibration results
camera_matrix = np.array([
    [1.06150525e+03, 0.00000000e+00, 9.61646225e+02],
    [0.00000000e+00, 1.06613057e+03, 5.21156498e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
])

dist_coeffs = np.array([[0, 0, 0, 0, 0]])

# Known real-world width of the target object
REAL_WIDTH = 5.7  # cm

# Correction factor to improve accuracy
CORRECTION_FACTOR = 0.6

# Define the camera's horizontal field of view
HORIZONTAL_FOV = 60  # Adjust this based on your camera's specifications

# Selected objects for tracking
selected_objects = []

# Function to start object detection
def start_detection():
    global selected_objects, canvas, frame_label

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
        tracking_display = []
        for result in results:
            if len(result.boxes) > 0:
                for box in result.boxes:
                    # Extract bounding box coordinates
                    x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()
                    class_name = model.names[int(box.cls.cpu().numpy())]

                    # Check if the detected class is in the selected objects
                    if class_name in selected_objects:
                        # Convert coordinates to integer
                        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

                        # Calculate bounding box width
                        perceived_width = x_max - x_min

                        # Calculate distance using the distance formula with a correction factor
                        if perceived_width > 0:  # Prevent division by zero
                            focal_length = (camera_matrix[0][0] + camera_matrix[1][1]) / 2  # Average of fx and fy
                            distance = (REAL_WIDTH * focal_length) / perceived_width
                            distance *= CORRECTION_FACTOR  # Apply correction factor
                            distance = round(distance, 2)

                            # Calculate the center point of the bounding box
                            object_center_x = (x_min + x_max) / 2

                            # Calculate the angle within the camera's field of view
                            frame_width = undistorted_frame.shape[1]
                            angle_from_center = ((object_center_x - frame_width / 2) / (frame_width / 2)) * (HORIZONTAL_FOV / 2)
                            angle_from_center = round(angle_from_center, 2)

                            # Draw the bounding box on the frame
                            cv2.rectangle(undistorted_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                            cv2.putText(undistorted_frame, f'{class_name}: {distance} cm', (x_min, y_min - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                            # Add tracking info for display in the UI
                            tracking_display.append(f"{class_name}: {distance} cm, {angle_from_center} deg")

        # Update tracking info in the UI
        tracking_info.set("\n".join(tracking_display))

        # Convert the frame to ImageTk format
        frame_rgb = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        frame_label.imgtk = imgtk
        frame_label.configure(image=imgtk)

        
        root.update()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

# Function to update selected objects
def update_selected_objects():
    global selected_objects
    selected_objects = []
    if red_card_var.get():
        selected_objects.append("RED_CARD_3")
    if yellow_card_var.get():
        selected_objects.append("YELLOW_CARD_0")
    if blue_card_var.get():
        selected_objects.append("BLUE_CARD_7")
    if green_card_var.get():
        selected_objects.append("GREEN_CARD_5")
    if red_card_8_var.get():
        selected_objects.append("RED_CARD_8")

# main UI window
root = tk.Tk()
root.title("Object Detection and Tracking Interface")

# Variables for checkboxes
red_card_var = tk.BooleanVar()
yellow_card_var = tk.BooleanVar()
blue_card_var = tk.BooleanVar()
green_card_var = tk.BooleanVar()
red_card_8_var = tk.BooleanVar()
tracking_info = tk.StringVar()

#UI elements
tk.Label(root, text="Select Objects to Track:").pack()

red_card_checkbox = tk.Checkbutton(root, text="RED_CARD_3", variable=red_card_var, command=update_selected_objects)
yellow_card_checkbox = tk.Checkbutton(root, text="YELLOW_CARD_0", variable=yellow_card_var, command=update_selected_objects)
blue_card_checkbox = tk.Checkbutton(root, text="BLUE_CARD_7", variable=blue_card_var, command=update_selected_objects)
green_card_checkbox = tk.Checkbutton(root, text="GREEN_CARD_5", variable=green_card_var, command=update_selected_objects)
red_card_8_checkbox = tk.Checkbutton(root, text="RED_CARD_8", variable=red_card_8_var, command=update_selected_objects)

red_card_checkbox.pack()
yellow_card_checkbox.pack()
blue_card_checkbox.pack()
green_card_checkbox.pack()
red_card_8_checkbox.pack()

start_button = tk.Button(root, text="Start Detection", command=start_detection)
start_button.pack()

frame_label = tk.Label(root)
frame_label.pack()

tracking_label = tk.Label(root, textvariable=tracking_info, font=("Helvetica", 12))
tracking_label.pack()

# Run the UI loop
root.mainloop()
