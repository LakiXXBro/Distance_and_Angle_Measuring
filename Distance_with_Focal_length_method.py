import cv2
import numpy as np
from ultralytics import YOLO  # Import YOLOv8 class from ultralytics library
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# Load YOLOv8 model
model = YOLO("C:/Users/LakiBitz/Desktop/UnoCardDetection/runs/detect/train/weights/best.pt")  # Adjust the path to your model weights

# Camera calibration results (replace with your own values from the calibration)
camera_matrix = np.array([
    [1.06150525e+03, 0.00000000e+00, 9.61646225e+02],
    [0.00000000e+00, 1.06613057e+03, 5.21156498e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
])

dist_coeffs = np.array([[0, 0, 0, 0, 0]])

# Known real-world widths of the target objects
REAL_WIDTHS = {
    "RED_CARD_3": 5.7,
    "YELLOW_CARD_0": 6.7,
    "BLUE_CARD_7": 7.7,
    "GREEN_CARD_5": 8.7,
    "RED_CARD_8": 9.7,
    "RED_CARD_6": 5.7  # Width for verification card
}

# Correction factor to improve accuracy (based on empirical testing)
CORRECTION_FACTOR = 0.6

# Define the camera's horizontal field of view (in degrees)
HORIZONTAL_FOV = 60  # Adjust this based on your camera's specifications

# Selected objects for tracking
selected_objects = []

# Global variable for verification status
verification_passed = False

# Function to process the camera feed for verification
def verification_with_camera():
    global verification_passed

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Undistort the frame using the camera matrix and distortion coefficients
        undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs)

        # Use YOLO to detect objects in the undistorted frame
        results = model.predict(source=undistorted_frame, save=False, conf=0.25)

        for result in results:
            if len(result.boxes) > 0:
                for box in result.boxes:
                    # Extract bounding box coordinates
                    x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()
                    class_name = model.names[int(box.cls.cpu().numpy())]

                    # Check if the detected class is the verification card
                    if class_name == "RED_CARD_6":
                        verification_passed = True
                        cap.release()
                        show_main_page()
                        return

        # Convert the frame to ImageTk format
        frame_rgb = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        frame_label.imgtk = imgtk
        frame_label.configure(image=imgtk)

        # Handle the UI update loop
        root.update_idletasks()
        root.update()

    cap.release()

# Function to verify using a password
def verification_with_password():
    def check_password():
        if password_entry.get() == "mdx2024":
            show_main_page()
        else:
            error_label.config(text="Incorrect password. Try again.")

    for widget in root.winfo_children():
        widget.destroy()

    tk.Label(root, text="Enter Password for Verification", font=("Helvetica", 16)).pack(pady=10)
    password_entry = tk.Entry(root, show="*", font=("Helvetica", 14))
    password_entry.pack(pady=10)
    tk.Button(root, text="Verify", command=check_password, font=("Helvetica", 14)).pack(pady=10)
    error_label = tk.Label(root, text="", font=("Helvetica", 12), fg="red")
    error_label.pack()

# Function to process the camera feed and display it in the main detection page
def process_camera_feed():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Undistort the frame using the camera matrix and distortion coefficients
        undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs)

        # Convert the frame to ImageTk format
        frame_rgb = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        frame_label.imgtk = imgtk
        frame_label.configure(image=imgtk)

        # Handle the UI update loop
        root.update_idletasks()
        root.update()

    cap.release()

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

                        # Calculate bounding box width (in pixels)
                        perceived_width = x_max - x_min

                        # Calculate distance using the distance formula with a correction factor
                        if perceived_width > 0 and class_name in REAL_WIDTHS:  # Prevent division by zero and check for valid class
                            focal_length = (camera_matrix[0][0] + camera_matrix[1][1]) / 2  # Average of fx and fy
                            real_width = REAL_WIDTHS[class_name]  # Get the real width for the detected class
                            distance = (real_width * focal_length) / perceived_width
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
                            tracking_display.append(f"{class_name}: {distance:6.2f} cm, {angle_from_center:6.2f} deg")

        # Ensure consistent spacing by padding the tracking info
        padded_display = [f"{line:<40}" for line in tracking_display]
        tracking_info.set("\n".join(padded_display))

        # Convert the frame to ImageTk format
        frame_rgb = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        frame_label.imgtk = imgtk
        frame_label.configure(image=imgtk)

        # Handle the UI update loop
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

# Function to show the main page after verification
def show_main_page():
    for widget in root.winfo_children():
        widget.destroy()

    # Create a frame for the UI layout
    main_frame = tk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True)

    # Left frame for checkboxes and tracking info
    left_frame = tk.Frame(main_frame, width=300)
    left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

    # Add a gap between the left and right frames
    spacer = tk.Frame(main_frame, width=20)
    spacer.pack(side=tk.LEFT)

    # Right frame for the camera feed
    right_frame = tk.Frame(main_frame, width=900, height=600)
    right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

    # Variables for checkboxes
    global red_card_var, yellow_card_var, blue_card_var, green_card_var, red_card_8_var, tracking_info
    red_card_var = tk.BooleanVar()
    yellow_card_var = tk.BooleanVar()
    blue_card_var = tk.BooleanVar()
    green_card_var = tk.BooleanVar()
    red_card_8_var = tk.BooleanVar()
    tracking_info = tk.StringVar()

    # Create UI elements in the left frame
    tk.Label(left_frame, text="Select Objects to Track:").pack(anchor=tk.W)

    red_card_checkbox = tk.Checkbutton(left_frame, text="RED_CARD_3", variable=red_card_var, command=update_selected_objects)
    yellow_card_checkbox = tk.Checkbutton(left_frame, text="YELLOW_CARD_0", variable=yellow_card_var, command=update_selected_objects)
    blue_card_checkbox = tk.Checkbutton(left_frame, text="BLUE_CARD_7", variable=blue_card_var, command=update_selected_objects)
    green_card_checkbox = tk.Checkbutton(left_frame, text="GREEN_CARD_5", variable=green_card_var, command=update_selected_objects)
    red_card_8_checkbox = tk.Checkbutton(left_frame, text="RED_CARD_8", variable=red_card_8_var, command=update_selected_objects)

    red_card_checkbox.pack(anchor=tk.W)
    yellow_card_checkbox.pack(anchor=tk.W)
    blue_card_checkbox.pack(anchor=tk.W)
    green_card_checkbox.pack(anchor=tk.W)
    red_card_8_checkbox.pack(anchor=tk.W)

    start_button = tk.Button(left_frame, text="Start Detection", command=start_detection)
    start_button.pack(pady=10)

    tracking_label = tk.Label(left_frame, textvariable=tracking_info, font=("Helvetica", 12), justify=tk.LEFT, width=40, anchor=tk.W)
    tracking_label.pack(anchor=tk.W)

    # Camera feed display in the right frame
    global frame_label
    frame_label = tk.Label(right_frame, width=900, height=600)
    frame_label.pack(fill=tk.BOTH, expand=True)

    # Start the camera feed in the background
    root.after(0, process_camera_feed)

# Initial verification page
root = tk.Tk()
root.title("Object Verification and Detection Interface")
root.geometry("1200x600")  # Set fixed size of the window

# Verification interface
tk.Label(root, text="Verification Required", font=("Helvetica", 18)).pack(pady=20)
tk.Label(root, text="Please verify using one of the following methods:", font=("Helvetica", 14)).pack(pady=10)
tk.Button(root, text="Verify with Camera", command=verification_with_camera, font=("Helvetica", 14)).pack(pady=10)
tk.Button(root, text="Verify with Password", command=verification_with_password, font=("Helvetica", 14)).pack(pady=10)

frame_label = tk.Label(root)
frame_label.pack(fill=tk.BOTH, expand=True)

# Run the UI loop
root.mainloop()
