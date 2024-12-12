# Import necessary libraries
import cv2  # OpenCV library for image processing and computer vision
import numpy as np  # For numerical operations and matrix manipulations
from ultralytics import YOLO  # For YOLO object detection model
import tkinter as tk  # For GUI development
from tkinter import ttk  # For themed widgets in Tkinter
from PIL import Image, ImageTk  # For image handling in Tkinter
from tkinter import messagebox  # For displaying message boxes

# Load the YOLO model with the specified weight file
model = YOLO("C:/Users/LakiBitz/Desktop/UnoUno/pythonProject2/runs/detect/Iphone_Box_Dataset/weights/best.pt")

# Camera calibration parameters
CAMERA_MATRIX = np.array([[1.63697859e+03, 0.00000000e+00, 9.50278068e+02],
                          [0.00000000e+00, 1.63623452e+03, 5.21825081e+02],
                          [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])  # Intrinsic parameters of the camera
DISTORTION_COEFFS = np.array([[4.31059459e-01, -2.37940632e+00, -6.04646635e-04, -1.24375097e-03, 3.48162996e+00]])  # Distortion coefficients

# Real-world object widths (in cm) for specific classes
CLASS_WIDTHS = {
    "Iphone_X": 8.9,
    "Iphone_6": 8.6,
    "Iphone_SE": 8.0,
    "Iphone_15": 9.0,
    "Iphone_13": 8.95,
}
CORRECTION_FACTOR = 1  # Correction factor for distance calculation (used only if required)

# My Camera's horizontal field of view (in degrees)
HORIZONTAL_FOV = 70
selected_classes = set()  # Set of classes selected for detection
is_detecting = False  # Flag to indicate if detection is active

# Function to verify access using my ID card
def verify_id_card():
    cap = cv2.VideoCapture(0)

    def check_for_id_card():
        ret, frame = cap.read()  # Capture a frame from the webcam
        if not ret:  # If no frame is captured, return
            return

        # Run the YOLO model on the captured frame
        results = model.predict(source=frame, save=False, conf=0.70)

        # Check if the ID card class is detected
        for result in results:
            if len(result.boxes) > 0:
                for box in result.boxes:
                    class_name = model.names[int(box.cls.cpu().numpy())]
                    if class_name == "ACCESS_ID":  # If ID card is detected
                        cap.release()  # Release the camera
                        cv2.destroyAllWindows()  # Close all OpenCV windows
                        switch_to_main_program()  # Proceed to the main program
                        return

        # Check again after 100 ms
        root.after(100, check_for_id_card)

    check_for_id_card()

# Function to verify access using a password
def verify_password():
    entered_password = password_entry.get()  # Get the entered password
    if entered_password == "mdx2024":  # Check if the password is correct
        switch_to_main_program()  # Proceed to the main program
    else:
        messagebox.showerror("Error", "Incorrect password. Please try again.")  # Show error message

# Function to switch to the main program GUI
def switch_to_main_program():
    verification_frame.pack_forget()  # Hide the verification frame
    main_program_frame.pack(fill="both", expand=True)  # Show the main program frame

# Function to update the set of selected classes for detection
def update_selected_classes():
    global selected_classes
    selected_classes.clear()  # Clear the set
    for class_name, var in class_checkboxes.items():
        if var.get():  # Add selected classes to the set
            selected_classes.add(class_name)

# Function to start object detection
def start_detection():
    global is_detecting
    is_detecting = True  # Set detection flag
    cap = cv2.VideoCapture(0)  # Access the webcam

    def update_frame():
        if not is_detecting:
            cap.release()  # Release the camera if detection stops
            return

        ret, frame = cap.read()  # Capture a frame from the webcam
        if not ret:
            return

        # Undistort the captured frame
        h, w = frame.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            CAMERA_MATRIX, DISTORTION_COEFFS, (w, h), 1, (w, h)
        )
        mapx, mapy = cv2.initUndistortRectifyMap(
            CAMERA_MATRIX, DISTORTION_COEFFS, None, new_camera_matrix, (w, h), 5
        )
        undistorted_frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

        # Use YOLO to detect objects in the frame
        results = model.predict(source=undistorted_frame, save=False, conf=0.70)

        detection_details = ""  # Initialize detection details text

        for result in results:
            if len(result.boxes) > 0:
                for box in result.boxes:
                    x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()  # Bounding box coordinates
                    class_name = model.names[int(box.cls.cpu().numpy())]

                    if class_name not in selected_classes:  # Ignore unselected classes
                        continue

                    real_width = CLASS_WIDTHS.get(class_name, 8.0)  # Get the real width of the object
                    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
                    perceived_width = x_max - x_min

                    if perceived_width > 0:  # Calculate distance if perceived width is valid
                        focal_length = 480  # focal length
                        distance = (real_width * focal_length) / perceived_width  # Calculate distance
                        distance *= CORRECTION_FACTOR  # Apply correction factor if required
                        distance = round(distance, 2)

                        # Calculate angle from center of the frame
                        object_center_x = (x_min + x_max) / 2
                        frame_width = undistorted_frame.shape[1]
                        angle_from_center = (
                            (object_center_x - frame_width / 2)
                            / (frame_width / 2)
                        ) * (HORIZONTAL_FOV / 2)
                        angle_from_center = round(angle_from_center, 2)

                        # Draw bounding box and label on the frame
                        cv2.rectangle(
                            undistorted_frame,
                            (x_min, y_min),
                            (x_max, y_max),
                            (0, 255, 0),
                            2,
                        )
                        cv2.putText(
                            undistorted_frame,
                            f"{class_name}",
                            (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2,
                        )

                        # Append detection details
                        detection_details += f"{class_name} -> Distance: {distance} cm, Angle: {angle_from_center}°\n"

        # Update the detection details label
        details_label.config(text=detection_details.strip())

        # Convert the frame to Tkinter-compatible format
        rgb_frame = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        img_tk = ImageTk.PhotoImage(image=img)
        video_label.img_tk = img_tk
        video_label.configure(image=img_tk)
        root.after(10, update_frame)  # Update the frame every 10 ms

    update_frame()

# Function to stop object detection
def stop_detection():
    global is_detecting
    is_detecting = False  # Set detection flag to false
    details_label.config(text="")  # Clear detection details




#************************************************ GUI Setup ********************************************************************************
# Main application window
root = tk.Tk()
root.title("Object Detection Program")
root.geometry("1000x700")

# Verification frame (initial screen)
verification_frame = tk.Frame(root)
verification_frame.pack(fill="both", expand=True)

# Welcome and verification instructions
intro_text = (
    "Welcome to the Object Detection Program\n"
    "Please verify your access to continue."
)
intro_label = tk.Label(
    verification_frame,
    text=intro_text,
    font=("Arial", 18, "bold"),
    fg="blue",
    justify="center",
)
intro_label.pack(pady=20)

# Access verification options
tk.Label(
    verification_frame,
    text="Access Verification",
    font=("Arial", 16),
).pack(pady=10)
tk.Label(
    verification_frame,
    text="Choose one of the following options to verify access:",
    font=("Arial", 12),
).pack(pady=5)

# ID card verification button
tk.Button(
    verification_frame,
    text="Verify with ID Card",
    font=("Arial", 12),
    command=verify_id_card,
).pack(pady=10)

# Password verification
tk.Label(
    verification_frame, text="Or enter the password:", font=("Arial", 12)
).pack(pady=5)
password_entry = tk.Entry(
    verification_frame, show="*", font=("Arial", 12)
)
password_entry.pack(pady=5)
tk.Button(
    verification_frame,
    text="Verify with Password",
    font=("Arial", 12),
    command=verify_password,
).pack(pady=10)

# Main program frame (detection screen)
main_program_frame = tk.Frame(root)

# Instructions for detection
instruction_label = tk.Label(
    main_program_frame,
    text="Select the objects you want to detect and estimate the distance and angle.",
    font=("Arial", 14),
    bg="lightblue",
    fg="black",
)
instruction_label.pack(fill="x", pady=5)

# Left frame for class selection
left_frame = tk.Frame(
    main_program_frame, width=300, height=600, bg="lightgray"
)
left_frame.pack(side="left", fill="y")

# Right frame for video display and detection details
right_frame = tk.Frame(main_program_frame, width=700, height=600)
right_frame.pack(side="right", fill="both", expand=True)

# Checkboxes for class selection
class_checkboxes = {}
for class_name in CLASS_WIDTHS.keys():
    var = tk.BooleanVar(value=False)
    checkbox = ttk.Checkbutton(
        left_frame,
        text=class_name,
        variable=var,
        command=update_selected_classes,
    )
    checkbox.pack(anchor="w", padx=10, pady=5)
    class_checkboxes[class_name] = var

# Start detection button
start_button = ttk.Button(
    left_frame, text="Start Detection", command=start_detection
)
start_button.pack(pady=10)

# Stop detection button
stop_button = ttk.Button(
    left_frame, text="Stop Detection", command=stop_detection
)
stop_button.pack(pady=10)

# Video display area
video_label = tk.Label(right_frame)
video_label.pack(fill="both", expand=True)

# Detection details label
details_label = tk.Label(
    right_frame,
    text="",
    justify="left",
    bg="white",
    anchor="nw",
    font=("Arial", 12),
)
details_label.pack(fill="x", pady=10)
details_label.config(height=6)

# Initialize selected classes
update_selected_classes()

# Run the GUI event loop
root.mainloop()
