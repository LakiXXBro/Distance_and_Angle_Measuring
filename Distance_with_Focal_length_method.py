import cv2
import numpy as np
from ultralytics import YOLO
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from tkinter import messagebox

model = YOLO("C:/Users/LakiBitz/Desktop/UnoUno/pythonProject2/runs/detect/train27/weights/best.pt") 

# Camera calibration parameters
CAMERA_MATRIX = np.array([[1.63697859e+03, 0.00000000e+00, 9.50278068e+02],
                          [0.00000000e+00, 1.63623452e+03, 5.21825081e+02],
                          [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
DISTORTION_COEFFS = np.array([[4.31059459e-01, -2.37940632e+00, -6.04646635e-04, -1.24375097e-03, 3.48162996e+00]])

CLASS_WIDTHS = {
    "Iphone_X": 8.9,
    "Iphone_6": 8.6,
    "Iphone_SE": 8.0,
    "Iphone_15": 9.0,
    "Iphone_13": 8.95,
}
CORRECTION_FACTOR = 1

# Define the camera's horizontal field of view (in degrees)
HORIZONTAL_FOV = 70
selected_classes = set()
is_detecting = False


def verify_id_card():
    cap = cv2.VideoCapture(0)

    def check_for_id_card():
        ret, frame = cap.read()
        if not ret:
            return

        results = model.predict(source=frame, save=False, conf=0.25)

        for result in results:
            if len(result.boxes) > 0:
                for box in result.boxes:
                    class_name = model.names[int(box.cls.cpu().numpy())]
                    if class_name == "ACCESS_ID":
                        cap.release()
                        cv2.destroyAllWindows()
                        switch_to_main_program()
                        return

        root.after(100, check_for_id_card)

    check_for_id_card()


def verify_password():
    entered_password = password_entry.get()
    if entered_password == "mdx2024":
        switch_to_main_program()
    else:
        messagebox.showerror("Error", "Incorrect password. Please try again.")


def switch_to_main_program():
    verification_frame.pack_forget()
    main_program_frame.pack(fill="both", expand=True)


def update_selected_classes():
    global selected_classes
    selected_classes.clear()
    for class_name, var in class_checkboxes.items():
        if var.get():
            selected_classes.add(class_name)


def start_detection():
    global is_detecting
    is_detecting = True
    cap = cv2.VideoCapture(0)

    def update_frame():
        if not is_detecting:
            cap.release()
            return

        ret, frame = cap.read()
        if not ret:
            return

        # Undistort the frame
        h, w = frame.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            CAMERA_MATRIX, DISTORTION_COEFFS, (w, h), 1, (w, h)
        )
        mapx, mapy = cv2.initUndistortRectifyMap(
            CAMERA_MATRIX, DISTORTION_COEFFS, None, new_camera_matrix, (w, h), 5
        )
        undistorted_frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

        # YOLO to detect objects in the undistorted frame
        results = model.predict(source=undistorted_frame, save=False, conf=0.25)

        detection_details = ""

        for result in results:
            if len(result.boxes) > 0:
                for box in result.boxes:
                    x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()
                    class_name = model.names[int(box.cls.cpu().numpy())]


                    if class_name not in selected_classes:
                        continue

                    real_width = CLASS_WIDTHS.get(class_name, 8.0)
                    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
                    perceived_width = x_max - x_min

                    if perceived_width > 0:
                        focal_length = 480
                        distance = (real_width * focal_length) / perceived_width
                        distance *= CORRECTION_FACTOR
                        distance = round(distance, 2)

                        object_center_x = (x_min + x_max) / 2
                        frame_width = undistorted_frame.shape[1]
                        angle_from_center = (
                            (object_center_x - frame_width / 2)
                            / (frame_width / 2)
                        ) * (HORIZONTAL_FOV / 2)
                        angle_from_center = round(angle_from_center, 2)

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


                        detection_details += f"{class_name} -> Distance: {distance} cm, Angle: {angle_from_center}°\n"

        # Update the detection details label
        details_label.config(text=detection_details.strip())

        # Convert frame to Tkinter-compatible format
        rgb_frame = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        img_tk = ImageTk.PhotoImage(image=img)
        video_label.img_tk = img_tk
        video_label.configure(image=img_tk)
        root.after(10, update_frame)

    update_frame()


def stop_detection():
    global is_detecting
    is_detecting = False
    details_label.config(text="")  # Clear detection details


#GUI
root = tk.Tk()
root.title("Object Detection Program")
root.geometry("1000x700")

# Verification Frame
verification_frame = tk.Frame(root)
verification_frame.pack(fill="both", expand=True)

# Introduction Label
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

# Instructions
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

# ID Card Verification Button
tk.Button(
    verification_frame,
    text="Verify with ID Card",
    font=("Arial", 12),
    command=verify_id_card,
).pack(pady=10)

# Password Verification
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

# Main Program Frame
main_program_frame = tk.Frame(root)

# Instruction label
instruction_label = tk.Label(
    main_program_frame,
    text="Select the objects you want to detect and estimate the distance and angle.",
    font=("Arial", 14),
    bg="lightblue",
    fg="black",
)
instruction_label.pack(fill="x", pady=5)

# Left frame for checkboxes
left_frame = tk.Frame(
    main_program_frame, width=300, height=600, bg="lightgray"
)
left_frame.pack(side="left", fill="y")

# Right frame for video and details
right_frame = tk.Frame(main_program_frame, width=700, height=600)
right_frame.pack(side="right", fill="both", expand=True)

# Create checkboxes for each class
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

# Start button
start_button = ttk.Button(
    left_frame, text="Start Detection", command=start_detection
)
start_button.pack(pady=10)

# Stop button
stop_button = ttk.Button(
    left_frame, text="Stop Detection", command=stop_detection
)
stop_button.pack(pady=10)

# Video display
video_label = tk.Label(right_frame)
video_label.pack(fill="both", expand=True)

# Detection details label with fixed height
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

# Run the GUI
root.mainloop()
