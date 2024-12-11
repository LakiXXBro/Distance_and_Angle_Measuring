import cv2
import numpy as np
import os

# Path to the folder containing checkerboard images
image_folder_path = r'C:\Users\LakiBitz\Downloads\checkerboardx'

# Checkerboard dimensions (number of inner corners per a chessboard row and column)
checkerboard_size = (6, 9)  # Adjust based on your checkerboard pattern
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points (3D points in real-world space)
objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)

obj_points = []  # 3D points in real-world space
img_points = []  # 2D points in image plane

# Read and process each image in the folder
image_files = [f for f in os.listdir(image_folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

for file_name in image_files:
    image_path = os.path.join(image_folder_path, file_name)
    img = cv2.imread(image_path)

    if img is None:
        print(f"Failed to load image: {image_path}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

    if ret:
        print(f"Checkerboard detected in: {file_name}")
        obj_points.append(objp)
        # Refine corner positions
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        img_points.append(corners2)

        # Optionally draw and display the corners
        img = cv2.drawChessboardCorners(img, checkerboard_size, corners2, ret)
        cv2.imshow('Checkerboard', img)
        cv2.waitKey(500)
    else:
        print(f"No checkerboard detected in: {file_name}")

cv2.destroyAllWindows()

# Perform camera calibration
if len(obj_points) > 0:
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, gray.shape[::-1], None, None
    )

    print("Camera Matrix:\n", camera_matrix)
    print("Distortion Coefficients:\n", dist_coeffs)
    print("Reprojection Error:", ret)
else:
    print("No valid checkerboard patterns found. Calibration aborted.")
