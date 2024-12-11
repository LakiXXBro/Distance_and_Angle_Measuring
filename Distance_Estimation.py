import torch
import cv2
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from ultralytics import YOLO

# Step 1: Load YOLOv8 model
model = YOLO("C:/Users/LakiBitz/Desktop/UnoCardDetection/runs/detect/train/weights/best.pt")

# Step 2: Perform object detection (using webcam or an image)
results = model.predict(source=0, show=True, conf=0.7)  # Replace '0' with an image path for static images

# Step 3: Load MiDaS model
model_type = "DPT_Large"  # Options: "DPT_Large", "DPT_Hybrid", "MiDaS_small"
midas = torch.hub.load("intel-isl/MiDaS", model_type)

# Move MiDaS to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
midas.eval()

# Load transforms to preprocess the image for MiDaS
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
if model_type in ["DPT_Large", "DPT_Hybrid"]:
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

# Step 4: Estimate depth for the original image
original_img = results[0].orig_img
original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
input_batch = transform(original_rgb).to(device)

with torch.no_grad():
    original_prediction = midas(input_batch)

    # Resize the depth map to match the original image size
    original_prediction = torch.nn.functional.interpolate(
        original_prediction.unsqueeze(1),
        size=original_rgb.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    original_depth_map = original_prediction.cpu().numpy()

# Assume we have already calibrated the camera and calculated the scaling factor
S = 0.5  # Example scaling factor; replace with your actual calculated scaling factor

# Step 5: Process each detected object to estimate depth and calculate distance
for result in results:
    boxes = result.boxes  # Get bounding boxes for detected objects

    for box in boxes:
        # Extract bounding box coordinates
        xmin, ymin, xmax, ymax = map(int, box.xyxy[0])  # Convert to integers

        # Crop the detected object from the original frame
        roi = original_img[ymin:ymax, xmin:xmax]

        # Convert cropped region to RGB (if needed)
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        # Apply the MiDaS transform to the cropped ROI
        input_batch = transform(roi_rgb).to(device)

        # Predict depth for the cropped ROI
        with torch.no_grad():
            prediction = midas(input_batch)

            # Resize the depth map to match the original ROI size
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=roi_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

            depth_map = prediction.cpu().numpy()

        # Calculate the median depth value for the ROI to minimize background influence
        median_depth = np.median(depth_map)

        # Calculate the absolute distance using the scaling factor
        absolute_distance = S * median_depth
        print(f"Estimated Absolute Distance for Detected Object: {absolute_distance:.2f} meters")

        # Normalize depth map for visualization
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        normalized_depth = (depth_map - depth_min) / (depth_max - depth_min)
        depth_image = (normalized_depth * 255).astype(np.uint8)

        # Resize the original camera feed for better display
        scale_percent = 50  # Resize scale (50% of the original size)
        width = int(original_img.shape[1] * scale_percent / 100)
        height = int(original_img.shape[0] * scale_percent / 100)
        resized_orig_img = cv2.resize(original_img, (width, height), interpolation=cv2.INTER_AREA)

        # Display the resized original camera feed, depth map, detected object, and resized depth map of the original image
        cv2.imshow('Camera Feed with Detection', resized_orig_img)
        cv2.imshow('Depth Map', depth_image)
        cv2.imshow('Detected Object', roi)
        cv2.imshow('Depth Map of Original Image', original_depth_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
