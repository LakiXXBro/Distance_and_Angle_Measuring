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

# Step 4: Estimate depth for the reference object
original_img = results[0].orig_img
original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
input_batch = transform(original_rgb).to(device)

# Known distance of the reference object in meters
D_actual = 0.30  # Reference object distance in meters

# Use YOLO to detect the reference object and estimate depth with MiDaS
for result in results:
    boxes = result.boxes  # Get bounding boxes for detected objects

    # Assuming the reference object is the first detected object (update this as per your use case)
    ref_box = boxes[0]  # Get the reference object bounding box

    # Extract bounding box coordinates for reference object
    xmin, ymin, xmax, ymax = map(int, ref_box.xyxy[0])  # Convert to integers
    ref_roi = original_img[ymin:ymax, xmin:xmax]  # Crop the reference object from the original frame

    # Convert cropped region to RGB (if needed)
    ref_roi_rgb = cv2.cvtColor(ref_roi, cv2.COLOR_BGR2RGB)

    # Apply the MiDaS transform to the cropped reference ROI
    input_batch = transform(ref_roi_rgb).to(device)

    # Predict depth for the reference ROI
    with torch.no_grad():
        ref_prediction = midas(input_batch)

        # Resize the depth map to match the reference ROI size
        ref_prediction = torch.nn.functional.interpolate(
            ref_prediction.unsqueeze(1),
            size=ref_roi_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        ref_depth_map = ref_prediction.cpu().numpy()

    # Calculate the median depth value for the reference object
    D_relative_ref = np.median(ref_depth_map)

    # Calculate the scaling factor using the known distance of the reference object
    S = D_actual / D_relative_ref
    print(f"Scaling Factor: {S:.2f}")

    break  # Assuming only one reference object is needed
