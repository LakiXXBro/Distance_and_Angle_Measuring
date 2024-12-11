import torch
import cv2
import numpy as np

#MiDaS model
model_type = "DPT_Large"
midas = torch.hub.load("intel-isl/MiDaS", model_type)

# Move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
midas.eval()

# Load transforms to preprocess the image
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform
# Load an image
img = cv2.imread("C:/Users/LakiBitz/Downloads/pic4.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Apply transforms to the image
input_batch = transform(img).to(device)

# Predict depth
with torch.no_grad():
    prediction = midas(input_batch)

    # Resize depth map to match the input image
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    depth_map = prediction.cpu().numpy()

# Normalize depth map for visualization
depth_min = depth_map.min()
depth_max = depth_map.max()
normalized_depth = (depth_map - depth_min) / (depth_max - depth_min)
depth_image = (normalized_depth * 255).astype(np.uint8)

# Resize depth image for display
scale_percent = 50  # Resize scale (50% of the original size)
width = int(depth_image.shape[1] * scale_percent / 100)
height = int(depth_image.shape[0] * scale_percent / 100)
resized_depth_image = cv2.resize(depth_image, (width, height), interpolation=cv2.INTER_AREA)


colormap = cv2.applyColorMap(resized_depth_image, cv2.COLORMAP_JET)

# Display the colored depth map
cv2.imshow('Colored Depth Map', colormap)
cv2.waitKey(0)
cv2.destroyAllWindows()
