import cv2
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image
from torchvision import transforms

# Load YOLO model
def load_yolo_model(model_path):
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return None

# Load MiDaS model
def load_midas_model():
    try:
        midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small', pretrained=True)
        midas.eval()
        return midas
    except Exception as e:
        print(f"Error loading MiDaS model: {e}")
        return None

# Define the transformation for MiDaS
def get_midas_transform():
    return transforms.Compose([
        transforms.Resize((384, 384)),  # Resize to the input size of MiDaS
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize
    ])

# Generate depth map using MiDaS
def generate_depth_map(midas, transform, image):
    try:
        image_pil = Image.fromarray(image)
        img_batch = transform(image_pil).unsqueeze(0).to('cuda')  # Add batch dimension and move to GPU

        with torch.no_grad():
            depth_map = midas(img_batch)
            depth_map = torch.nn.functional.interpolate(depth_map.unsqueeze(1), size=image.shape[:2], mode='bicubic',
                                                        align_corners=False).squeeze()
            depth_map = depth_map.cpu().numpy()

        # Display the depth map
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        normalized_depth = (depth_map - depth_min) / (depth_max - depth_min)
        depth_image = (normalized_depth * 255).astype(np.uint8)
        cv2.imshow('Depth Map', depth_image)
        cv2.waitKey(1)
        
        # Print depth map matrix values
        print("Depth Map Matrix:")
        print(depth_map)
        
        return depth_map
    except Exception as e:
        print(f"Error generating depth map: {e}")
        return None

# Calculate distance for each detected object
def calculate_distances(depth_map, results, image):
    depth_scale = 0.001  # Adjust based on your camera specifications
    distances = []

    for result in results:
        boxes = result.boxes
        for box in boxes.data:
            x_min, y_min, x_max, y_max, confidence = box[:5].tolist()
            class_id = int(box[5].item())

            centroid_x = int((x_min + x_max) / 2)
            centroid_y = int((y_min + y_max) / 2)

            # Extract depth value
            if 0 <= centroid_y < depth_map.shape[0] and 0 <= centroid_x < depth_map.shape[1]:
                depth_value = depth_map[centroid_y, centroid_x]
                distance = (depth_value * depth_scale) if depth_value > 0 else 0
                distances.append((distance, (x_min, y_min, x_max, y_max)))

                # Annotate image with distance
                cv2.putText(image, f"Distance: {distance:.2f} m", (int(x_min), int(y_min - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 2)
                print(f"Depth value at centroid: {depth_value}")

                # Draw the centroid
                cv2.circle(image, (centroid_x, centroid_y), 5, (0, 255, 0), -1)
                cv2.putText(image, f"Centroid: ({centroid_x}, {centroid_y})", (centroid_x + 10, centroid_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return distances

# Main function
def main(yolo_model_path):
    yolo_model = load_yolo_model(yolo_model_path)
    if yolo_model is None:
        return

    midas = load_midas_model()
    if midas is None:
        return

    transform = get_midas_transform()
    midas.to('cuda')  # Move MiDaS model to GPU if available

    # Start webcam capture
    cap = cv2.VideoCapture(0)  # 0 for the default camera
    while True:
        success, image = cap.read()
        if not success:
            print("Error: Could not read frame from webcam.")
            break

        results = yolo_model(image)  # Perform object detection
        depth_map = generate_depth_map(midas, transform, image)
        if depth_map is None:
            print("Error: Depth map generation failed.")
            continue

        distances = calculate_distances(depth_map, results, image)

        cv2.imshow('Detected Objects with Distances', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

# Entry point
if __name__ == "__main__":
    yolo_model_path = "C:/Users/LakiBitz/Desktop/UnoCardDetection/runs/detect/train/weights/best.pt"  # Your YOLO model path
    main(yolo_model_path)
