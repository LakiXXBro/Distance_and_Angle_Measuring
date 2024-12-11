import cv2
import torch
from PIL import Image
from torchvision import transforms

# Loading MiDaS model
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
        transforms.Resize((384, 384)),
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize
    ])

# Generating depth map using MiDaS
def generate_depth_map(midas, transform, image):
    try:
        image_pil = Image.fromarray(image)
        img_batch = transform(image_pil).unsqueeze(0).to('cuda')  # Add batch dimension and move to GPU

        with torch.no_grad():
            depth_map = midas(img_batch)
            depth_map = torch.nn.functional.interpolate(depth_map.unsqueeze(1), size=image.shape[:2], mode='bicubic',
                                                        align_corners=False).squeeze()
            depth_map = depth_map.cpu().numpy()
        return depth_map
    except Exception as e:
        print(f"Error generating depth map: {e}")
        return None

# Main function
def main():
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

        depth_map = generate_depth_map(midas, transform, image)
        if depth_map is None:
            print("Error: Depth map generation failed.")
            continue

        # Normalize depth map for display
        depth_map_display = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        depth_map_colored = cv2.applyColorMap(depth_map_display, cv2.COLORMAP_JET)

        # Display the depth map
        cv2.imshow('Depth Map', depth_map_colored)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
