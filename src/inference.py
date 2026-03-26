import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from pathlib import Path

def predict_tumor(image_path, model_path):
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Re-initialize the Architecture
    model = smp.Unet(
        encoder_name="resnet34",        
        in_channels=1,                  
        classes=1,                      
    ).to(device)

    # 3. Load the Trained Weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # Set to evaluation mode

    # 4. Preprocess the Input Image
    raw_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Ensure it's 512x512
    image_resized = cv2.resize(raw_image, (512, 512))
    # Normalize and convert to Tensor [1, 1, 512, 512]
    input_tensor = torch.from_numpy(image_resized).float().unsqueeze(0).unsqueeze(0) / 255.0
    input_tensor = input_tensor.to(device)

    # 5. Run Inference
    with torch.no_grad():
        output = model(input_tensor)
        # Apply Sigmoid to get probability (0 to 1)
        prediction = torch.sigmoid(output).squeeze().cpu().numpy()
    
    # 6. Thresholding
    # Any pixel with > 50% probability is considered "Tumor"
    binary_prediction = (prediction > 0.5).astype(np.uint8)

    return image_resized, binary_prediction

def visualize_result(image, mask):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title("AI Prediction (Mask)")
    plt.imshow(mask, cmap='jet')

    # Overlay the mask on the image in red
    overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    overlay[mask == 1] = [255, 0, 0] # Red color for tumor

    plt.subplot(1, 3, 3)
    plt.title("Overlay Analysis")
    plt.imshow(overlay)
    
    plt.show()

if __name__ == "__main__":
    TEST_IMG = "data/processed/train/images/pic.png"
    MODEL_PATH = "models/unet_mammogram.pth"
    
    img, pred = predict_tumor(TEST_IMG, MODEL_PATH)
    visualize_result(img, pred)