import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

def predict_malignancy(image_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load the PNG (Handling potential Unicode path issues)
    img_array = np.fromfile(image_path, np.uint8)
    raw_image = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    
    if raw_image is None:
        print(f"❌ Could not read image: {image_path}")
        return None, None

    # 2. Re-initialize the Architecture
    model = smp.Unet(
        encoder_name="resnet34",        
        in_channels=1,                  
        classes=1,                      
    ).to(device)

    # 3. Load the Trained Weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 4. Preprocess (Must match the Test Dataset exactly)
    transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=(0,), std=(1,)),
        ToTensorV2(),
    ])
    
    input_tensor = transform(image=raw_image)['image'].unsqueeze(0).to(device)

    # 5. Run Inference
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.sigmoid(output).squeeze().cpu().numpy()
    
    # 6. Binary Threshold (0.5 is standard, but can be adjusted)
    binary_prediction = (prediction > 0.5).astype(np.uint8)
    
    # Resize raw image for visualization
    display_img = cv2.resize(raw_image, (512, 512))

    return display_img, binary_prediction

def visualize_malignancy(image, mask):
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title("Malignant Mammogram (PNG)")
    plt.imshow(image, cmap='gray')

    # Create a Red Overlay for the tumor
    overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # Color pixels where mask is 1 as Red
    overlay[mask == 1] = [255, 0, 0] 

    plt.subplot(1, 2, 2)
    plt.title("AI Detection (Malignant Mass)")
    plt.imshow(overlay)
    
    plt.show()

if __name__ == "__main__":
    # Update these paths to your new PNG test images
    TEST_PNG = r"D:\Projet_Federateur\AI\Breast Oncology Surveillance System\data\processed\test\images\malignant(200).png"
    MODEL_PATH = r"D:\Projet_Federateur\AI\Breast Oncology Surveillance System\models\best_malignant_model.pth"
    
    img, pred = predict_malignancy(TEST_PNG, MODEL_PATH)
    if img is not None:
        visualize_malignancy(img, pred)