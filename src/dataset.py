import os
import cv2
import torch
import numpy as np
import random
from torch.utils.data import Dataset

class MultiModalityDataset(Dataset):
    def __init__(self, image_dir, mask_dir, is_train=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.is_train = is_train
        # Filter for PNG files and sort for consistency
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        
        # Setup CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # This helps the AI see tumor boundaries in grainy Ultrasound images
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        # 1. Load as Grayscale
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            raise FileNotFoundError(f"Could not load image/mask: {img_name}")

        # 2. Force Resize to 512x512
        image = cv2.resize(image, (512, 512))
        mask = cv2.resize(mask, (512, 512))

        # 3. MODALITY-SPECIFIC PROCESSING
        # We check the filename prefix set by our unify script
        if "us_" in img_name.lower():
            # Apply CLAHE to Ultrasounds to pop the tumor edges
            image = self.clahe.apply(image)
        elif "mg_" in img_name.lower():
            # Mammograms are usually high-contrast enough already
            pass

        # 4. DATA AUGMENTATION (Training Only)
        if self.is_train:
            # Random Horizontal Flip
            if random.random() > 0.5:
                image = cv2.flip(image, 1)
                mask = cv2.flip(mask, 1)
            
            # Random Vertical Flip
            if random.random() > 0.5:
                image = cv2.flip(image, 0)
                mask = cv2.flip(mask, 0)

            # Random 90-degree Rotations
            k = random.randint(0, 3) 
            image = np.rot90(image, k)
            mask = np.rot90(mask, k)

            # Random Intensity Inversion (Handle Dark/Light Tumor variations)
            if random.random() > 0.3: # 30% chance to invert
                image = 255 - image

        # 5. NORMALIZATION & TENSOR CONVERSION
        # Convert to float32 and scale to [0, 1]
        image_np = image.astype(np.float32) / 255.0
        
        # Ensure mask is strictly binary (0.0 or 1.0)
        mask_np = (mask > 127).astype(np.float32)

        # Use .copy() to ensure the array is contiguous in memory after rotations
        image_tensor = torch.from_numpy(image_np.copy()).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_np.copy()).unsqueeze(0)

        return image_tensor, mask_tensor