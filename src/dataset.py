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
        
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Directory not found: {image_dir}")
            
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            raise FileNotFoundError(f"Could not load image/mask: {img_name}")

        image = cv2.resize(image, (512, 512))
        mask = cv2.resize(mask, (512, 512))

        # --- PRE-PROCESSING ---
        if "us_" in img_name.lower():
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            image = clahe.apply(image)

        # --- INTENSITY & GEOMETRIC AUGMENTATIONS (TRAIN ONLY) ---
        if self.is_train:
            # 1. Random Brightness & Contrast (Simulates different sensor gains)
            if random.random() > 0.5:
                alpha = random.uniform(0.8, 1.2) # Contrast factor
                beta = random.uniform(-20, 20)   # Brightness shift
                image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

            # 2. Random Flips
            if random.random() > 0.5:
                image = cv2.flip(image, 1)
                mask = cv2.flip(mask, 1)
            if random.random() > 0.5:
                image = cv2.flip(image, 0)
                mask = cv2.flip(mask, 0)

            # 3. Random Rotations
            k = random.randint(0, 3) 
            if k > 0:
                image = np.rot90(image, k)
                mask = np.rot90(mask, k)

        # --- NORMALIZATION & NOISE ---
        image_np = image.astype(np.float32) / 255.0
        mask_np = (mask > 127).astype(np.float32)

        if self.is_train:
            # 4. Gaussian Noise (Essential for graininess robustness)
            if random.random() > 0.3:
                noise = np.random.normal(0, 0.02, image_np.shape).astype(np.float32)
                image_np = np.clip(image_np + noise, 0, 1)

            # 5. Intensity Inversion
            if random.random() > 0.2: 
                image_np = 1.0 - image_np

        # --- FINAL TENSOR CONVERSION ---
        image_tensor = torch.from_numpy(image_np.copy()).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_np.copy()).unsqueeze(0)

        return image_tensor, mask_tensor