import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class MultiModalityDataset(Dataset):
    # Ensure 'img_size' is exactly as written here
    def __init__(self, image_dir, mask_dir, img_size=512, mode="us"):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.mode = mode
        
        # Filter out non-image files like .DS_Store
        self.images = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        
        # Load as Grayscale
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None or mask is None:
            # Fallback if a file is corrupted
            return torch.zeros((1, self.img_size, self.img_size)), torch.zeros((1, self.img_size, self.img_size))

        # Resize to the size expected by ResNet-50
        image = cv2.resize(image, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size))
        
        # --- MEDICAL PREPROCESSING ---
        if self.mode == "us":
            # Enhance contrast for Ultrasound scans
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            image = clahe.apply(image)
            
        # Convert to Tensors [C, H, W]
        image_tensor = torch.from_numpy(image).float().unsqueeze(0) / 255.0
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0) / 255.0
        
        # Ensure mask is strictly 0s and 1s
        mask_tensor = (mask_tensor > 0.5).float()
        
        return image_tensor, mask_tensor