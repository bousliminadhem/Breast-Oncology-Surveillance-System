import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class MultiModalityDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size=512, mode="us"):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.mode = mode
        
        # Valid image extensions
        self.images = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        # --- ELITE AUGMENTATION PIPELINE ---
        # We apply these transforms to keep the model from "memorizing" the data
        self.transform = A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            # ElasticTransform is the "Secret Sauce" for Ultrasound
            A.OneOf([
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
                A.GridDistortion(p=0.3),
                A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.3),
            ], p=0.3),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=(0.485,), std=(0.229,)), # Standard ImageNet values
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        
        # Load grayscale
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None or mask is None:
            return torch.zeros((1, self.img_size, self.img_size)), torch.zeros((1, self.img_size, self.img_size))

        # --- MEDICAL PREPROCESSING ---
        if self.mode == "us":
            # CLAHE is essential for Ultrasound speckle contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            image = clahe.apply(image)
        
        # Apply Albumentations (Ensure mask is transformed exactly like the image)
        augmented = self.transform(image=image, mask=mask)
        image_tensor = augmented['image']
        mask_tensor = augmented['mask'].float().unsqueeze(0) / 255.0
        
        # Binarize mask
        mask_tensor = (mask_tensor > 0.5).float()
        
        return image_tensor, mask_tensor