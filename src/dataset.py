import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class MammographyDatasetV3(Dataset):
    def __init__(self, image_dir, mask_dir, is_train=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.png')]
        
        # Define Augmentations
        if is_train:
            self.transform = A.Compose([
                A.Resize(512, 512),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                A.Normalize(mean=(0,), std=(1,)),
                ToTensorV2(),
            ])
        else:
            # No random flips for testing, just resize and normalize
            self.transform = A.Compose([
                A.Resize(512, 512),
                A.Normalize(mean=(0,), std=(1,)),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        # Load as Grayscale
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Ensure mask is binary (0 or 255)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Apply Albumentations (handles both image and mask together)
        augmented = self.transform(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask'].float().unsqueeze(0) / 255.0 # Add channel and normalize

        return image, mask