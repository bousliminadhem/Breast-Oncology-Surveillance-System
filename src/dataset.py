import os
import cv2
import torch
import numpy as np
import random
from torch.utils.data import Dataset

class MultiModalityDataset(Dataset):
    def __init__(self, image_dir, mask_dir, is_train=True):
        """
        Custom Dataset for Breast Oncology Surveillance System.
        Args:
            image_dir (str): Path to folder containing .png images.
            mask_dir (str): Path to folder containing corresponding .png masks.
            is_train (bool): If True, applies random augmentations.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.is_train = is_train
        
        # Filter for PNG files and sort for consistency between images and masks
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Directory not found: {image_dir}")
            
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        # 1. Load as Grayscale (Medical standard)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            raise FileNotFoundError(f"Could not load image/mask: {img_name}")

        # 2. Force Resize to 512x512 (Input layer requirement)
        image = cv2.resize(image, (512, 512))
        mask = cv2.resize(mask, (512, 512))

        # 3. MODALITY-SPECIFIC PROCESSING (CLAHE for Ultrasound)
        # We create the CLAHE object locally here to allow Multiprocessing on Windows
        if "us_" in img_name.lower():
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            image = clahe.apply(image)
        elif "mg_" in img_name.lower():
            # Mammography usually has high native contrast; skip CLAHE to avoid artifacts
            pass

        # 4. DATA AUGMENTATION (Training Only)
        # We perform geometric transforms on BOTH image and mask simultaneously
        if self.is_train:
            # Random Horizontal Flip
            if random.random() > 0.5:
                image = cv2.flip(image, 1)
                mask = cv2.flip(mask, 1)
            
            # Random Vertical Flip
            if random.random() > 0.5:
                image = cv2.flip(image, 0)
                mask = cv2.flip(mask, 0)

            # Random 90-degree Rotations (0, 90, 180, or 270)
            k = random.randint(0, 3) 
            if k > 0:
                image = np.rot90(image, k)
                mask = np.rot90(mask, k)

            # Random Intensity Inversion (Simulates different tissue density looks)
            if random.random() > 0.3: 
                image = 255 - image

        # 5. NORMALIZATION & TENSOR CONVERSION
        # Scaling pixel values to [0.0, 1.0] for stable gradient descent
        image_np = image.astype(np.float32) / 255.0
        
        # Ensure mask is strictly binary (127 is the threshold for 8-bit grayscale)
        mask_np = (mask > 127).astype(np.float32)

        # CRITICAL: .copy() ensures the memory is contiguous after np.rot90/flip
        # PyTorch tensors require contiguous memory to be sent to the GPU (CUDA)
        image_tensor = torch.from_numpy(image_np.copy()).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_np.copy()).unsqueeze(0)

        return image_tensor, mask_tensor