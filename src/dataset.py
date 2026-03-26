import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

class MammographyDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        # Get list of all images (filenames match in both folders)
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.png')]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_name = self.images[index]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        
        # Load as grayscale (0)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Normalize pixels to [0, 1] range
        image = np.array(image, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.float32) / 255.0
        
        # PyTorch expects (Channel, Height, Width)
        # We add the channel dimension manually for grayscale
        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)
        
        return torch.from_numpy(image), torch.from_numpy(mask)