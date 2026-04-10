"""
Dataset utilities for BOSS segmentation training.

This module is intentionally independent from train.py to avoid circular imports.
"""

from __future__ import annotations

import os
from typing import Optional

import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class MultiModalityDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        img_size: int = 512,
        mode: str = "us",
        is_train: bool = True,
        transform: Optional[A.Compose] = None,
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.mode = mode
        self.is_train = is_train

        self.images = sorted(
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
        )

        self.transform = transform or self._build_default_transform()

    def _build_default_transform(self) -> A.Compose:
        if self.is_train:
            return A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomRotate90(p=0.5),
                A.Normalize(mean=(0.485,), std=(0.229,)),
                ToTensorV2(),
            ])

        return A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.Normalize(mean=(0.485,), std=(0.229,)),
            ToTensorV2(),
        ])

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            zero = torch.zeros((1, self.img_size, self.img_size), dtype=torch.float32)
            return zero, zero

        if self.mode == "us":
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            image = clahe.apply(image)

        augmented = self.transform(image=image, mask=mask)
        image_tensor = augmented["image"].float()
        mask_tensor = augmented["mask"].float().unsqueeze(0) / 255.0
        mask_tensor = (mask_tensor > 0.5).float()

        return image_tensor, mask_tensor
