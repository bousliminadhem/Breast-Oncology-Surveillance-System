"""
dataset_mg.py — Mammography Segmentation Dataset
=================================================
Clean PyTorch Dataset for binary tumor segmentation on mammography images.

Key design decisions:
  • Full-image approach (resize 1024→512) — NOT patch-based.
  • Rigorous filename intersection: only images present in BOTH /TIFF/ and /Masks/ are used.
  • CLAHE contrast enhancement + percentile normalization before augmentation.
  • Albumentations pipeline with spatial + intensity transforms for ~260 image pool.
  • Every __getitem__ verifies image/mask filename correspondence.
  • Detailed logging of pairing statistics — zero silent failures.

Root path: /content/drive/MyDrive/Breast Oncology Surveillance System/data/1024
  ├── TIFF/   (source mammography images, 1024x1024 PNGs)
  └── Masks/  (binary tumor masks, 1024x1024 PNGs)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ─── Module logger ──────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(name)s] %(levelname)s: %(message)s"))
    logger.addHandler(_handler)

# ─── Supported extensions ───────────────────────────────────────────────────
VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

# ─── Normalization constants (used in inference too — see app.py) ───────────
MG_MEAN = (0.5,)
MG_STD = (0.25,)


# ═════════════════════════════════════════════════════════════════════════════
#  FILENAME PAIRING — STRICT INTERSECTION
# ═════════════════════════════════════════════════════════════════════════════

def _list_valid_files(directory: str) -> List[str]:
    """List all files with valid image extensions in a directory."""
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    return sorted([
        f for f in os.listdir(directory)
        if f.lower().endswith(VALID_EXTENSIONS) and not f.startswith(".")
    ])


def get_paired_filenames(image_dir: str, mask_dir: str) -> List[str]:
    """
    Compute the strict intersection of filenames between image_dir and mask_dir.

    Only filenames present in BOTH directories are returned.
    Logs detailed statistics about discarded files.

    Returns:
        Sorted list of filenames that exist in both directories.
    """
    image_files = set(_list_valid_files(image_dir))
    mask_files = set(_list_valid_files(mask_dir))

    paired = sorted(image_files & mask_files)

    # ── Detailed logging ────────────────────────────────────────────────
    orphan_images = image_files - mask_files
    orphan_masks = mask_files - image_files

    logger.info("=" * 60)
    logger.info("DATASET PAIRING REPORT")
    logger.info("=" * 60)
    logger.info(f"  Image directory : {image_dir}")
    logger.info(f"  Mask directory  : {mask_dir}")
    logger.info(f"  Total images    : {len(image_files)}")
    logger.info(f"  Total masks     : {len(mask_files)}")
    logger.info(f"  Valid pairs     : {len(paired)}")
    logger.info(f"  Discarded images (no mask)  : {len(orphan_images)}")
    logger.info(f"  Discarded masks  (no image) : {len(orphan_masks)}")

    if orphan_images:
        logger.warning(f"  Orphan images: {list(orphan_images)[:10]}{'...' if len(orphan_images) > 10 else ''}")
    if orphan_masks:
        logger.warning(f"  Orphan masks:  {list(orphan_masks)[:10]}{'...' if len(orphan_masks) > 10 else ''}")

    logger.info("=" * 60)

    if len(paired) == 0:
        raise RuntimeError(
            f"No valid image-mask pairs found.\n"
            f"  Images in {image_dir}: {len(image_files)}\n"
            f"  Masks in {mask_dir}: {len(mask_files)}\n"
            f"  Check that filenames match exactly between the two directories."
        )

    return paired


# ═════════════════════════════════════════════════════════════════════════════
#  AUGMENTATION PIPELINES
# ═════════════════════════════════════════════════════════════════════════════

def build_train_transforms(image_size: int = 512) -> A.Compose:
    """
    Training augmentation pipeline designed for small mammography datasets (~260 images).

    Includes:
      - Spatial: Flip, Rotate, ElasticTransform, GridDistortion, ShiftScaleRotate
      - Intensity: CLAHE, RandomBrightnessContrast, GaussNoise, GaussianBlur
      - Resize + Normalize + ToTensor
    """
    return A.Compose([
        A.Resize(image_size, image_size),

        # ── Spatial augmentations ──
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.15,
            rotate_limit=15,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.5,
        ),
        A.ElasticTransform(
            alpha=120,
            sigma=120 * 0.05,
            p=0.3,
        ),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),

        # ── Intensity augmentations ──
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.4,
        ),
        A.GaussNoise(p=0.2),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),

        # ── Normalize + ToTensor ──
        A.Normalize(mean=MG_MEAN, std=MG_STD, max_pixel_value=1.0),
        ToTensorV2(),
    ])


def build_val_transforms(image_size: int = 512) -> A.Compose:
    """Validation/inference transform: resize + normalize + to tensor."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=MG_MEAN, std=MG_STD, max_pixel_value=1.0),
        ToTensorV2(),
    ])


# ═════════════════════════════════════════════════════════════════════════════
#  DATASET CLASS
# ═════════════════════════════════════════════════════════════════════════════

class MammographySegDataset(Dataset):
    """
    Full-image mammography segmentation dataset.

    Each sample is a (image, mask) pair where:
      - image: float32 tensor (1, H, W), CLAHE-enhanced + percentile-normalized
      - mask:  float32 tensor (1, H, W), binary {0, 1}

    Args:
        image_dir:   Path to /TIFF/ folder containing source mammography images.
        mask_dir:    Path to /Masks/ folder containing binary tumor masks.
        filenames:   Pre-filtered list of filenames to use (from get_paired_filenames).
        transform:   Albumentations Compose pipeline.
        image_size:  Target spatial resolution (default 512 for T4 GPU stability).
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        filenames: Sequence[str],
        transform: A.Compose,
        image_size: int = 512,
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.transform = transform

        # ── Validate every filename exists in BOTH directories ──────────
        self.filenames: List[str] = []
        skipped = 0
        for fname in filenames:
            img_path = os.path.join(image_dir, fname)
            msk_path = os.path.join(mask_dir, fname)
            if os.path.isfile(img_path) and os.path.isfile(msk_path):
                self.filenames.append(fname)
            else:
                skipped += 1
                logger.warning(f"Skipped '{fname}': missing from image or mask dir.")

        if skipped > 0:
            logger.warning(f"Total skipped during Dataset init: {skipped}")

        if len(self.filenames) == 0:
            raise RuntimeError("MammographySegDataset: No valid samples after validation.")

        logger.info(f"MammographySegDataset initialized: {len(self.filenames)} samples")

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        fname = self.filenames[idx]
        img_path = os.path.join(self.image_dir, fname)
        msk_path = os.path.join(self.mask_dir, fname)

        # ── Load image ──────────────────────────────────────────────────
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise IOError(f"Failed to load image: {img_path}")

        # Convert to single-channel grayscale if needed
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # ── Load mask ───────────────────────────────────────────────────
        mask = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise IOError(f"Failed to load mask: {msk_path}")

        # ── CLAHE contrast enhancement ──────────────────────────────────
        if image.dtype != np.uint8:
            # Normalize to uint8 for CLAHE
            image = ((image - image.min()) / max(image.max() - image.min(), 1) * 255).astype(np.uint8)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)

        # ── Percentile normalization to [0, 1] ──────────────────────────
        image = image.astype(np.float32)
        p1, p99 = np.percentile(image, 1), np.percentile(image, 99)
        denom = max(p99 - p1, 1e-6)
        image = np.clip((image - p1) / denom, 0.0, 1.0)

        # ── Binarize mask ───────────────────────────────────────────────
        mask = (mask > 0).astype(np.float32)

        # ── Expand to HWC for Albumentations (H, W, 1) ─────────────────
        image = np.expand_dims(image, axis=-1)  # (H, W, 1)

        # ── Apply augmentations ─────────────────────────────────────────
        augmented = self.transform(image=image, mask=mask)

        image_tensor = augmented["image"].float()      # (1, H, W)
        mask_tensor = augmented["mask"].float()         # (H, W)

        # Binarize after augmentation (interpolation may create non-binary values)
        mask_tensor = (mask_tensor > 0.5).float()

        # Ensure mask is (1, H, W)
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)

        return image_tensor, mask_tensor


# ═════════════════════════════════════════════════════════════════════════════
#  SAFE COLLATE (handles rare corrupt samples gracefully)
# ═════════════════════════════════════════════════════════════════════════════

def safe_collate(batch):
    """Filter out None samples and stack the rest."""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return torch.zeros(0), torch.zeros(0)
    images = torch.stack([b[0] for b in batch])
    masks = torch.stack([b[1] for b in batch])
    return images, masks
