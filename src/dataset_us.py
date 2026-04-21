"""
dataset_us.py — Breast Ultrasound Segmentation Dataset
=======================================================
Clean PyTorch Dataset for binary lesion segmentation on breast ultrasound images.

Key design decisions:
  • Zero-mismatch policy: every image MUST have an exact filename match in masks/.
    A descriptive FileNotFoundError is raised immediately on any mismatch.
  • Full-image resize approach (→ IMAGE_SIZE × IMAGE_SIZE).
  • CLAHE contrast enhancement + percentile normalization before augmentation.
  • Albumentations pipeline with LOCKED spatial transforms (image + mask)
    and intensity transforms applied ONLY to the image.
  • Post-augmentation mask re-binarization to eliminate interpolation artifacts.
  • Safety check utility to inspect unique mask values in the dataset.

Data layout:
  <DATA_ROOT>/
    ├── train/
    │   ├── images/
    │   └── masks/
    └── test/
        ├── images/
        └── masks/
"""

from __future__ import annotations

import logging
import os
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

# ─── Supported image extensions ─────────────────────────────────────────────
VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


# ═════════════════════════════════════════════════════════════════════════════
#  FILENAME PAIRING — ZERO MISMATCH POLICY
# ═════════════════════════════════════════════════════════════════════════════

def _list_valid_files(directory: str) -> List[str]:
    """List all files with valid image extensions in a directory."""
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    return sorted([
        f for f in os.listdir(directory)
        if f.lower().endswith(VALID_EXTENSIONS) and not f.startswith(".")
    ])


def get_strict_paired_filenames(image_dir: str, mask_dir: str) -> List[str]:
    """
    Enforce STRICT filename synchronization between image_dir and mask_dir.

    Unlike a soft intersection, this function RAISES immediately if any image
    file does not have a matching mask file. This is the Zero Mismatch Policy.

    Returns:
        Sorted list of filenames present in BOTH directories.

    Raises:
        FileNotFoundError: if any image has no matching mask (with details).
    """
    image_files = set(_list_valid_files(image_dir))
    mask_files = set(_list_valid_files(mask_dir))

    orphan_images = sorted(image_files - mask_files)
    orphan_masks = sorted(mask_files - image_files)

    # ── Log pairing report ──────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("US DATASET PAIRING REPORT")
    logger.info("=" * 60)
    logger.info(f"  Image directory : {image_dir}")
    logger.info(f"  Mask directory  : {mask_dir}")
    logger.info(f"  Total images    : {len(image_files)}")
    logger.info(f"  Total masks     : {len(mask_files)}")
    logger.info(f"  Orphan images   : {len(orphan_images)}")
    logger.info(f"  Orphan masks    : {len(orphan_masks)}")

    # ── ZERO MISMATCH: raise on any orphan image ────────────────────────
    if orphan_images:
        sample = orphan_images[:10]
        raise FileNotFoundError(
            f"ZERO MISMATCH VIOLATION: {len(orphan_images)} image(s) in "
            f"'{image_dir}' have no matching mask in '{mask_dir}'.\n"
            f"First offenders: {sample}\n"
            f"Fix: add the missing masks or remove the orphan images."
        )

    if orphan_masks:
        logger.warning(
            f"  {len(orphan_masks)} mask(s) have no matching image "
            f"(they will be ignored): {orphan_masks[:10]}"
        )

    paired = sorted(image_files & mask_files)

    if len(paired) == 0:
        raise RuntimeError(
            f"No valid image-mask pairs found.\n"
            f"  Images in {image_dir}: {len(image_files)}\n"
            f"  Masks in {mask_dir}: {len(mask_files)}\n"
            f"  Ensure filenames match exactly."
        )

    logger.info(f"  Valid pairs     : {len(paired)}")
    logger.info("=" * 60)

    return paired


# ═════════════════════════════════════════════════════════════════════════════
#  SAFETY CHECK — Mask Binarization Inspector
# ═════════════════════════════════════════════════════════════════════════════

def safety_check_masks(mask_dir: str, num_samples: int = 10) -> None:
    """
    Load a sample of masks and print their unique pixel values.

    This is a pre-training sanity check to verify masks are correctly
    binarized and free of compression artifacts.

    Args:
        mask_dir:    Path to the directory containing mask images.
        num_samples: Number of masks to inspect (default 10).
    """
    files = _list_valid_files(mask_dir)
    if len(files) == 0:
        print(f"[SAFETY CHECK] ⚠️  No mask files found in: {mask_dir}")
        return

    num_samples = min(num_samples, len(files))
    print(f"\n{'='*60}")
    print(f"SAFETY CHECK — Mask Binarization Inspector")
    print(f"Directory: {mask_dir}")
    print(f"Inspecting {num_samples} / {len(files)} masks")
    print(f"{'='*60}")

    all_clean = True
    for fname in files[:num_samples]:
        path = os.path.join(mask_dir, fname)
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"  ❌ {fname}: FAILED TO LOAD")
            all_clean = False
            continue

        unique_vals = np.unique(mask)
        is_binary = set(unique_vals.tolist()).issubset({0, 255}) or \
                     set(unique_vals.tolist()).issubset({0, 1})
        status = "✅" if is_binary else "⚠️  NOT BINARY"
        print(f"  {status} {fname}: unique={unique_vals.tolist()}, "
              f"shape={mask.shape}, dtype={mask.dtype}")
        if not is_binary:
            all_clean = False

    if all_clean:
        print(f"\n✅ All {num_samples} inspected masks are correctly binarized.")
    else:
        print(f"\n⚠️  Some masks have non-binary values. "
              f"They will be thresholded during loading (mask > 0.5).")
    print(f"{'='*60}\n")


# ═════════════════════════════════════════════════════════════════════════════
#  AUGMENTATION PIPELINES
# ═════════════════════════════════════════════════════════════════════════════

def build_train_transforms(image_size: int = 256) -> A.Compose:
    """
    Training augmentation pipeline for breast ultrasound.

    AUGMENTATION CONSISTENCY GUARANTEE:
      - Spatial transforms (flip, rotate, shift-scale-rotate, elastic) are
        applied to BOTH image and mask (Albumentations handles this via the
        `image` + `mask` keys automatically).
      - Intensity transforms (brightness/contrast, noise, blur) ONLY affect
        the image — they are placed AFTER all spatial transforms and use
        Albumentations' built-in behavior (mask is never passed to them).

    Pipeline:
      1. Resize → target resolution
      2. Spatial: HFlip, VFlip, ShiftScaleRotate, ElasticTransform
      3. Intensity (image only): BrightnessContrast, GaussNoise, GaussianBlur
      4. Normalize + ToTensor
    """
    return A.Compose([
        A.Resize(image_size, image_size),

        # ── Spatial augmentations (locked for image + mask) ──
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.15,
            rotate_limit=20,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.5,
        ),
        A.ElasticTransform(
            alpha=80,
            sigma=80 * 0.05,
            p=0.3,
        ),

        # ── Intensity augmentations (image ONLY — mask is untouched) ──
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.4,
        ),
        A.GaussNoise(p=0.2),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),

        # ── Normalize + ToTensor ──
        A.Normalize(mean=(0.5,), std=(0.25,), max_pixel_value=1.0),
        ToTensorV2(),
    ])


def build_val_transforms(image_size: int = 256) -> A.Compose:
    """Validation/inference transform: resize + normalize + to tensor."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.5,), std=(0.25,), max_pixel_value=1.0),
        ToTensorV2(),
    ])


# ═════════════════════════════════════════════════════════════════════════════
#  DATASET CLASS
# ═════════════════════════════════════════════════════════════════════════════

class UltrasoundSegDataset(Dataset):
    """
    Full-image breast ultrasound segmentation dataset.

    Each sample is a (image, mask) pair where:
      - image: float32 tensor (1, H, W), CLAHE-enhanced + percentile-normalized
      - mask:  float32 tensor (1, H, W), binary {0.0, 1.0}

    Args:
        image_dir:   Path to images/ folder.
        mask_dir:    Path to masks/ folder.
        filenames:   Pre-validated list of filenames (from get_strict_paired_filenames).
        transform:   Albumentations Compose pipeline.
        image_size:  Target spatial resolution.
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        filenames: Sequence[str],
        transform: A.Compose,
        image_size: int = 256,
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.transform = transform

        # ── Validate every filename exists in BOTH directories ──────────
        self.filenames: List[str] = []
        for fname in filenames:
            img_path = os.path.join(image_dir, fname)
            msk_path = os.path.join(mask_dir, fname)
            if not os.path.isfile(img_path):
                raise FileNotFoundError(
                    f"Image file missing: {img_path} — "
                    f"Zero Mismatch Policy violated."
                )
            if not os.path.isfile(msk_path):
                raise FileNotFoundError(
                    f"Mask file missing: {msk_path} — "
                    f"Zero Mismatch Policy violated."
                )
            self.filenames.append(fname)

        if len(self.filenames) == 0:
            raise RuntimeError("UltrasoundSegDataset: No valid samples.")

        logger.info(f"UltrasoundSegDataset initialized: {len(self.filenames)} samples")

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
            image = (
                (image - image.min()) / max(image.max() - image.min(), 1) * 255
            ).astype(np.uint8)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)

        # ── Percentile normalization to [0, 1] ──────────────────────────
        image = image.astype(np.float32)
        p1, p99 = np.percentile(image, 1), np.percentile(image, 99)
        denom = max(p99 - p1, 1e-6)
        image = np.clip((image - p1) / denom, 0.0, 1.0)

        # ── Binarize mask (threshold to eliminate PNG compression artifacts) ──
        mask = (mask > 127).astype(np.float32)  # threshold at 0.5 of uint8 range

        # ── Expand to HWC for Albumentations (H, W, 1) ─────────────────
        image = np.expand_dims(image, axis=-1)

        # ── Apply augmentations ─────────────────────────────────────────
        augmented = self.transform(image=image, mask=mask)

        image_tensor = augmented["image"].float()      # (1, H, W)
        mask_tensor = augmented["mask"].float()         # (H, W)

        # ── Re-binarize after augmentation (interpolation may create non-binary values)
        mask_tensor = (mask_tensor > 0.5).float()

        # Ensure mask shape is (1, H, W)
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)

        # ── Shape assertions (No Silent Failures) ──────────────────────
        assert image_tensor.shape == (1, self.image_size, self.image_size), \
            f"Image shape mismatch: expected (1, {self.image_size}, {self.image_size}), " \
            f"got {image_tensor.shape}"
        assert mask_tensor.shape == (1, self.image_size, self.image_size), \
            f"Mask shape mismatch: expected (1, {self.image_size}, {self.image_size}), " \
            f"got {mask_tensor.shape}"

        return image_tensor, mask_tensor


# ═════════════════════════════════════════════════════════════════════════════
#  SAFE COLLATE (handles rare corrupt/None samples gracefully)
# ═════════════════════════════════════════════════════════════════════════════

def safe_collate(batch):
    """Filter out None samples and stack the rest."""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return torch.zeros(0), torch.zeros(0)
    images = torch.stack([b[0] for b in batch])
    masks = torch.stack([b[1] for b in batch])
    return images, masks
