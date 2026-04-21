"""
train_mg.py — Mammography Segmentation Training Pipeline
=========================================================
Production-grade training script for binary tumor segmentation on mammography.

Architecture:  U-Net++ (UnetPlusPlus) with EfficientNet-B4 encoder (ImageNet pretrained)
Loss:          Dice Loss + Binary Cross Entropy (solves class imbalance)
Optimization:  AdamW + CosineAnnealingWarmRestarts + AMP + Gradient Accumulation
Metrics:       Dice Coefficient (F1) + IoU (Jaccard Index)
Checkpointing: Best model saved by Validation Dice score

Designed for Google Colab T4 GPU (16 GB VRAM).

Usage:
    python train_mg.py
"""

from __future__ import annotations

import logging
import os
import random
import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from dataset_mg import (
    MammographySegDataset,
    get_paired_filenames,
    build_train_transforms,
    build_val_transforms,
    safe_collate,
)

# ─── Logger ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

# ── Data paths ──
DATA_ROOT = "/content/drive/MyDrive/Breast Oncology Surveillance System/data/1024"
IMAGE_DIR = os.path.join(DATA_ROOT, "TIFF")
MASK_DIR = os.path.join(DATA_ROOT, "Masks")
SAVE_DIR = "/content/drive/MyDrive/Breast Oncology Surveillance System/models"

# ── Model ──
ARCHITECTURE = "UnetPlusPlus"
ENCODER = "efficientnet-b4"
ENCODER_WEIGHTS = "imagenet"
IN_CHANNELS = 1
NUM_CLASSES = 1

# ── Training ──
IMAGE_SIZE = 512          # Resize from 1024 → 512 for T4 stability
BATCH_SIZE = 4            # Physical batch size on T4
ACCUM_STEPS = 4           # Effective batch = 4 × 4 = 16
EPOCHS = 100
LR = 3e-4
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.2
SEED = 42
NUM_WORKERS = 2
PATIENCE = 25             # Early stopping patience

# ── Loss weights ──
DICE_WEIGHT = 0.6
BCE_WEIGHT = 0.4
BCE_POS_WEIGHT = 3.0      # Upweight foreground class (tumors are small)

# ── Scheduler ──
COSINE_T0 = 20            # Restart period for CosineAnnealingWarmRestarts
COSINE_T_MULT = 2         # Multiply period after each restart

# ── Device ──
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═════════════════════════════════════════════════════════════════════════════
#  REPRODUCIBILITY
# ═════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int = SEED):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ═════════════════════════════════════════════════════════════════════════════
#  MODEL
# ═════════════════════════════════════════════════════════════════════════════

def build_model() -> nn.Module:
    """
    Build U-Net++ with EfficientNet-B4 encoder pretrained on ImageNet.

    Key choices:
      - UnetPlusPlus: dense skip connections improve boundary precision.
      - EfficientNet-B4: strong feature extraction at reasonable VRAM cost.
      - scse attention: channel + spatial squeeze-excitation in decoder.
      - in_channels=1: grayscale mammography input.
      - classes=1: binary segmentation output (tumor vs background).
    """
    model = smp.UnetPlusPlus(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=IN_CHANNELS,
        classes=NUM_CLASSES,
        decoder_attention_type="scse",
    )
    model = model.to(DEVICE)

    # Log parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {ARCHITECTURE} + {ENCODER}")
    logger.info(f"  Total parameters    : {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")

    return model


# ═════════════════════════════════════════════════════════════════════════════
#  LOSS FUNCTION — Dice + BCE (Class Imbalance Strategy)
# ═════════════════════════════════════════════════════════════════════════════

class DiceBCELoss(nn.Module):
    """
    Combined Dice Loss + Binary Cross Entropy for class-imbalanced segmentation.

    Why this combination:
      - Dice Loss directly optimizes the Dice coefficient (overlap metric),
        naturally handling class imbalance by focusing on the minority class.
      - BCE provides pixel-level supervision and stable gradients early in training.
      - BCE pos_weight further upweights the tumor (foreground) class.

    Args:
        dice_weight:  Weight for Dice Loss component.
        bce_weight:   Weight for BCE component.
        bce_pos_weight: Positive class weight for BCEWithLogitsLoss.
        smooth:       Smoothing factor for Dice Loss (numerical stability).
    """

    def __init__(
        self,
        dice_weight: float = DICE_WEIGHT,
        bce_weight: float = BCE_WEIGHT,
        bce_pos_weight: float = BCE_POS_WEIGHT,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth

        pos_weight = torch.tensor([bce_pos_weight], device=DEVICE)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def _dice_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute soft Dice Loss from logits."""
        probs = torch.sigmoid(logits)
        # Flatten spatial dimensions
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        cardinality = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice_loss = self._dice_loss(logits, targets)
        bce_loss = self.bce(logits, targets)
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss


# ═════════════════════════════════════════════════════════════════════════════
#  METRICS — Dice Coefficient (F1) + IoU (Jaccard)
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> Tuple[float, float]:
    """
    Compute Dice Coefficient and IoU from model logits.

    Args:
        logits:    Raw model output (B, 1, H, W) — before sigmoid.
        targets:   Ground truth binary masks (B, 1, H, W).
        threshold: Binarization threshold for predictions.
        eps:       Smoothing to avoid division by zero.

    Returns:
        (dice, iou) — batch-averaged metrics.
    """
    preds = (torch.sigmoid(logits) > threshold).float()

    preds_flat = preds.view(preds.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)

    intersection = (preds_flat * targets_flat).sum(dim=1)
    pred_sum = preds_flat.sum(dim=1)
    target_sum = targets_flat.sum(dim=1)

    # Dice = 2*|P ∩ T| / (|P| + |T|)
    dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)

    # IoU = |P ∩ T| / |P ∪ T|
    union = pred_sum + target_sum - intersection
    iou = (intersection + eps) / (union + eps)

    return dice.mean().item(), iou.mean().item()


# ═════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════

def split_filenames(
    filenames: list,
    val_split: float = VAL_SPLIT,
    seed: int = SEED,
) -> Tuple[list, list]:
    """Split filenames into train/val sets with deterministic shuffle."""
    if len(filenames) < 2:
        raise RuntimeError("Need at least 2 paired files for train/val split.")

    shuffled = list(filenames)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    n_val = max(1, int(round(len(shuffled) * val_split)))
    n_val = min(n_val, len(shuffled) - 1)  # Ensure at least 1 train sample

    val_files = shuffled[:n_val]
    train_files = shuffled[n_val:]

    logger.info(f"Train/Val split: {len(train_files)} train, {len(val_files)} val "
                f"({val_split*100:.0f}% val)")

    return train_files, val_files


def create_dataloaders(
    batch_size: int = BATCH_SIZE,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders.

    Performs:
      1. Rigorous filename intersection between TIFF/ and Masks/.
      2. Deterministic train/val split.
      3. Separate augmentation pipelines for train (heavy) and val (minimal).
    """
    # ── Step 1: Find all valid pairs ──
    paired_files = get_paired_filenames(IMAGE_DIR, MASK_DIR)

    # ── Step 2: Train/Val split ──
    train_files, val_files = split_filenames(paired_files)

    # ── Step 3: Build datasets ──
    train_transform = build_train_transforms(IMAGE_SIZE)
    val_transform = build_val_transforms(IMAGE_SIZE)

    train_ds = MammographySegDataset(
        image_dir=IMAGE_DIR,
        mask_dir=MASK_DIR,
        filenames=train_files,
        transform=train_transform,
        image_size=IMAGE_SIZE,
    )

    val_ds = MammographySegDataset(
        image_dir=IMAGE_DIR,
        mask_dir=MASK_DIR,
        filenames=val_files,
        transform=val_transform,
        image_size=IMAGE_SIZE,
    )

    # ── Step 4: DataLoaders ──
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=safe_collate,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=safe_collate,
        drop_last=False,
    )

    logger.info(f"DataLoaders ready: "
                f"train={len(train_ds)} samples ({len(train_loader)} batches), "
                f"val={len(val_ds)} samples ({len(val_loader)} batches)")

    return train_loader, val_loader


# ═════════════════════════════════════════════════════════════════════════════
#  TRAINING LOOP — Single Epoch
# ═════════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    accum_steps: int = ACCUM_STEPS,
) -> Dict[str, float]:
    """
    Train for one epoch with AMP and gradient accumulation.

    Returns:
        Dict with 'loss', 'dice', 'iou' averaged over the epoch.
    """
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    valid_steps = 0

    optimizer.zero_grad(set_to_none=True)

    for step, (images, masks) in enumerate(loader):
        if images.numel() == 0:
            continue

        images = images.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)

        # ── Forward pass with AMP ──
        with autocast("cuda", enabled=torch.cuda.is_available()):
            logits = model(images)
            loss = criterion(logits, masks)
            loss_scaled = loss / accum_steps

        # ── Backward pass ──
        scaler.scale(loss_scaled).backward()
        valid_steps += 1

        # ── Optimizer step every accum_steps ──
        if valid_steps % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # ── Track metrics ──
        total_loss += loss.item()
        with torch.no_grad():
            dice, iou = compute_metrics(logits, masks)
            total_dice += dice
            total_iou += iou

    # ── Flush remaining gradients ──
    if valid_steps > 0 and valid_steps % accum_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    if valid_steps == 0:
        raise RuntimeError("Training loader produced no valid batches.")

    return {
        "loss": total_loss / valid_steps,
        "dice": total_dice / valid_steps,
        "iou": total_iou / valid_steps,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  VALIDATION LOOP
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> Dict[str, float]:
    """
    Validate the model on the validation set.

    Returns:
        Dict with 'loss', 'dice', 'iou' averaged over the validation set.
    """
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    valid_steps = 0

    for images, masks in loader:
        if images.numel() == 0:
            continue

        images = images.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)

        with autocast("cuda", enabled=torch.cuda.is_available()):
            logits = model(images)
            loss = criterion(logits, masks)

        total_loss += loss.item()
        dice, iou = compute_metrics(logits, masks)
        total_dice += dice
        total_iou += iou
        valid_steps += 1

    if valid_steps == 0:
        raise RuntimeError("Validation loader produced no valid batches.")

    return {
        "loss": total_loss / valid_steps,
        "dice": total_dice / valid_steps,
        "iou": total_iou / valid_steps,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  CHECKPOINT MANAGEMENT
# ═════════════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    metrics: Dict[str, float],
    filepath: str,
):
    """
    Save model checkpoint with full training state and metadata.

    Checkpoint format is compatible with app.py inference loader:
      - model_state_dict: raw model weights (no wrapper prefix)
      - config: architecture/encoder metadata for auto-detection
      - metrics: val_dice, val_iou for reporting
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "metrics": metrics,
        # ── Metadata for app.py compatibility ──
        "architecture": ARCHITECTURE,
        "encoder": ENCODER,
        "config": {
            "architecture": ARCHITECTURE,
            "encoder": ENCODER,
            "image_size": IMAGE_SIZE,
            "in_channels": IN_CHANNELS,
            "classes": NUM_CLASSES,
        },
        # ── Convenience fields for app.py sidebar ──
        "val_dice": metrics.get("val_dice", 0.0),
        "val_iou": metrics.get("val_iou", 0.0),
    }

    torch.save(checkpoint, filepath)
    logger.info(f"  ✅ Checkpoint saved: {filepath} "
                f"(epoch {epoch}, val_dice={metrics.get('val_dice', 0):.4f})")


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN TRAINING FUNCTION
# ═════════════════════════════════════════════════════════════════════════════

def train_model():
    """
    Full training pipeline:
      1. Set seeds for reproducibility.
      2. Create data loaders with strict pairing.
      3. Build model + loss + optimizer + scheduler.
      4. Train with AMP + gradient accumulation.
      5. Validate and checkpoint on best Validation Dice.
      6. Early stopping if no improvement.
    """
    set_seed(SEED)

    logger.info("=" * 70)
    logger.info("MAMMOGRAPHY SEGMENTATION TRAINING PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Device          : {DEVICE}")
    logger.info(f"Architecture    : {ARCHITECTURE} + {ENCODER}")
    logger.info(f"Image size      : {IMAGE_SIZE}x{IMAGE_SIZE}")
    logger.info(f"Batch size      : {BATCH_SIZE} (effective: {BATCH_SIZE * ACCUM_STEPS})")
    logger.info(f"Epochs          : {EPOCHS}")
    logger.info(f"Learning rate   : {LR}")
    logger.info(f"Loss            : Dice({DICE_WEIGHT}) + BCE({BCE_WEIGHT}, pos_weight={BCE_POS_WEIGHT})")
    logger.info(f"Scheduler       : CosineAnnealingWarmRestarts(T0={COSINE_T0})")
    logger.info(f"Early stopping  : patience={PATIENCE}")
    logger.info(f"Save dir        : {SAVE_DIR}")
    logger.info("=" * 70)

    # ── Data ──
    train_loader, val_loader = create_dataloaders(batch_size=BATCH_SIZE)

    # ── Model ──
    model = build_model()

    # ── Loss ──
    criterion = DiceBCELoss(
        dice_weight=DICE_WEIGHT,
        bce_weight=BCE_WEIGHT,
        bce_pos_weight=BCE_POS_WEIGHT,
    )

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    # ── Scheduler ──
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=COSINE_T0,
        T_mult=COSINE_T_MULT,
    )

    # ── AMP scaler ──
    scaler = GradScaler("cuda", enabled=torch.cuda.is_available())

    # ── Training state ──
    best_val_dice = 0.0
    patience_counter = 0
    save_path = os.path.join(SAVE_DIR, "best_mg_model.pth")

    # ── Print header ──
    header = (
        f"{'Epoch':>5} │ {'Train Loss':>10} │ {'Val Loss':>8} │ "
        f"{'Val Dice':>8} │ {'Val IoU':>7} │ {'LR':>9} │ {'Time':>6}"
    )
    logger.info("")
    logger.info(header)
    logger.info("─" * len(header))

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        # ── Train ──
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            accum_steps=ACCUM_STEPS,
        )

        # ── Validate ──
        val_metrics = validate(model, val_loader, criterion)

        # ── Step scheduler ──
        scheduler.step()

        # ── Timing ──
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        # ── Log ──
        logger.info(
            f"{epoch:5d} │ {train_metrics['loss']:10.4f} │ {val_metrics['loss']:8.4f} │ "
            f"{val_metrics['dice']:8.4f} │ {val_metrics['iou']:7.4f} │ "
            f"{current_lr:9.2e} │ {epoch_time:5.1f}s"
        )

        # ── Checkpoint on best Validation Dice ──
        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            patience_counter = 0

            save_metrics = {
                "train_loss": train_metrics["loss"],
                "train_dice": train_metrics["dice"],
                "train_iou": train_metrics["iou"],
                "val_loss": val_metrics["loss"],
                "val_dice": val_metrics["dice"],
                "val_iou": val_metrics["iou"],
            }
            save_checkpoint(model, optimizer, scheduler, epoch, save_metrics, save_path)
        else:
            patience_counter += 1

        # ── Early stopping ──
        if patience_counter >= PATIENCE:
            logger.info(f"\n⛔ Early stopping triggered at epoch {epoch} "
                        f"(no improvement for {PATIENCE} epochs)")
            break

    # ── Training complete ──
    logger.info("")
    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info(f"  Best Validation Dice : {best_val_dice:.4f}")
    logger.info(f"  Checkpoint saved to  : {save_path}")
    logger.info("=" * 70)


# ═════════════════════════════════════════════════════════════════════════════
#  OOM-SAFE ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        train_model()
    except RuntimeError as exc:
        if torch.cuda.is_available() and "out of memory" in str(exc).lower():
            logger.error("=" * 70)
            logger.error("CUDA OUT OF MEMORY — Retrying with reduced batch size")
            logger.error("=" * 70)
            torch.cuda.empty_cache()

            # Halve batch size, double accumulation to maintain effective batch
            BATCH_SIZE_FALLBACK = max(1, BATCH_SIZE // 2)
            ACCUM_STEPS_FALLBACK = ACCUM_STEPS * 2

            logger.info(f"Fallback: batch_size={BATCH_SIZE_FALLBACK}, "
                        f"accum_steps={ACCUM_STEPS_FALLBACK}")

            # Override globals for retry
            import train_mg
            train_mg.BATCH_SIZE = BATCH_SIZE_FALLBACK
            train_mg.ACCUM_STEPS = ACCUM_STEPS_FALLBACK
            train_model()
        else:
            raise
