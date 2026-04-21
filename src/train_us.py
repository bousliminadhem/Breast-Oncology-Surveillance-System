"""
train_us.py — Breast Ultrasound Segmentation Training Pipeline
================================================================
Production-grade training script for binary lesion segmentation on breast
ultrasound images, optimized for Google Colab T4 GPU (16 GB VRAM).

Architecture:  U-Net++ (UnetPlusPlus) with EfficientNet-B4 encoder (ImageNet)
Loss:          Tversky Loss (α=0.7, β=0.3) + BCE with pos_weight
               → Tversky α > β penalizes False Negatives more heavily,
                 which is critical in oncology (missing a tumor is worse
                 than a false alarm).
               → BCE pos_weight upweights the foreground class to handle
                 the extreme class imbalance (tiny lesion vs large background).
Activation:    Sigmoid (binary segmentation, output is logits → sigmoid).
               Both Tversky and BCEWithLogitsLoss operate on raw logits
               and apply sigmoid internally — mathematically coherent.
Metrics:       Dice Score, IoU (Jaccard), False Negative Rate (FNR)
Optimization:  AdamW + CosineAnnealingWarmRestarts + AMP + Gradient Accumulation
Checkpointing: Best model saved by Validation Dice score

Usage:
    python train_us.py
"""

from __future__ import annotations

import gc
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
from tqdm import tqdm
import segmentation_models_pytorch as smp

from dataset_us import (
    UltrasoundSegDataset,
    get_strict_paired_filenames,
    build_train_transforms,
    build_val_transforms,
    safe_collate,
    safety_check_masks,
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

# ── Data paths (Google Colab with Drive mounted) ──
DATA_ROOT = "/content/drive/MyDrive/Breast Oncology Surveillance System/data/data/processed_us"
TRAIN_IMAGE_DIR = os.path.join(DATA_ROOT, "train", "images")
TRAIN_MASK_DIR = os.path.join(DATA_ROOT, "train", "masks")
TEST_IMAGE_DIR = os.path.join(DATA_ROOT, "test", "images")
TEST_MASK_DIR = os.path.join(DATA_ROOT, "test", "masks")
SAVE_DIR = "/content/drive/MyDrive/Breast Oncology Surveillance System/models"

# ── Model ──
ARCHITECTURE = "UnetPlusPlus"
ENCODER = "efficientnet-b4"
ENCODER_WEIGHTS = "imagenet"
IN_CHANNELS = 1       # Grayscale ultrasound
NUM_CLASSES = 1        # Binary: lesion vs background

# ── Training ──
IMAGE_SIZE = 256       # Ultrasound images are typically smaller than mammography
BATCH_SIZE = 8         # Starting batch size for T4 (will auto-fallback)
ACCUM_STEPS = 2        # Effective batch = 8 × 2 = 16
EPOCHS = 100
LR = 3e-4
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.2        # 80/20 from train set (test set used for final evaluation)
SEED = 42
NUM_WORKERS = 2
PATIENCE = 25          # Early stopping patience

# ── Loss ──
TVERSKY_ALPHA = 0.7    # FN penalty weight (high → penalize missed tumors)
TVERSKY_BETA = 0.3     # FP penalty weight
BCE_POS_WEIGHT = 3.0   # Upweight foreground in BCE
TVERSKY_WEIGHT = 0.6   # Contribution of Tversky Loss
BCE_WEIGHT = 0.4       # Contribution of BCE

# ── Scheduler ──
COSINE_T0 = 20         # Restart period for CosineAnnealingWarmRestarts
COSINE_T_MULT = 2      # Multiply period after each restart

# ── Device ──
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═════════════════════════════════════════════════════════════════════════════
#  T4 VRAM MANAGEMENT — Auto Batch Size Calculator
# ═════════════════════════════════════════════════════════════════════════════

def validate_batch_size(
    model: nn.Module,
    target_batch: int = BATCH_SIZE,
    image_size: int = IMAGE_SIZE,
    in_channels: int = IN_CHANNELS,
) -> int:
    """
    Validate that the target batch size fits in GPU VRAM.

    Uses a fast DOWNWARD-ONLY approach: try the target batch size with a
    single dummy forward+backward pass. If OOM, halve and retry.
    At most 4 attempts — completes in seconds, not minutes.

    Args:
        model:          The segmentation model (already on DEVICE).
        target_batch:   Desired batch size to validate.
        image_size:     Spatial resolution of input images.
        in_channels:    Number of input channels.

    Returns:
        Largest feasible batch size ≤ target_batch (minimum 1).
    """
    if not torch.cuda.is_available():
        logger.info("No CUDA device — using batch_size=2 for CPU fallback.")
        return 2

    model.eval()
    test_batch = target_batch
    max_attempts = 4

    for attempt in range(max_attempts):
        if test_batch < 1:
            test_batch = 1

        try:
            torch.cuda.empty_cache()
            gc.collect()

            logger.info(f"  VRAM check: trying batch_size={test_batch}...")

            dummy_input = torch.randn(
                test_batch, in_channels, image_size, image_size,
                device=DEVICE, dtype=torch.float32,
            )
            dummy_target = torch.zeros(
                test_batch, 1, image_size, image_size,
                device=DEVICE, dtype=torch.float32,
            )

            with autocast("cuda"):
                output = model(dummy_input)
                loss = nn.functional.binary_cross_entropy_with_logits(output, dummy_target)
            loss.backward()

            # Success — clean up and return
            del dummy_input, dummy_target, output, loss
            torch.cuda.empty_cache()
            model.zero_grad(set_to_none=True)
            gc.collect()
            model.train()

            logger.info(f"  ✅ batch_size={test_batch} fits in VRAM.")
            return test_batch

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"  ⚠️  batch_size={test_batch} caused OOM. Halving...")
                # Clean up — some vars may not exist if OOM happened early
                for var_name in ['dummy_input', 'dummy_target', 'output', 'loss']:
                    if var_name in locals():
                        del locals()[var_name]
                torch.cuda.empty_cache()
                model.zero_grad(set_to_none=True)
                gc.collect()
                test_batch = max(1, test_batch // 2)
            else:
                raise

    # Final fallback
    torch.cuda.empty_cache()
    gc.collect()
    model.train()
    logger.info(f"  Fallback: using batch_size={test_batch}")
    return max(1, test_batch)


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
      - UnetPlusPlus: dense skip connections improve boundary precision,
        which is essential for lesion delineation in noisy ultrasound.
      - EfficientNet-B4: excellent accuracy/VRAM tradeoff for T4.
      - scse attention: channel + spatial squeeze-excitation in decoder.
      - in_channels=1: grayscale ultrasound input.
      - classes=1: binary segmentation (lesion vs background).
      - No final activation: model outputs raw logits (sigmoid applied in loss).
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
#  LOSS FUNCTION — Tversky + BCE (FN-Aware, Class-Imbalanced)
# ═════════════════════════════════════════════════════════════════════════════

class TverskyBCELoss(nn.Module):
    """
    Combined Tversky Loss + Binary Cross Entropy for class-imbalanced
    segmentation with emphasis on minimizing False Negatives.

    Mathematical coherence:
      - Both components operate on RAW LOGITS (before sigmoid).
      - smp.losses.TverskyLoss(from_logits=True) applies sigmoid internally.
      - nn.BCEWithLogitsLoss applies sigmoid internally.
      - The model's final layer outputs logits (no activation).
      → This is mathematically coherent: sigmoid is applied exactly once.

    Why Tversky over Dice:
      - Tversky(α=0.7, β=0.3) penalizes False Negatives 2.33× more than
        False Positives. In oncology, missing a tumor (FN) is catastrophic.
      - Standard Dice treats FP and FN equally (α=β=0.5).

    Why add BCE:
      - BCE provides pixel-level gradients for stable early training.
      - pos_weight further upweights the minority class (tumor).
    """

    def __init__(
        self,
        tversky_alpha: float = TVERSKY_ALPHA,
        tversky_beta: float = TVERSKY_BETA,
        tversky_weight: float = TVERSKY_WEIGHT,
        bce_weight: float = BCE_WEIGHT,
        bce_pos_weight: float = BCE_POS_WEIGHT,
    ):
        super().__init__()
        self.tversky_weight = tversky_weight
        self.bce_weight = bce_weight

        self.tversky = smp.losses.TverskyLoss(
            mode="binary",
            from_logits=True,
            alpha=tversky_alpha,
            beta=tversky_beta,
            smooth=1.0,
        )

        pos_weight = torch.tensor([bce_pos_weight], device=DEVICE)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute combined loss from raw logits.

        Args:
            logits:  Model output (B, 1, H, W), raw logits.
            targets: Ground truth binary masks (B, 1, H, W), float {0, 1}.
        """
        assert logits.shape == targets.shape, \
            f"Logits shape {logits.shape} != targets shape {targets.shape}"

        tversky_loss = self.tversky(logits, targets)
        bce_loss = self.bce(logits, targets)
        return self.tversky_weight * tversky_loss + self.bce_weight * bce_loss


# ═════════════════════════════════════════════════════════════════════════════
#  METRICS — Dice, IoU, False Negative Rate
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> Dict[str, float]:
    """
    Compute Dice, IoU, and False Negative Rate from model logits.

    False Negative Rate = FN / (TP + FN)
      → The fraction of actual tumor pixels that the model MISSED.
      → Lower is better. In oncology, this is the most critical metric.

    Args:
        logits:    Raw model output (B, 1, H, W) — before sigmoid.
        targets:   Ground truth binary masks (B, 1, H, W).
        threshold: Binarization threshold for predictions.
        eps:       Smoothing for numerical stability.

    Returns:
        Dict with 'dice', 'iou', 'fnr' (batch-averaged).
    """
    preds = (torch.sigmoid(logits) > threshold).float()

    preds_flat = preds.view(preds.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)

    # True Positives, False Negatives, False Positives
    tp = (preds_flat * targets_flat).sum(dim=1)
    fn = ((1 - preds_flat) * targets_flat).sum(dim=1)
    fp = (preds_flat * (1 - targets_flat)).sum(dim=1)

    # Dice = 2*TP / (2*TP + FP + FN)
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)

    # IoU = TP / (TP + FP + FN)
    iou = (tp + eps) / (tp + fp + fn + eps)

    # False Negative Rate = FN / (TP + FN)
    # Only compute for samples that have actual positive pixels
    total_positives = tp + fn
    fnr = torch.where(
        total_positives > 0,
        fn / (total_positives + eps),
        torch.zeros_like(fn),
    )

    return {
        "dice": dice.mean().item(),
        "iou": iou.mean().item(),
        "fnr": fnr.mean().item(),
    }


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
    n_val = min(n_val, len(shuffled) - 1)

    val_files = shuffled[:n_val]
    train_files = shuffled[n_val:]

    logger.info(
        f"Train/Val split: {len(train_files)} train, {len(val_files)} val "
        f"({val_split*100:.0f}% val)"
    )
    return train_files, val_files


def create_dataloaders(
    batch_size: int = BATCH_SIZE,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders from the train/ split.

    Steps:
      1. Strict filename intersection (Zero Mismatch Policy).
      2. Deterministic train/val split.
      3. Separate augmentation pipelines.
    """
    # ── Step 1: Strict pairing on train data ──
    paired_files = get_strict_paired_filenames(TRAIN_IMAGE_DIR, TRAIN_MASK_DIR)

    # ── Step 2: Train/Val split ──
    train_files, val_files = split_filenames(paired_files)

    # ── Step 3: Build transforms ──
    train_transform = build_train_transforms(IMAGE_SIZE)
    val_transform = build_val_transforms(IMAGE_SIZE)

    # ── Step 4: Build datasets ──
    train_ds = UltrasoundSegDataset(
        image_dir=TRAIN_IMAGE_DIR,
        mask_dir=TRAIN_MASK_DIR,
        filenames=train_files,
        transform=train_transform,
        image_size=IMAGE_SIZE,
    )

    val_ds = UltrasoundSegDataset(
        image_dir=TRAIN_IMAGE_DIR,
        mask_dir=TRAIN_MASK_DIR,
        filenames=val_files,
        transform=val_transform,
        image_size=IMAGE_SIZE,
    )

    # ── Step 5: DataLoaders ──
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

    logger.info(
        f"DataLoaders ready: "
        f"train={len(train_ds)} samples ({len(train_loader)} batches), "
        f"val={len(val_ds)} samples ({len(val_loader)} batches)"
    )
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
    epoch: int,
    accum_steps: int = ACCUM_STEPS,
) -> Dict[str, float]:
    """
    Train for one epoch with AMP, gradient accumulation, and tqdm progress bar.

    Returns:
        Dict with 'loss', 'dice', 'iou', 'fnr' averaged over the epoch.
    """
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_fnr = 0.0
    valid_steps = 0

    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(
        enumerate(loader),
        total=len(loader),
        desc=f"  Train Epoch {epoch}",
        leave=False,
        ncols=120,
    )

    for step, (images, masks) in pbar:
        if images.numel() == 0:
            continue

        images = images.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)

        # ── Shape assertions ──
        assert images.ndim == 4 and images.shape[1] == IN_CHANNELS, \
            f"Expected images (B, {IN_CHANNELS}, H, W), got {images.shape}"
        assert masks.ndim == 4 and masks.shape[1] == NUM_CLASSES, \
            f"Expected masks (B, {NUM_CLASSES}, H, W), got {masks.shape}"

        # ── Forward pass with AMP ──
        with autocast("cuda", enabled=torch.cuda.is_available()):
            logits = model(images)

            assert logits.shape == masks.shape, \
                f"Model output shape {logits.shape} != mask shape {masks.shape}"

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
            metrics = compute_metrics(logits, masks)
            total_dice += metrics["dice"]
            total_iou += metrics["iou"]
            total_fnr += metrics["fnr"]

        # ── Update progress bar ──
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "dice": f"{metrics['dice']:.3f}",
            "fnr": f"{metrics['fnr']:.3f}",
        })

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
        "fnr": total_fnr / valid_steps,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  VALIDATION LOOP
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    epoch: int,
) -> Dict[str, float]:
    """
    Validate the model on the validation set with tqdm progress bar.

    Returns:
        Dict with 'loss', 'dice', 'iou', 'fnr' averaged over the validation set.
    """
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_fnr = 0.0
    valid_steps = 0

    pbar = tqdm(
        loader,
        total=len(loader),
        desc=f"    Val Epoch {epoch}",
        leave=False,
        ncols=120,
    )

    for images, masks in pbar:
        if images.numel() == 0:
            continue

        images = images.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)

        with autocast("cuda", enabled=torch.cuda.is_available()):
            logits = model(images)
            loss = criterion(logits, masks)

        total_loss += loss.item()
        metrics = compute_metrics(logits, masks)
        total_dice += metrics["dice"]
        total_iou += metrics["iou"]
        total_fnr += metrics["fnr"]
        valid_steps += 1

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "dice": f"{metrics['dice']:.3f}",
        })

    if valid_steps == 0:
        raise RuntimeError("Validation loader produced no valid batches.")

    return {
        "loss": total_loss / valid_steps,
        "dice": total_dice / valid_steps,
        "iou": total_iou / valid_steps,
        "fnr": total_fnr / valid_steps,
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

    Compatible with app.py inference loader.
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
        "val_dice": metrics.get("val_dice", 0.0),
        "val_iou": metrics.get("val_iou", 0.0),
        "val_fnr": metrics.get("val_fnr", 0.0),
    }

    torch.save(checkpoint, filepath)
    logger.info(
        f"  ✅ Checkpoint saved: {filepath} "
        f"(epoch {epoch}, val_dice={metrics.get('val_dice', 0):.4f}, "
        f"val_fnr={metrics.get('val_fnr', 0):.4f})"
    )


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN TRAINING FUNCTION
# ═════════════════════════════════════════════════════════════════════════════

def train_model():
    """
    Full training pipeline:
      1. Set seeds for reproducibility.
      2. Run safety check on masks.
      3. Create data loaders with strict pairing.
      4. Build model + loss + optimizer + scheduler.
      5. Auto-estimate batch size or use default with OOM fallback.
      6. Train with AMP + gradient accumulation + tqdm.
      7. Validate and checkpoint on best Validation Dice.
      8. Early stopping if no improvement.
    """
    set_seed(SEED)

    logger.info("=" * 70)
    logger.info("BREAST ULTRASOUND SEGMENTATION TRAINING PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Device          : {DEVICE}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info(f"GPU             : {gpu_name} ({gpu_mem:.1f} GB)")
    logger.info(f"Architecture    : {ARCHITECTURE} + {ENCODER}")
    logger.info(f"Image size      : {IMAGE_SIZE}x{IMAGE_SIZE}")
    logger.info(f"Epochs          : {EPOCHS}")
    logger.info(f"Learning rate   : {LR}")
    logger.info(f"Loss            : Tversky(α={TVERSKY_ALPHA}, β={TVERSKY_BETA}, "
                f"w={TVERSKY_WEIGHT}) + BCE(w={BCE_WEIGHT}, pos_weight={BCE_POS_WEIGHT})")
    logger.info(f"Scheduler       : CosineAnnealingWarmRestarts(T0={COSINE_T0})")
    logger.info(f"Early stopping  : patience={PATIENCE}")
    logger.info(f"Save dir        : {SAVE_DIR}")
    logger.info("=" * 70)

    # ── Step 1: Safety check on training masks ──
    logger.info("\n📋 Running mask safety check...")
    safety_check_masks(TRAIN_MASK_DIR, num_samples=15)

    # ── Step 2: Build model first (needed for batch size estimation) ──
    model = build_model()

    # ── Step 3: Estimate or set batch size ──
    batch_size = BATCH_SIZE
    accum_steps = ACCUM_STEPS

    if torch.cuda.is_available():
        validated_batch = validate_batch_size(
            model,
            target_batch=batch_size,
            image_size=IMAGE_SIZE,
            in_channels=IN_CHANNELS,
        )
        # Use the validated batch size (may be smaller if OOM)
        if validated_batch < batch_size:
            # Increase accumulation to compensate
            accum_steps = max(1, (batch_size * accum_steps) // validated_batch)
            batch_size = validated_batch
            logger.info(
                f"Adjusted for T4: batch_size={batch_size}, "
                f"accum_steps={accum_steps} "
                f"(effective={batch_size * accum_steps})"
            )
        else:
            logger.info(
                f"Batch size OK: {batch_size} "
                f"(effective={batch_size * accum_steps})"
            )

    # ── Step 4: Create dataloaders ──
    train_loader, val_loader = create_dataloaders(batch_size=batch_size)

    # ── Step 5: Loss ──
    criterion = TverskyBCELoss(
        tversky_alpha=TVERSKY_ALPHA,
        tversky_beta=TVERSKY_BETA,
        tversky_weight=TVERSKY_WEIGHT,
        bce_weight=BCE_WEIGHT,
        bce_pos_weight=BCE_POS_WEIGHT,
    )

    # ── Step 6: Optimizer ──
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    # ── Step 7: Scheduler ──
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=COSINE_T0,
        T_mult=COSINE_T_MULT,
    )

    # ── Step 8: AMP scaler ──
    scaler = GradScaler("cuda", enabled=torch.cuda.is_available())

    # ── Training state ──
    best_val_dice = 0.0
    patience_counter = 0
    save_path = os.path.join(SAVE_DIR, "best_us_model.pth")

    # ── Print header ──
    header = (
        f"{'Epoch':>5} │ {'TrnLoss':>8} │ {'ValLoss':>8} │ "
        f"{'ValDice':>8} │ {'ValIoU':>7} │ {'ValFNR':>7} │ "
        f"{'LR':>9} │ {'Time':>6}"
    )
    logger.info("")
    logger.info(header)
    logger.info("─" * len(header))

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        # ── Train ──
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            epoch=epoch, accum_steps=accum_steps,
        )

        # ── Validate ──
        val_metrics = validate(model, val_loader, criterion, epoch=epoch)

        # ── Step scheduler ──
        scheduler.step()

        # ── Timing ──
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        # ── Log ──
        logger.info(
            f"{epoch:5d} │ {train_metrics['loss']:8.4f} │ "
            f"{val_metrics['loss']:8.4f} │ {val_metrics['dice']:8.4f} │ "
            f"{val_metrics['iou']:7.4f} │ {val_metrics['fnr']:7.4f} │ "
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
                "train_fnr": train_metrics["fnr"],
                "val_loss": val_metrics["loss"],
                "val_dice": val_metrics["dice"],
                "val_iou": val_metrics["iou"],
                "val_fnr": val_metrics["fnr"],
            }
            save_checkpoint(
                model, optimizer, scheduler, epoch, save_metrics, save_path
            )
        else:
            patience_counter += 1

        # ── Early stopping ──
        if patience_counter >= PATIENCE:
            logger.info(
                f"\n⛔ Early stopping at epoch {epoch} "
                f"(no improvement for {PATIENCE} epochs)"
            )
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
            gc.collect()

            # Halve batch size, double accumulation to maintain effective batch
            import train_us
            train_us.BATCH_SIZE = max(1, BATCH_SIZE // 2)
            train_us.ACCUM_STEPS = ACCUM_STEPS * 2

            logger.info(
                f"Fallback: batch_size={train_us.BATCH_SIZE}, "
                f"accum_steps={train_us.ACCUM_STEPS}"
            )
            train_model()
        else:
            raise
