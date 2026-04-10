"""
train.py — BOSS Boundary-Precision Trainer  (Ultrasound / Mammography)
=======================================================================
Enhancements over baseline:
  1. Architecture  : UnetPlusPlus + MIT-B2 Transformer encoder
  2. Loss          : Focal + Boundary (distance-transform) + Dice combo
  3. Deep Supervision : auxiliary decoder outputs folded into loss
  4. TTA           : ttach-based test-time augmentation in validation
  5. Augmentation  : physics-informed US transforms (speckle, elastic, …)
  6. Post-processing: morphological cleanup + optional SimpleCRF

Author : Senior DL Engineer — Medical Imaging
"""

from __future__ import annotations

import os
import random
import argparse
import warnings
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from scipy.ndimage import distance_transform_edt
from skimage.morphology import remove_small_objects, binary_closing, disk

# Optional: ttach for TTA  (pip install ttach)
try:
    import ttach
    _TTACH_AVAILABLE = True
except ImportError:
    warnings.warn("ttach not installed — TTA disabled.  pip install ttach")
    _TTACH_AVAILABLE = False

# Optional: SimpleCRF  (pip install SimpleCRF)
try:
    import denseCRF                   # part of SimpleCRF package
    _CRF_AVAILABLE = True
except ImportError:
    _CRF_AVAILABLE = False

from dataset import MultiModalityDataset


# ══════════════════════════════════════════════
#  GLOBAL CONFIG
# ══════════════════════════════════════════════
torch.backends.cudnn.benchmark = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE    = 512
BATCH_SIZE  = 8
EPOCHS      = 80
INITIAL_LR  = 1e-4
VAL_SPLIT   = 0.15
SEED        = 42
PATIENCE    = 10
GRAD_CLIP   = 1.0
NUM_WORKERS = 4

# Loss weights
W_FOCAL     = 0.4   # focal loss weight
W_BOUNDARY  = 0.3   # boundary (distance) loss weight
W_DICE      = 0.3   # dice loss weight

# Deep supervision: auxiliary output scale factor
DS_WEIGHT   = 0.4   # weight for intermediate outputs vs. final output


# ══════════════════════════════════════════════
#  1.  ARCHITECTURE
# ══════════════════════════════════════════════

def build_model(mode: str) -> nn.Module:
    """
    Build a UnetPlusPlus with a Transformer encoder.

    ┌─────────────┬──────────────────────────────────────────────────────┐
    │  Modality   │  Encoder choice & rationale                          │
    ├─────────────┼──────────────────────────────────────────────────────┤
    │  US  (us)   │  mit_b2 — Mix Transformer B2: hierarchical           │
    │             │  self-attention captures long-range US tissue context │
    │             │  while still being compact (25 M params).            │
    ├─────────────┼──────────────────────────────────────────────────────┤
    │  MG  (mg)   │  tu-maxvit_tiny_tf_224 — Multi-Axis ViT: combines    │
    │             │  local window + dilated global attention; excellent   │
    │             │  for fine microcalcification detail at 512×512.      │
    └─────────────┴──────────────────────────────────────────────────────┘

    UnetPlusPlus (U²-Net style dense skip connections) reduces the
    semantic gap between encoder and decoder stages — critical for
    tight boundary reconstruction compared to plain U-Net.
    """
    encoder_map = {
        "us": "mit_b2",
        "mg": "tu-maxvit_tiny_tf_224",
    }
    encoder = encoder_map[mode]

    model = smp.UnetPlusPlus(
        encoder_name           = encoder,
        encoder_weights        = "imagenet",
        in_channels            = 1,
        classes                = 1,
        decoder_attention_type = "scse",
        # aux_params enables a classification head — we repurpose it as
        # the anchor for deep supervision hooks below.
        aux_params             = None,
    )
    return model


# ── Deep Supervision wrapper ──────────────────────────────────────────

class DeepSupervisionWrapper(nn.Module):
    """
    Wraps an SMP model and taps into the decoder's intermediate feature
    maps to produce auxiliary segmentation outputs at multiple scales.

    SMP UnetPlusPlus exposes `model.decoder.blocks` — a list of
    DecodeBlock objects.  We attach lightweight 1×1 conv heads to the
    last N blocks and upsample their outputs to the input resolution.
    """

    def __init__(self, base_model: nn.Module, num_aux: int = 3):
        super().__init__()
        self.model   = base_model
        self.num_aux = num_aux

        # Infer the channel count of each decoder stage by running a
        # dummy forward pass and inspecting intermediate activations.
        self._aux_heads = nn.ModuleList()
        self._hooks     : list = []
        self._feats     : list = []
        self._register_hooks()

    # ── Hook registration ─────────────────────────────────────────────

    def _register_hooks(self):
        """Attach forward hooks to the last `num_aux` decoder blocks."""
        decoder_blocks = list(self.model.decoder.blocks)
        tap_blocks     = decoder_blocks[-self.num_aux:]

        # Dummy pass to learn channel dims
        dummy = torch.zeros(1, 1, IMG_SIZE, IMG_SIZE)
        _tmp_feats: list = []

        def _tmp_hook(_, __, out):
            _tmp_feats.append(out)

        handles = [b.register_forward_hook(_tmp_hook) for b in tap_blocks]
        with torch.no_grad():
            self.model(dummy)
        for h in handles:
            h.remove()

        # Build one 1×1 conv head per tapped block
        for feat in _tmp_feats:
            c = feat.shape[1]
            self._aux_heads.append(nn.Conv2d(c, 1, kernel_size=1))

        # Register the real hooks
        def make_hook(idx):
            def _hook(_, __, out):
                self._feats.append((idx, out))
            return _hook

        for i, blk in enumerate(tap_blocks):
            self._hooks.append(blk.register_forward_hook(make_hook(i)))

    # ── Forward ──────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor):
        self._feats.clear()
        final_logit = self.model(x)          # (B,1,H,W)

        aux_logits = []
        for idx, feat in sorted(self._feats, key=lambda t: t[0]):
            head   = self._aux_heads[idx].to(feat.device)
            aux    = head(feat)              # (B,1,h,w)
            aux_up = F.interpolate(aux, size=x.shape[-2:], mode="bilinear", align_corners=False)
            aux_logits.append(aux_up)

        return final_logit, aux_logits       # list len == num_aux

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()


# ══════════════════════════════════════════════
#  2.  BOUNDARY-AWARE LOSS
# ══════════════════════════════════════════════

# ── 2a. Focal Loss ────────────────────────────

class FocalLoss(nn.Module):
    """
    Binary Focal Loss.
    FL(p) = −αt(1−pt)^γ log(pt)

    γ=2.0 down-weights easy negatives; α=0.25 compensates class imbalance.
    Operates on raw logits.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce    = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs  = torch.sigmoid(logits)
        pt     = targets * probs + (1 - targets) * (1 - probs)
        alpha  = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal  = alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


# ── 2b. Boundary / Distance-Transform Loss ───

def _compute_distance_map_batch(masks: torch.Tensor) -> torch.Tensor:
    """
    Given a binary mask tensor (B,1,H,W) compute the normalised
    distance transform for each sample — in NumPy then back to tensor.
    Returns a float32 tensor of the same shape.
    """
    masks_np = masks.squeeze(1).cpu().numpy().astype(bool)  # (B,H,W)
    dist_maps = []
    for m in masks_np:
        # Distance from each pixel to the nearest GT boundary pixel
        dt_in  = distance_transform_edt(m)         # inside
        dt_out = distance_transform_edt(~m)        # outside
        dt     = dt_in + dt_out                    # combined; 0 on boundary
        # Normalise [0,1] so pixels near the boundary get high weight
        dt_norm = 1.0 - (dt / (dt.max() + 1e-6))
        dist_maps.append(dt_norm)
    return torch.from_numpy(np.stack(dist_maps)).unsqueeze(1).float()  # (B,1,H,W)


class BoundaryLoss(nn.Module):
    """
    Boundary loss: BCE weighted by the distance-transform of the GT mask.
    Pixels close to the boundary are penalised more heavily, forcing the
    model to concentrate its gradients at the tumor edge.

    Reference: Kervadec et al., "Boundary loss for highly unbalanced
    segmentation", MIDL 2019.
    """

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dist_w = _compute_distance_map_batch(targets).to(logits.device)
        bce    = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        return (dist_w * bce).mean()


# ── 2c. Composite Boundary-Precision Loss ────

class BoundaryPrecisionLoss(nn.Module):
    """
    Focal(w=0.4) + BoundaryLoss(w=0.3) + DiceLoss(w=0.3)

    • Focal    → forces learning on hard / boundary pixels
    • Boundary → gradient pressure localised at contour
    • Dice     → ensures global overlap quality doesn't degrade
    """

    def __init__(
        self,
        w_focal   : float = W_FOCAL,
        w_boundary: float = W_BOUNDARY,
        w_dice    : float = W_DICE,
    ):
        super().__init__()
        self.w_focal    = w_focal
        self.w_boundary = w_boundary
        self.w_dice     = w_dice

        self.focal    = FocalLoss(alpha=0.25, gamma=2.0)
        self.boundary = BoundaryLoss()
        self.dice     = smp.losses.DiceLoss(mode="binary")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (
            self.w_focal    * self.focal(logits, targets)
            + self.w_boundary * self.boundary(logits, targets)
            + self.w_dice     * self.dice(logits, targets)
        )


# ══════════════════════════════════════════════
#  3.  TRANSFORMS
# ══════════════════════════════════════════════

def build_train_transforms_us(img_size: int = IMG_SIZE) -> A.Compose:
    """
    Physics-informed ultrasound augmentation.
    Key additions over baseline:
      - ElasticTransform  : tissue deformation
      - GridDistortion    : transducer-pressure artefacts
      - MultiplicativeNoise: gain/TGC variation (speckle)
      - GaussNoise        : thermal electronic noise
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        # ── Spatial ──────────────────────────────────────────────
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                           rotate_limit=15, p=0.4),
        # ── Physics-informed distortions ─────────────────────────
        A.OneOf([
            A.ElasticTransform(
                alpha=120, sigma=12,          # larger alpha → stronger deformation
                alpha_affine=10, p=1.0),
            A.GridDistortion(
                num_steps=5, distort_limit=0.2, p=1.0),
            A.OpticalDistortion(
                distort_limit=0.06, shift_limit=0.04, p=1.0),
        ], p=0.5),
        # ── Speckle / noise simulation ───────────────────────────
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 40.0), p=1.0),        # thermal noise
            A.MultiplicativeNoise(
                multiplier=(0.85, 1.15), per_channel=False, p=1.0),  # gain variation
        ], p=0.4),
        # ── Intensity / contrast ─────────────────────────────────
        A.RandomBrightnessContrast(
            brightness_limit=0.25, contrast_limit=0.25, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.CLAHE(clip_limit=2.5, tile_grid_size=(8, 8), p=0.3),
        # ── Normalize ────────────────────────────────────────────
        A.Normalize(mean=(0.485,), std=(0.229,)),
        ToTensorV2(),
    ])


def build_train_transforms_mg(img_size: int = IMG_SIZE) -> A.Compose:
    """Mammography augmentation — minimal distortion, edge-preserving."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                           rotate_limit=15, p=0.4),
        A.GridDistortion(num_steps=3, distort_limit=0.05, p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.15,
                                   contrast_limit=0.3, p=0.4),
        A.Sharpen(alpha=(0.1, 0.3), lightness=(0.8, 1.2), p=0.3),
        A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.3),
        A.Normalize(mean=(0.485,), std=(0.229,)),
        ToTensorV2(),
    ])


def build_val_transforms(img_size: int = IMG_SIZE) -> A.Compose:
    """Deterministic val/test pipeline — resize + normalise only."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485,), std=(0.229,)),
        ToTensorV2(),
    ])


# ══════════════════════════════════════════════
#  4.  TEST-TIME AUGMENTATION  (TTA)
# ══════════════════════════════════════════════

def tta_predict(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """
    Apply TTA using the `ttach` library when available.
    Augmentations: identity + H-flip + V-flip + H+V-flip.
    Returns averaged probability map (B,1,H,W).

    Falls back to a single forward pass when ttach is unavailable.
    """
    if not _TTACH_AVAILABLE:
        with torch.no_grad():
            logits = model(images)
            if isinstance(logits, tuple):   # DeepSupervisionWrapper
                logits = logits[0]
            return torch.sigmoid(logits)

    transforms = ttach.Compose([
        ttach.HorizontalFlip(),
        ttach.VerticalFlip(),
        ttach.Rotate90(angles=[0, 90, 180, 270]),
    ])

    tta_model  = ttach.SegmentationTTAWrapper(
        model         = model,
        transforms    = transforms,
        merge_mode    = "mean",
        output_mask_key= None,         # our model returns (logits, aux) tuple — handle below
    )

    # ttach doesn't natively handle tuple outputs — do it manually
    probs_list: list[torch.Tensor] = []
    with torch.no_grad():
        for transformer in transforms:
            aug_images = transformer.augment_image(images)
            out        = model(aug_images)
            logits     = out[0] if isinstance(out, tuple) else out
            probs      = torch.sigmoid(logits)
            deaug_prob = transformer.deaugment_mask(probs)
            probs_list.append(deaug_prob)

    return torch.stack(probs_list).mean(dim=0)


# ══════════════════════════════════════════════
#  5.  POST-PROCESSING
# ══════════════════════════════════════════════

def morphological_cleanup(
    mask_uint8    : np.ndarray,
    min_size      : int = 300,
    closing_radius: int = 3,
) -> np.ndarray:
    """
    1. Remove small false-positive islands (< min_size px).
    2. Binary closing to smooth contours and fill tiny holes.

    Args:
        mask_uint8      : Binary mask, dtype uint8, values {0,1}.
        min_size        : Min connected component area to keep.
        closing_radius  : Structuring element radius for closing.
    Returns:
        Cleaned binary mask, dtype uint8.
    """
    mask_bool  = mask_uint8.astype(bool)
    # Remove small isolated islands
    cleaned    = remove_small_objects(mask_bool, min_size=min_size)
    # Close small gaps / rough edges
    cleaned    = binary_closing(cleaned, footprint=disk(closing_radius))
    return cleaned.astype(np.uint8)


def crf_refine(
    image_gray: np.ndarray,
    prob_map  : np.ndarray,
    num_iters : int = 5,
) -> np.ndarray:
    """
    Dense CRF refinement via SimpleCRF (pydensecrf).
    Encourages boundary alignment with the image gradients.

    Falls back to the raw probability map if SimpleCRF is unavailable.

    Args:
        image_gray : uint8 grayscale (H,W).
        prob_map   : float32 probability map in [0,1] (H,W).
        num_iters  : CRF inference iterations.
    Returns:
        Refined probability map (H,W) float32.
    """
    if not _CRF_AVAILABLE:
        warnings.warn("SimpleCRF not installed — skipping CRF refinement.")
        return prob_map

    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral

    H, W = image_gray.shape
    # Convert to 3-channel (CRF expects RGB)
    img_rgb = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)

    d = dcrf.DenseCRF2D(W, H, 2)

    # Unary potential from our probability map
    probs     = np.stack([1 - prob_map, prob_map], axis=0).astype(np.float32)
    unary     = unary_from_softmax(probs)
    d.setUnaryEnergy(unary)

    # Pairwise: appearance (colour) term
    d.addPairwiseBilateral(
        sxy=(80, 80), srgb=(13, 13, 13),
        rgbim=img_rgb, compat=10
    )
    # Pairwise: smoothness term
    d.addPairwiseGaussian(sxy=(3, 3), compat=3)

    Q      = d.inference(num_iters)
    refined = np.array(Q)[1].reshape(H, W)
    return refined.astype(np.float32)


def postprocess(
    prob_map  : np.ndarray,
    threshold : float      = 0.5,
    image_gray: Optional[np.ndarray] = None,
    use_crf   : bool       = False,
) -> np.ndarray:
    """
    Full post-processing pipeline:
      1. Optional CRF refinement
      2. Threshold to binary mask
      3. Morphological cleanup
    """
    if use_crf and image_gray is not None:
        prob_map = crf_refine(image_gray, prob_map)

    mask = (prob_map > threshold).astype(np.uint8)
    mask = morphological_cleanup(mask)
    return mask


# ══════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def compute_metrics(preds_logits: torch.Tensor, masks: torch.Tensor,
                    threshold: float = 0.5):
    tp, fp, fn, tn = smp.metrics.get_stats(
        preds_logits, masks.long(),
        mode="binary", threshold=threshold,
    )
    dice = smp.metrics.f1_score( tp, fp, fn, tn, reduction="micro").item()
    iou  = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro").item()
    return dice, iou


class EarlyStopping:
    def __init__(self, patience: int = PATIENCE, min_delta: float = 1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best_loss = float("inf")
        self.triggered = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
        return self.triggered


# ══════════════════════════════════════════════
#  TRAIN ONE EPOCH  (with Deep Supervision)
# ══════════════════════════════════════════════

def train_one_epoch(
    model       : nn.Module,
    loader      : DataLoader,
    criterion   : nn.Module,
    optimizer   : torch.optim.Optimizer,
    scaler      : GradScaler,
    ds_weight   : float = DS_WEIGHT,
) -> float:
    """
    Training loop with Deep Supervision.

    The model (DeepSupervisionWrapper) returns:
        final_logit  : (B,1,H,W)  — main prediction
        aux_logits   : list[(B,1,H,W)]  — intermediate predictions

    Total loss = criterion(final) + ds_weight * mean(criterion(aux_i))
    """
    model.train()
    total_loss = 0.0

    for images, masks in loader:
        images, masks = images.to(DEVICE), masks.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        with autocast():
            final_logit, aux_logits = model(images)

            # ── Main loss ──────────────────────────────────────────
            loss_main = criterion(final_logit, masks)

            # ── Auxiliary (deep supervision) losses ────────────────
            if aux_logits:
                loss_aux = torch.stack([
                    criterion(aux, masks) for aux in aux_logits
                ]).mean()
            else:
                loss_aux = torch.tensor(0.0, device=DEVICE)

            loss = loss_main + ds_weight * loss_aux

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(loader)


# ══════════════════════════════════════════════
#  VALIDATION LOOP  (with TTA)
# ══════════════════════════════════════════════

def validate(
    model     : nn.Module,
    loader    : DataLoader,
    criterion : nn.Module,
    use_tta   : bool = True,
) -> tuple[float, float, float]:
    """
    Returns (val_loss, val_dice, val_iou).
    Uses TTA for metric computation; loss computed on single forward pass
    (TTA is expensive — acceptable trade-off).
    """
    model.eval()
    val_loss = val_dice = val_iou = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)

            # ── Loss: single forward pass ──────────────────────────
            with autocast():
                final_logit, _ = model(images)
                loss = criterion(final_logit, masks)
            val_loss += loss.item()

            # ── Metrics: TTA-averaged probabilities ────────────────
            probs_tta    = tta_predict(model, images)   # (B,1,H,W)  [0,1]
            logits_equiv = torch.log(probs_tta.clamp(1e-6, 1 - 1e-6)
                                     / (1 - probs_tta.clamp(1e-6, 1 - 1e-6)))
            d, i = compute_metrics(logits_equiv, masks)
            val_dice += d
            val_iou  += i

    n = len(loader)
    return val_loss / n, val_dice / n, val_iou / n


# ══════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════

def train_model(mode: str, use_tta: bool = True):
    set_seed(SEED)

    device_name = (torch.cuda.get_device_name(0)
                   if torch.cuda.is_available() else "CPU")
    print(f"\n🧬  BOSS Boundary-Precision Trainer  |  Mode: {mode.upper()}  |  Device: {device_name}\n")

    # ── Paths ────────────────────────────────
    data_dir  = f"data/processed_{mode}/train"
    save_path = f"models/{mode}_best.pth"
    os.makedirs("models", exist_ok=True)

    # ── Dataset ──────────────────────────────
    # We override the transform after construction so we can use our
    # enhanced mode-specific augmentation pipelines.
    full_dataset = MultiModalityDataset(
        image_dir = os.path.join(data_dir, "images"),
        mask_dir  = os.path.join(data_dir, "masks"),
        img_size  = IMG_SIZE,
        mode      = mode,
        is_train  = True,
    )

    # Inject the enhanced transforms
    full_dataset.transform = (
        build_train_transforms_us(IMG_SIZE)
        if mode == "us"
        else build_train_transforms_mg(IMG_SIZE)
    )

    n_total = len(full_dataset)
    n_val   = max(1, int(n_total * VAL_SPLIT))
    n_train = n_total - n_val

    train_ds, val_ds = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED),
    )

    # Validation subset must use deterministic transforms
    val_ds.dataset  = MultiModalityDataset(
        image_dir = os.path.join(data_dir, "images"),
        mask_dir  = os.path.join(data_dir, "masks"),
        img_size  = IMG_SIZE,
        mode      = mode,
        is_train  = False,
    )

    loader_kw    = dict(num_workers=NUM_WORKERS, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, **loader_kw)

    print(f"📦  Samples: {n_total} total → {n_train} train | {n_val} val\n")

    # ── Model ────────────────────────────────
    base_model = build_model(mode).to(DEVICE)
    model      = DeepSupervisionWrapper(base_model, num_aux=3).to(DEVICE)

    encoder_name = ("mit_b2" if mode == "us" else "tu-maxvit_tiny_tf_224")
    print(f"🏗️   Architecture : UnetPlusPlus + {encoder_name} + SCSE + DeepSupervision")

    # ── Loss ─────────────────────────────────
    criterion = BoundaryPrecisionLoss(
        w_focal=W_FOCAL, w_boundary=W_BOUNDARY, w_dice=W_DICE
    )
    print(f"🔥  Loss : Focal({W_FOCAL}) + Boundary({W_BOUNDARY}) + Dice({W_DICE})\n")

    # ── Optimizer & Scheduler ─────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=INITIAL_LR, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6,
    )

    scaler      = GradScaler()
    early_stop  = EarlyStopping(patience=PATIENCE)
    best_val    = float("inf")
    use_tta_flag = use_tta and _TTACH_AVAILABLE

    print(f"{'Epoch':>6} | {'Train':>10} | {'Val':>8} | {'Dice':>8} | {'IoU':>7} | {'LR':>10}"
          + ("  [TTA]" if use_tta_flag else ""))
    print("─" * 72)

    for epoch in range(1, EPOCHS + 1):

        train_loss            = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler
        )
        val_loss, val_dice, val_iou = validate(
            model, val_loader, criterion, use_tta=use_tta_flag
        )

        scheduler.step(epoch - 1)
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"{epoch:>6} | {train_loss:>10.4f} | {val_loss:>8.4f} | "
            f"{val_dice:>8.4f} | {val_iou:>7.4f} | {lr:>10.2e}"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "epoch"           : epoch,
                    "model_state"     : model.state_dict(),
                    "optimizer_state" : optimizer.state_dict(),
                    "val_loss"        : best_val,
                    "val_dice"        : val_dice,
                    "val_iou"         : val_iou,
                    "encoder"         : encoder_name,
                    "architecture"    : "UnetPlusPlus",
                },
                save_path,
            )
            print(f"  ⭐  Checkpoint → val_loss={best_val:.4f}  dice={val_dice:.4f}  iou={val_iou:.4f}")

        if early_stop.step(val_loss):
            print(f"\n⏹️  Early stop at epoch {epoch} (patience={PATIENCE}).")
            break

    model.remove_hooks()
    print(f"\n✅  Done.  Best val_loss: {best_val:.4f}  →  {save_path}\n")


# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BOSS Boundary-Precision Trainer")
    parser.add_argument("--mode", required=True, choices=["us", "mg"])
    parser.add_argument("--no-tta", action="store_true",
                        help="Disable TTA during validation")
    args = parser.parse_args()
    train_model(args.mode, use_tta=not args.no_tta)