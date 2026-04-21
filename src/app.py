"""
app.py — BOSS Streamlit Inference App  (Boundary-Precision Edition)
=====================================================================
Compatible with train.py's checkpoints:
    • Architectures : smp.MAnet (US) and/or smp.UnetPlusPlus (MG)
    • Encoders      : mit_b2 (US) | tu-maxvit_tiny_tf_224 (MG)
    • State dict    : keys optionally prefixed with "model." (DeepSupervisionWrapper)
    • Normalisation : A.Normalize(mean=0.485, std=0.229) — single channel
    • Output        : tuple (final_logit, aux_logits) — only final used
    • Post-proc     : CRF refinement  +  morphological cleanup

Author: Senior Medical Imaging Software Engineer
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import warnings
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
import streamlit as st
import segmentation_models_pytorch as smp
from skimage.morphology import remove_small_objects, binary_closing, disk

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Optional: pydensecrf  (pip install pydensecrf) ───────────────────────────
try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax
    _CRF_AVAILABLE = True
except ImportError:
    _CRF_AVAILABLE = False
    warnings.warn(
        "pydensecrf not installed — CRF refinement will be skipped.\n"
        "  Install with: pip install pydensecrf"
    )


# ═════════════════════════════════════════════
#  PAGE CONFIG
# ═════════════════════════════════════════════
st.set_page_config(page_title="BOSS — Oncology AI", layout="wide")
st.title("🎗️ Breast Oncology Surveillance System (BOSS)")
st.caption(
    "UnetPlusPlus · Transformer Encoder · Boundary-Precision Post-Processing"
)

# ═════════════════════════════════════════════
#  SIDEBAR SETTINGS
# ═════════════════════════════════════════════
st.sidebar.header("⚙️ Settings")

MODE = st.sidebar.selectbox("Select Modality", ["Ultrasound (US)", "Mammography (MG)"])
THRESHOLD = st.sidebar.slider(
    "Segmentation Threshold", min_value=0.1, max_value=0.9,
    value=0.5, step=0.05,
    help="Pixels with probability above this value are classified as tumour."
)

st.sidebar.markdown("---")
USE_CRF = st.sidebar.checkbox(
    "🔬 CRF Boundary Refinement",
    value=_CRF_AVAILABLE,
    disabled=not _CRF_AVAILABLE,
    help="Dense CRF aligns the mask boundary to image gradients. Requires pydensecrf."
)
if not _CRF_AVAILABLE:
    st.sidebar.info("pydensecrf not installed. CRF disabled.")

MORPH_MIN_SIZE = st.sidebar.number_input(
    "Morphology — Min Region Size (px)", min_value=0, max_value=2000,
    value=300, step=50,
    help="Removes spurious connected components smaller than this area."
)
MORPH_RADIUS = st.sidebar.number_input(
    "Morphology — Closing Radius (px)", min_value=1, max_value=10,
    value=3, step=1,
    help="Structuring-element radius for binary closing (gap fill / contour smoothing)."
)


# ═════════════════════════════════════════════
#  MODEL CONFIGURATION  (matches train.py)
# ═════════════════════════════════════════════
_MODE_CONFIG = {
    # NOTE: "architecture" here is only a default.  If the checkpoint
    # contains an "architecture" field (e.g. "MAnet+mit_b2") it will
    # be parsed and will override these values to ensure an exact
    # match with the training-time model.
    "Ultrasound (US)": {
        "encoder"      : "mit_b2",
        "architecture" : "MAnet",
        "model_path"   : r"D:\Projet Fédérateur\AI\Breast Oncology Surveillance System\models\us_model.pth",
    },
    "Mammography (MG)": {
        "encoder"      : "efficientnet-b4",
        "architecture" : "UnetPlusPlus",
        "model_path"   : r"D:\Projet Fédérateur\AI\Breast Oncology Surveillance System\models\mg_model.pth",
    },
}

ENCODER    = _MODE_CONFIG[MODE]["encoder"]
MODEL_PATH = _MODE_CONFIG[MODE]["model_path"]

# ─── Resolution: 512px for both MG and US (mirrors train_mg.py / train.py)
IMG_SIZE = 512


# ═════════════════════════════════════════════
#  INFERENCE TRANSFORM  (must match training val pipeline EXACTLY)
# ═════════════════════════════════════════════

# MG normalization constants (from dataset_mg.py)
_MG_MEAN = (0.5,)
_MG_STD  = (0.25,)

# US normalization constants (from dataset.py / train.py)
_US_MEAN = (0.485,)
_US_STD  = (0.229,)


def _apply_clahe(image_gray: np.ndarray, clip_limit: float = 2.0, grid: int = 8) -> np.ndarray:
    """Apply CLAHE contrast enhancement (mirrors dataset_mg.py preprocessing)."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid, grid))
    if image_gray.dtype != np.uint8:
        image_gray = image_gray.astype(np.uint8)
    return clahe.apply(image_gray)


def _percentile_normalize(image: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    """Percentile-based normalization to [0, 1] (mirrors dataset_mg.py)."""
    p_low  = np.percentile(image, low)
    p_high = np.percentile(image, high)
    if p_high - p_low < 1e-6:
        return np.zeros_like(image, dtype=np.float32)
    clipped = np.clip(image, p_low, p_high)
    return ((clipped - p_low) / (p_high - p_low)).astype(np.float32)


def _get_infer_transform(mode: str) -> A.Compose:
    """Build the inference transform matching the training val pipeline."""
    if mode == "mg":
        # MG: dataset_mg.py applies CLAHE + percentile norm BEFORE augmentations,
        # then A.Normalize(mean=0.5, std=0.25, max_pixel_value=1.0)
        return A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(mean=_MG_MEAN, std=_MG_STD, max_pixel_value=1.0),
            ToTensorV2(),
        ])
    else:
        # US: original pipeline
        return A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(mean=_US_MEAN, std=_US_STD),
            ToTensorV2(),
        ])


# ═════════════════════════════════════════════
#  ARCHITECTURE HELPERS
# ════════════════════════════════════════════

def _build_base_model(architecture: str, encoder_name: str) -> nn.Module:
    """Rebuild the bare SMP model to match the training checkpoint.

    The checkpoint's ``architecture`` field may look like
    ``"MAnet+mit_b2"``, ``"Unet+mit_b3"`` or
    ``"UnetPlusPlus+tu-maxvit_tiny_tf_224"``.
    This helper receives the parsed architecture name and encoder
    string, and instantiates the correct SMP class.
    """
    arch_norm = architecture.lower()

    if arch_norm in {"unetplusplus", "unet++", "unet_plus_plus"}:
        ModelCls = smp.UnetPlusPlus
        extra_kwargs = {"decoder_attention_type": "scse"}
    elif arch_norm in {"unet", "u-net"}:
        ModelCls = smp.Unet
        extra_kwargs = {"decoder_attention_type": "scse"}
    elif arch_norm in {"manet", "ma-net"}:
        ModelCls = smp.MAnet
        extra_kwargs = {}
    else:
        # Fallback: default to UnetPlusPlus but surface a warning via Streamlit.
        st.warning(f"Unknown architecture '{architecture}'. Falling back to UnetPlusPlus.")
        ModelCls = smp.UnetPlusPlus
        extra_kwargs = {"decoder_attention_type": "scse"}

    return ModelCls(
        encoder_name    = encoder_name,
        encoder_weights = None,   # trained weights loaded from checkpoint
        in_channels     = 1,
        classes         = 1,
        aux_params      = None,
        **extra_kwargs,
    )


def _strip_ds_prefix(state_dict: dict) -> dict:
    """
    The checkpoint was saved from a DeepSupervisionWrapper whose
    `self.model = base_model`.  That means every key is prefixed with
    "model." (e.g. "model.encoder.layer1.weight").

    This function strips that prefix so the weights can be loaded
    directly into the bare UnetPlusPlus.  Keys that do NOT start with
    "model." (e.g. the auxiliary conv heads "_aux_heads.*") are dropped
    because they have no corresponding layer in the base model.
    """
    new_sd: dict = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            new_sd[key[len("model."):]] = value
        # Auxiliary head weights (_aux_heads.*) are intentionally skipped
    return new_sd


def _resolve_checkpoint_path(configured_path: str, mode_key: str) -> str:
    """Resolve a checkpoint path with safe fallbacks if the exact file is missing."""
    if os.path.isfile(configured_path):
        return configured_path

    configured = Path(configured_path)
    models_dir = configured.parent
    if not models_dir.is_dir():
        raise FileNotFoundError(
            f"Model directory not found: {models_dir}"
        )

    all_pth = sorted(models_dir.glob("*.pth"))
    if not all_pth:
        raise FileNotFoundError(
            f"No checkpoint files (.pth) found in {models_dir}"
        )

    def _norm_name(name: str) -> str:
        # Normalize spacing/case so names like "mg _best.pth" match "mg_best.pth".
        return "".join(name.lower().split())

    wanted = _norm_name(configured.name)
    for candidate in all_pth:
        if _norm_name(candidate.name) == wanted:
            return str(candidate)

    if mode_key == "Mammography (MG)":
        tagged = [p for p in all_pth if "mg" in p.name.lower()]
    else:
        tagged = [p for p in all_pth if "us" in p.name.lower()]

    if tagged:
        # Prefer the newest file when several checkpoints are available.
        tagged.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return str(tagged[0])

    available = ", ".join(p.name for p in all_pth)
    raise FileNotFoundError(
        f"Checkpoint not found: {configured_path}. Available: {available}"
    )


def _infer_architecture_from_state_dict(state_dict: dict) -> Optional[str]:
    """Infer segmentation architecture from checkpoint parameter names."""
    keys = state_dict.keys()

    if any("decoder.blocks.x_0_0" in k for k in keys):
        return "UnetPlusPlus"

    if any(k.startswith("decoder.pab") for k in keys) or any(".pab." in k for k in keys):
        return "MAnet"

    if any(k.startswith("decoder.blocks.") for k in keys):
        return "Unet"

    return None


def _infer_encoder_candidates_from_state_dict(state_dict: dict) -> list[str]:
    """Infer likely encoder names from state_dict key patterns."""
    keys = list(state_dict.keys())
    candidates: list[str] = []

    if any(k.startswith("encoder.model.stem.") for k in keys):
        candidates.append("tu-maxvit_tiny_tf_224")

    if any(k.startswith("encoder.patch_embed1.") for k in keys):
        candidates.extend(["mit_b3", "mit_b2", "mit_b1", "mit_b4", "mit_b5"])

    if any(k.startswith("encoder._conv_stem") for k in keys):
        block_ids = []
        for k in keys:
            m = re.match(r"encoder\._blocks\.(\d+)\.", k)
            if m:
                block_ids.append(int(m.group(1)))

        block_count = (max(block_ids) + 1) if block_ids else 0
        if block_count >= 39:
            eff_priority = ["efficientnet-b5", "efficientnet-b6", "efficientnet-b7", "efficientnet-b4"]
        elif block_count >= 32:
            eff_priority = ["efficientnet-b4", "efficientnet-b3", "efficientnet-b5", "efficientnet-b2"]
        elif block_count >= 23:
            eff_priority = ["efficientnet-b2", "efficientnet-b1", "efficientnet-b3", "efficientnet-b0"]
        else:
            eff_priority = ["efficientnet-b0", "efficientnet-b1", "efficientnet-b2", "efficientnet-b3"]
        candidates.extend(eff_priority)

    # Deduplicate while preserving order.
    deduped: list[str] = []
    for c in candidates:
        if c not in deduped:
            deduped.append(c)
    return deduped


def _load_with_best_match(
    state_dict: dict,
    arch_name: str,
    encoder_name: str,
    mode: str,
) -> tuple[nn.Module, str, str]:
    """Try a small set of architecture/encoder candidates and return the first strict match."""
    inferred_arch = _infer_architecture_from_state_dict(state_dict)
    inferred_encs = _infer_encoder_candidates_from_state_dict(state_dict)

    arch_candidates = [arch_name]
    if inferred_arch and inferred_arch not in arch_candidates:
        arch_candidates.append(inferred_arch)

    if mode == "mg":
        for a in ["Unet", "UnetPlusPlus", "MAnet"]:
            if a not in arch_candidates:
                arch_candidates.append(a)
    else:
        for a in ["MAnet", "UnetPlusPlus", "Unet"]:
            if a not in arch_candidates:
                arch_candidates.append(a)

    enc_candidates = [encoder_name]
    for e in inferred_encs:
        if e not in enc_candidates:
            enc_candidates.append(e)

    if mode == "mg":
        for e in ["efficientnet-b4", "efficientnet-b3", "tu-maxvit_tiny_tf_224", "efficientnet-b5", "efficientnet-b0"]:
            if e not in enc_candidates:
                enc_candidates.append(e)
    else:
        for e in ["mit_b2", "mit_b3"]:
            if e not in enc_candidates:
                enc_candidates.append(e)

    errors: list[str] = []
    for arch_try in arch_candidates:
        for enc_try in enc_candidates:
            try:
                model_try = _build_base_model(arch_try, enc_try)
                model_try.load_state_dict(state_dict, strict=True)
                return model_try, arch_try, enc_try
            except Exception as exc:
                errors.append(f"{arch_try}+{enc_try}: {exc}")

    details = "\n".join(errors[:5])
    raise RuntimeError(
        "Could not match checkpoint weights to any tested architecture/encoder combination.\n"
        f"First attempts:\n{details}"
    )


# ═════════════════════════════════════════════
#  MODEL LOADER  (cached)
# ═════════════════════════════════════════════

def load_boss_model(mode):
    try:
        # 1. Resolve mode-specific config
        mode_key = "Ultrasound (US)" if mode == "us" else "Mammography (MG)"
        cfg      = _MODE_CONFIG[mode_key]

        # 2. Load checkpoint first so we can infer architecture/encoder
        checkpoint_path = _resolve_checkpoint_path(cfg["model_path"], mode_key)
        if checkpoint_path != cfg["model_path"]:
            st.sidebar.warning(
                f"Configured checkpoint not found. Using fallback: {os.path.basename(checkpoint_path)}"
            )
        checkpoint      = torch.load(checkpoint_path, map_location=DEVICE)

        # 3. Derive architecture and encoder from checkpoint metadata
        arch_name    = cfg.get("architecture", "UnetPlusPlus")
        encoder_name = cfg.get("encoder", "mit_b2")

        if isinstance(checkpoint, dict):
            # ── train_mg.py format: config sub-dict ──────────────────────
            ckpt_config = checkpoint.get("config", {})
            if isinstance(ckpt_config, dict):
                if ckpt_config.get("architecture"):
                    arch_name = ckpt_config["architecture"]
                if ckpt_config.get("encoder"):
                    encoder_name = ckpt_config["encoder"]

            # ── Legacy format: top-level architecture/encoder fields ─────
            arch_field = checkpoint.get("architecture")
            enc_field  = checkpoint.get("encoder")

            # Parse patterns like "MAnet+mit_b2" or "Unet+mit_b3"
            if arch_field:
                if "+" in arch_field:
                    base_arch, maybe_enc = arch_field.split("+", 1)
                    arch_name = base_arch.strip() or arch_name
                    if maybe_enc.strip():
                        encoder_name = maybe_enc.strip()
                else:
                    arch_name = arch_field.strip() or arch_name

            # If checkpoint carries a non-empty explicit encoder field, prefer it.
            if enc_field:
                encoder_name = enc_field

        # ── Extract state dict (support multiple checkpoint formats) ──────
        if isinstance(checkpoint, dict):
            if "ema_state_dict" in checkpoint:
                # Prefer EMA weights (smoother, better generalisation)
                raw_sd = checkpoint["ema_state_dict"]
            elif "model_state_dict" in checkpoint:
                raw_sd = checkpoint["model_state_dict"]
            elif "model_state" in checkpoint:
                raw_sd = checkpoint["model_state"]
            else:
                raw_sd = checkpoint           # legacy: entire dict is state_dict
        else:
            raw_sd = checkpoint

        # ── Strip the DeepSupervisionWrapper prefix ──────────────────────
        is_wrapped = any(k.startswith("model.") for k in raw_sd)
        state_dict = _strip_ds_prefix(raw_sd) if is_wrapped else raw_sd

        # 4. Build and load with strict matching, trying compatible candidates.
        model, loaded_arch, loaded_encoder = _load_with_best_match(
            state_dict=state_dict,
            arch_name=arch_name,
            encoder_name=encoder_name,
            mode=mode,
        )

        if loaded_arch != arch_name or loaded_encoder != encoder_name:
            st.sidebar.info(
                f"Auto-matched checkpoint to {loaded_arch}+{loaded_encoder}"
            )

        model.to(DEVICE).eval()

        # Surface checkpoint metadata for transparency
        if isinstance(checkpoint, dict):
            arch    = checkpoint.get("architecture", loaded_arch)
            enc     = checkpoint.get("encoder",      loaded_encoder)
            v_dice  = checkpoint.get("val_dice",     float("nan"))
            v_iou   = checkpoint.get("val_iou",      float("nan"))
            epoch   = checkpoint.get("epoch",        "?")
            st.sidebar.success(
                f"✅ **Loaded:** {arch} + {enc}\n\n"
                f"Epoch {epoch} · Val Dice `{v_dice:.4f}` · Val IoU `{v_iou:.4f}`"
            )
            st.sidebar.caption(f"Checkpoint: {checkpoint_path}")

        return model

    except Exception as exc:
        st.error("⚠️ Model loading failed.")
        st.exception(exc)
        return None


# ═════════════════════════════════════════════
#  POST-PROCESSING  (mirrors train.py exactly)
# ═════════════════════════════════════════════

def morphological_cleanup(
    mask_uint8    : np.ndarray,
    min_size      : int = 300,
    closing_radius: int = 3,
) -> np.ndarray:
    """
    1. Remove small false-positive islands smaller than `min_size` px.
    2. Binary closing with a disk structuring element to smooth contours
       and fill tiny holes at the tumour boundary.

    Args:
        mask_uint8      : Binary mask, dtype uint8, values {0, 1}.
        min_size        : Minimum connected-component area to retain.
        closing_radius  : Radius of the disk used for binary closing.
    Returns:
        Cleaned binary mask, dtype uint8.
    """
    mask_bool = mask_uint8.astype(bool)
    cleaned   = remove_small_objects(mask_bool, min_size=min_size)
    cleaned   = binary_closing(cleaned, footprint=disk(closing_radius))
    return cleaned.astype(np.uint8)


def crf_refine(
    image_gray: np.ndarray,
    prob_map  : np.ndarray,
    num_iters : int = 5,
    sxy       : tuple = (80, 80),
    srgb      : tuple = (13, 13, 13),
) -> np.ndarray:
    """
    Dense CRF refinement (pydensecrf).  Aligns the soft probability
    boundary to image gradients, producing a "shrink-wrapped" contour.

    Falls back to the raw probability map when pydensecrf is unavailable.

    Args:
        image_gray : uint8 grayscale array (H, W).
        prob_map   : float32 probability map in [0, 1], shape (H, W).
        num_iters  : Number of CRF mean-field iterations.
        sxy        : Bilateral spatial std (x, y). Smaller = finer spatial precision.
                     MG default: (40, 40) to align with high-res MG gradients.
                     US default: (80, 80).
        srgb       : Bilateral colour std (r, g, b). Smaller = tighter colour grouping.
                     MG default: (5, 5, 5).  US default: (13, 13, 13).
    Returns:
        Refined probability map, float32, shape (H, W).
    """
    if not _CRF_AVAILABLE:
        return prob_map

    H, W    = image_gray.shape
    img_rgb = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)

    d = dcrf.DenseCRF2D(W, H, 2)

    # Unary potentials from the network's soft probabilities
    probs = np.stack([1.0 - prob_map, prob_map], axis=0).astype(np.float32)
    probs = np.clip(probs, 1e-6, 1.0 - 1e-6)               # numerical safety
    d.setUnaryEnergy(unary_from_softmax(probs))

    # Appearance (bilateral) term — rewards coherent colour + position
    d.addPairwiseBilateral(
        sxy=sxy, srgb=srgb,
        rgbim=img_rgb, compat=10,
    )
    # Smoothness (Gaussian) term — penalises label roughness
    d.addPairwiseGaussian(sxy=(3, 3), compat=3)

    Q       = d.inference(num_iters)
    refined = np.array(Q)[1].reshape(H, W)
    return refined.astype(np.float32)


def postprocess(
    prob_map    : np.ndarray,
    image_gray  : Optional[np.ndarray] = None,
    threshold   : float = 0.5,
    use_crf     : bool  = False,
    min_size    : int   = 300,
    closing_radius: int = 3,
    crf_sxy     : tuple = (80, 80),
    crf_srgb    : tuple = (13, 13, 13),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Full post-processing pipeline:
      1. Optional CRF boundary refinement (uses `image_gray`).
      2. Threshold to binary mask.
      3. Morphological cleanup (remove islands, close contours).

    Args:
        crf_sxy  : Bilateral spatial std for CRF. Use (40,40) for MG.
        crf_srgb : Bilateral colour std for CRF. Use (5,5,5) for MG.

    Returns:
        refined_prob  : float32 probability map after optional CRF (H, W).
        binary_mask   : uint8 binary mask {0, 1} after cleanup (H, W).
    """
    refined_prob = prob_map.copy()

    if use_crf and image_gray is not None:
        refined_prob = crf_refine(
            image_gray, refined_prob,
            sxy=crf_sxy, srgb=crf_srgb,
        )

    binary_mask = (refined_prob > threshold).astype(np.uint8)
    binary_mask = morphological_cleanup(
        binary_mask, min_size=min_size, closing_radius=closing_radius
    )

    return refined_prob, binary_mask


# ═════════════════════════════════════════════
#  MAIN APP
# ═════════════════════════════════════════════

uploaded_file = st.file_uploader(
    "📂 Upload Medical Scan (JPG / PNG)",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:

    # ── 1. Load model ────────────────────────────────────────────────────
    mode = "us" if MODE == "Ultrasound (US)" else "mg"
    model = load_boss_model(mode)

    if model is None:
        st.stop()

    # ── 2. Decode uploaded image ─────────────────────────────────────────
    file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    orig_image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    if orig_image is None:
        st.error("❌ Could not decode the uploaded image.")
        st.stop()

    orig_h, orig_w = orig_image.shape

    # ── 3. Pre-process (identical to training val pipeline) ────────────────
    if mode == "mg":
        # MG pipeline (mirrors dataset_mg.py __getitem__):
        #   1. CLAHE contrast enhancement
        #   2. Percentile normalization → [0, 1]
        #   3. A.Normalize(0.5, 0.25, max_pixel_value=1.0) + Resize + ToTensor
        proc_image = _apply_clahe(orig_image)
        proc_image = _percentile_normalize(proc_image)  # float32 [0, 1]
    else:
        proc_image = orig_image  # US: raw uint8, A.Normalize handles it

    infer_transform = _get_infer_transform(mode)
    transformed    = infer_transform(image=proc_image, mask=np.zeros_like(orig_image))
    input_tensor   = transformed["image"]           # float32 (1, H, W)
    input_tensor   = input_tensor.unsqueeze(0)      # → (1, 1, H, W)

    # ── 4. Inference ─────────────────────────────────────────────────────
    with torch.no_grad():
        output = model(input_tensor.to(DEVICE))

        # DeepSupervisionWrapper returns (final_logit, [aux_logits…])
        # Plain UnetPlusPlus returns a single tensor — handle both gracefully.
        if isinstance(output, (tuple, list)):
            final_logit = output[0]                 # (1,1,H,W)
        else:
            final_logit = output

        prob_map_tensor = torch.sigmoid(final_logit)   # [0,1]

    # ── 5. Squeeze probability map to numpy (H,W) ─────────────────────────
    prob_map_model = prob_map_tensor.squeeze().cpu().numpy()   # (IMG_SIZE, IMG_SIZE) float32

    # ── 6. CRF + morphological post-processing ────────────────────────────
    # Resize the source image to model resolution so the CRF bilateral term
    # is computed at the same resolution as the probability map.
    image_model = cv2.resize(orig_image, (IMG_SIZE, IMG_SIZE))

    # MG: finer CRF parameters to exploit high-res gradient detail
    # US: standard parameters (wide spatial/colour range for smoothing)
    if mode == "mg":
        crf_sxy  = (40, 40)
        crf_srgb = (5, 5, 5)
    else:
        crf_sxy  = (80, 80)
        crf_srgb = (13, 13, 13)

    with st.spinner("🔬 Applying post-processing…"):
        refined_prob, mask_model = postprocess(
            prob_map    = prob_map_model,
            image_gray  = image_model,
            threshold   = THRESHOLD,
            use_crf     = USE_CRF and _CRF_AVAILABLE,
            min_size    = int(MORPH_MIN_SIZE),
            closing_radius = int(MORPH_RADIUS),
            crf_sxy     = crf_sxy,
            crf_srgb    = crf_srgb,
        )

    # ── 7. Scale mask back to original resolution ─────────────────────────
    mask_orig = cv2.resize(
        mask_model.astype(np.uint8),
        (orig_w, orig_h),
        interpolation=cv2.INTER_NEAREST,    # preserve hard edges
    )

    # ── 8. Build RGB overlay (red = tumour region) ────────────────────────
    overlay = cv2.merge([orig_image, orig_image, orig_image])
    overlay[mask_orig > 0] = [255, 50, 50]

    # ── 9. Display ────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.image(orig_image, caption="Original Patient Scan", use_container_width=True)
    with col2:
        st.image(overlay, caption="AI-Detected Oncology Signature", use_container_width=True)

    st.success("✅ Analysis complete — tumour region segmented successfully.")

    # ── 10. Metrics (computed from post-processed mask) ───────────────────
    st.markdown("---")
    st.subheader("📊 Analytics")

    tumour_px   = int(np.sum(mask_orig))
    total_px    = orig_h * orig_w
    coverage    = (tumour_px / total_px) * 100.0

    # Confidence: mean probability in the post-processed tumour region only.
    # We re-scale refined_prob to original resolution for this calculation.
    refined_orig = cv2.resize(refined_prob, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    tumour_region_probs = refined_orig[mask_orig > 0]
    avg_confidence = float(tumour_region_probs.mean()) if tumour_region_probs.size > 0 else 0.0

    # Global mean probability (raw, before post-processing)
    raw_confidence = float(prob_map_tensor.mean().item())

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("🔴 Estimated Tumour Density", f"{coverage:.2f} %")
    with m2:
        st.metric(
            "🎯 Region Confidence",
            f"{avg_confidence:.3f}",
            help="Mean post-CRF probability inside the segmented tumour region.",
        )
    with m3:
        st.metric(
            "📡 Global Model Confidence",
            f"{raw_confidence:.3f}",
            help="Mean raw sigmoid probability across the entire image.",
        )

    # ── Optional: expander with diagnostics ──────────────────────────────
    with st.expander("🔧 Diagnostic Info"):
        st.write({
            "Modality"          : MODE,
            "Encoder"           : ENCODER,
            "CRF Applied"       : USE_CRF and _CRF_AVAILABLE,
            "Threshold"         : THRESHOLD,
            "Morph Min Size"    : MORPH_MIN_SIZE,
            "Morph Closing R"   : MORPH_RADIUS,
            "Input Size"        : f"{IMG_SIZE}×{IMG_SIZE}",
            "Original Size"     : f"{orig_w}×{orig_h}",
            "Tumour Pixels"     : tumour_px,
            "Total Pixels"      : total_px,
            "Device"            : str(DEVICE),
        })
        st.image(
            (refined_prob * 255).astype(np.uint8),
            caption="Post-CRF Probability Map (512×512)",
            use_container_width=True,
        )
