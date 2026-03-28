import os
import shutil
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split

def prepare_and_save(pair_list, target_base, subset):
    for i, (img_p, mask_p, prefix) in enumerate(pair_list):
        img = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
        
        if img is None or mask is None: continue
        
        # Standardize to 512x512
        img = cv2.resize(img, (512, 512))
        mask = cv2.resize(mask, (512, 512))
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        new_name = f"{prefix}_{i}.png"
        cv2.imwrite(str(target_base / subset / "images" / new_name), img)
        cv2.imwrite(str(target_base / subset / "masks" / new_name), mask)

def unify_simple():
    processed_us = Path("data/processed_us")
    processed_mg = Path("data/processed_mg")
    
    # Clean/Rebuild
    for p in [processed_us, processed_mg]:
        for sub in ["train/images", "train/masks", "test/images", "test/masks"]:
            (p / sub).mkdir(parents=True, exist_ok=True)

    # --- ULTRASOUND (Standard Masks) ---
    us_pairs = []
    us_raw = Path("data/raw/Ultrasound")
    for cat in ['benign', 'malignant', 'normal']:
        folder = us_raw / cat
        if not folder.exists(): continue
        imgs = [f for f in folder.glob("*.png") if "_mask" not in f.name]
        for img_p in imgs:
            mask_p = folder / f"{img_p.stem}_mask.png"
            if mask_p.exists(): us_pairs.append((img_p, mask_p, f"US_{cat}"))

    # --- MAMMOGRAPHY (Standard Masks - NOT PLA) ---
    mg_pairs = []
    mg_img_dir = Path("data/raw/Mammography/512/TIFF")
    # Change this to your standard 'Mask' folder instead of 'PLA_PNG'
    mg_mask_dir = Path("data/raw/Mammography/512/Mask") 
    
    if mg_img_dir.exists() and mg_mask_dir.exists():
        for img_p in mg_img_dir.glob("*.png"):
            mask_p = mg_mask_dir / img_p.name
            if mask_p.exists():
                mg_pairs.append((img_p, mask_p, "MG_Standard"))

    # SPLIT
    us_train, us_test = train_test_split(us_pairs, test_size=0.15, random_state=42)
    mg_train, mg_test = train_test_split(mg_pairs, test_size=0.15, random_state=42)

    prepare_and_save(us_train, processed_us, "train")
    prepare_and_save(us_test, processed_us, "test")
    prepare_and_save(mg_train, processed_mg, "train")
    prepare_and_save(mg_test, processed_mg, "test")

    print(f"✅ Setup complete using STANDARD masks. US: {len(us_pairs)} | MG: {len(mg_pairs)}")

if __name__ == "__main__":
    unify_simple()