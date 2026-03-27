import os
import shutil
import random
from pathlib import Path

def organize_v3(source_folder, output_base, train_ratio=0.85):
    # 1. Setup paths
    src_path = Path(source_folder)
    out_path = Path(output_base)
    
    # We are looking for files that DON'T have "_mask" in the name to count images
    all_images = [f.name for f in src_path.glob("*.png") if "_mask" not in f.name]
    random.seed(42)
    random.shuffle(all_images)

    split_idx = int(len(all_images) * train_ratio)
    train_files = all_images[:split_idx]
    test_files = all_images[split_idx:]

    def move_files(file_list, subset):
        count = 0
        img_dir = out_path / subset / "images"
        mask_dir = out_path / subset / "masks"
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        for img_name in file_list:
            # Construct names: malignant(1).png and malignant(1)_mask.png
            mask_name = img_name.replace(".png", "_mask.png")
            
            if (src_path / mask_name).exists():
                shutil.copy(src_path / img_name, img_dir / img_name)
                shutil.copy(src_path / mask_name, mask_dir / img_name) # Rename mask to match image
                count += 1
        return count

    print(f"✅ Processed {move_files(train_files, 'train')} pairs for Training.")
    print(f"✅ Processed {move_files(test_files, 'test')} pairs for Testing.")

if __name__ == "__main__":
    # Point this to wherever your 210 PNGs are currently sitting
    SOURCE = r"D:\Path\To\Your\New\PNG_Folder" 
    DEST = r"D:\Projet_Federateur\AI\Breast Oncology Surveillance System\data\processed\V3"
    organize_v3(SOURCE, DEST)