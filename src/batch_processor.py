import pandas as pd
import os
from pathlib import Path
import cv2
from utils import medical_grade_preprocess, save_as_png

csv_path = r"D:\Projet Fédérateur\AI\BCT\data\raw\metadata\metadata.csv"
raw_data_root = r"data\raw\cbis_ddsm"
output_base = r"data\processed\train"

def run_batch_processing(csv_path, raw_data_root, output_base):
    # 1. Load the metadata
    df = pd.read_csv(csv_path)
    
    # 2. Define output paths
    img_out_dir = Path(output_base) / "images"
    mask_out_dir = Path(output_base) / "masks"
    img_out_dir.mkdir(parents=True, exist_ok=True)
    mask_out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting batch processing for {len(df)} cases...")

    for index, row in df.iterrows():
        # In your CSV, PatientID is the folder name (e.g., Mass-Training_P_00001_LEFT_MLO_1)
        folder_name = row['PatientID']
        patient_folder = Path(raw_data_root) / folder_name
        
        try:
            # Find all DICOM files inside this patient's folder
            dcm_files = list(patient_folder.rglob("*.dcm"))
            
            if len(dcm_files) < 2:
                print(f"⚠️ Skipping {folder_name}: Need at least 1 image and 1 mask (found {len(dcm_files)})")
                continue

            # IDENTIFICATION LOGIC:
            # The Full Mammogram is large (several MBs).
            # The ROI Mask is small (usually < 1MB).
            # We sort by file size: smallest first (mask), largest last (image).
            dcm_files.sort(key=lambda x: x.stat().st_size)
            
            full_mask_path = dcm_files[0]  # Smallest file
            full_img_path = dcm_files[-1]   # Largest file

            # 3. Process the Mammogram (with CLAHE)
            processed_img = medical_grade_preprocess(full_img_path)
            
            # 4. Process the Mask (Binary threshold)
            # We use the same function to ensure identical resizing/orientation
            processed_mask = medical_grade_preprocess(full_mask_path)
            _, processed_mask = cv2.threshold(processed_mask, 127, 255, cv2.THRESH_BINARY)

            # 5. Save both with the SAME name
            # We use the folder_name as the filename for perfect pairing
            save_as_png(processed_img, img_out_dir / f"{folder_name}.png")
            save_as_png(processed_mask, mask_out_dir / f"{folder_name}.png")
            
            if index % 10 == 0:
                print(f"✅ Processed {index}/{len(df)}: {folder_name}")

        except Exception as e:
            print(f"❌ Failed {folder_name}: {e}")

    print("\n✅ Batch processing complete! Check your data/processed folder.")

run_batch_processing(csv_path, raw_data_root, output_base)