import numpy as np
import pydicom
from pathlib import Path

def get_tumor_size(binary_mask, dicom_path):
    # 1. Load the original DICOM to get physical scale
    ds = pydicom.dcmread(dicom_path)
    
    # PixelSpacing is usually a list: [row_spacing, col_spacing] in mm
    # Example: [0.07, 0.07]
    spacing = ds.get("PixelSpacing", [1.0, 1.0]) 
    pixel_area_mm2 = float(spacing[0]) * float(spacing[1])
    
    # 2. Count the 'Tumor' pixels (where mask is 1)
    tumor_pixel_count = np.sum(binary_mask == 1)
    
    # 3. Calculate physical area
    total_area_mm2 = tumor_pixel_count * pixel_area_mm2
    
    return total_area_mm2, spacing[0]

# --- Integration Example ---
if __name__ == "__main__":
    # We will use this in your final dashboard
    print("Measurement module ready.")