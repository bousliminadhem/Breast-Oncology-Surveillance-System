import pydicom
import numpy as np
import cv2
import os

def medical_grade_preprocess(dicom_path, size=(512, 512)):
    """
    Standardizes mammograms using clinical-grade contrast enhancement.
    """
    # 1. Load DICOM
    ds = pydicom.dcmread(dicom_path)
    img = ds.pixel_array.astype(np.float32)

    # 2. Photometric Interpretation Check
    # Ensure white is always white and black is always black
    if ds.PhotometricInterpretation == "MONOCHROME1":
        img = np.max(img) - img

    # 3. Global Normalization (Min-Max Scaling)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img = (img * 255).astype(np.uint8)

    # 4. CLAHE (The "Medical Secret Sauce")
    # This enhances local contrast to make tumors "pop" from dense tissue
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_enhanced = clahe.apply(img)

    # 5. Resize
    img_final = cv2.resize(img_enhanced, size, interpolation=cv2.INTER_LANCZOS4)

    return img_final

def save_as_png(image_array, output_path):
    cv2.imwrite(output_path, image_array)