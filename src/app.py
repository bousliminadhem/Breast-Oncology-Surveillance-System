import streamlit as st
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import os
from PIL import Image

# --- CONFIGURATION ---
st.set_page_config(page_title="Breast Oncology AI", layout="wide")

MODELS = {
    "Ultrasound": {
        "path": "models/ultrasound_best.pth",
        "use_clahe": True,
        "desc": "Optimized for Dataset 1 (Nodules & Cysts)"
    },
    "Mammography": {
        "path": "models/mammography_best.pth",
        "use_clahe": False,
        "desc": "Optimized for Dataset 2 (Masses & Calcifications)"
    }
}

# --- MODEL LOADER ---
@st.cache_resource
def load_model(modality):
    config = MODELS[modality]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize U-Net ResNet-34
    model = smp.Unet(encoder_name="resnet34", in_channels=1, classes=1)
    
    if os.path.exists(config["path"]):
        model.load_state_dict(torch.load(config["path"], map_location=device))
    
    model.to(device).eval()
    return model, device

# --- PROCESSING LOGIC ---
def process_image(uploaded_file, modality):
    # Load and Resize
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    raw_img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    raw_img = cv2.resize(raw_img, (512, 512))
    
    # Modality Enhancement
    if MODELS[modality]["use_clahe"]:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        processed = clahe.apply(raw_img)
    else:
        processed = raw_img.copy()
        
    # To Tensor
    tensor = torch.from_numpy(processed).float().unsqueeze(0).unsqueeze(0) / 255.0
    return raw_img, tensor

# --- MAIN UI ---
def main():
    st.title("🎗️ Breast Oncology Surveillance System")
    st.markdown("---")

    # Sidebar
    st.sidebar.header("📋 Clinical Parameters")
    modality = st.sidebar.selectbox("Select Modality", list(MODELS.keys()))
    st.sidebar.info(MODELS[modality]["desc"])
    
    threshold = st.sidebar.slider("Detection Sensitivity", 0.1, 0.9, 0.5)
    
    uploaded_file = st.sidebar.file_uploader("Upload Medical Scan (PNG/JPG)", type=['png', 'jpg', 'jpeg'])

    if uploaded_file:
        col1, col2 = st.columns(2)
        
        # 1. Load Data & Model
        model, device = load_model(modality)
        original_img, input_tensor = process_image(uploaded_file, modality)
        
        # 2. Inference
        with torch.no_grad():
            output = model(input_tensor.to(device))
            pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
            binary_mask = (pred_mask > threshold).astype(np.uint8)
        
        # 3. Visualization
        with col1:
            st.subheader("Original Scan")
            st.image(original_img, use_column_width=True)
            
        with col2:
            st.subheader("AI Analysis")
            # Create Green Overlay
            overlay = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
            overlay[binary_mask == 1] = [0, 255, 0] # Highlight in Green
            
            # Blend original and green (for professional look)
            final_view = cv2.addWeighted(cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB), 0.7, overlay, 0.3, 0)
            st.image(final_view, use_column_width=True)

        # 4. Results
        area_percent = (np.sum(binary_mask) / (512*512)) * 100
        if area_percent > 0:
            st.error(f"⚠️ Malignancy Detected. Affected Area: {area_percent:.2f}%")
        else:
            st.success("✅ No Malignancy Detected in this scan.")

if __name__ == "__main__":
    main()