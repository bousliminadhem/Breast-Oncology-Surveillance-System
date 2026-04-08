import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
import os

# --- UI CONFIG ---
st.set_page_config(page_title="BOSS - Oncology AI", layout="wide")
st.title("🎗️ Breast Oncology Surveillance System (BOSS)")
st.sidebar.header("Settings")

MODE = st.sidebar.selectbox("Select Modality", ["Ultrasound (US)", "Mammography (MG)"])

# Using raw strings for Windows paths
if "US" in MODE:
    MODEL_PATH = r"models\us_elite_best.pth"
    ENCODER = "efficientnet-b4"  # Match your Elite Colab training
else:
    MODEL_PATH = r"models\mg_resnet50_best.pth"
    ENCODER = "resnet50"

# --- MODEL LOADING ---
@st.cache_resource
def load_boss_model(path, encoder_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define architecture to match the saved .pth exactly
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=1,
        classes=1,
        decoder_attention_type='scse' # Spatial and Channel Squeeze & Excitation
    )
    
    if not os.path.exists(path):
        st.error(f"❌ File really is missing at: {os.path.abspath(path)}")
        return None, None

    try:
        # Loading weights - mapping to CPU/GPU automatically
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device).eval()
        return model, device
    except Exception as e:
        st.error(f"⚠️ Architecture Mismatch: Your code is {encoder_name}, but the file might be different.")
        st.info(f"Technical Error: {e}")
        return None, None

# --- PROCESSING ---
uploaded_file = st.file_uploader("Upload Medical Scan...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    model, device = load_boss_model(MODEL_PATH, ENCODER)
    
    if model:
        # 1. Load and Preprocess
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        orig_h, orig_w = image.shape
        
        # 2. Resize and Normalize to 0-1 (Matches Medical-Grade training)
        input_tensor = cv2.resize(image, (512, 512))
        input_tensor = torch.from_numpy(input_tensor).float().unsqueeze(0).unsqueeze(0) / 255.0
        
        with torch.no_grad():
            # Get prediction and apply sigmoid (0 to 1 range)
            output = model(input_tensor.to(device))
            prediction = torch.sigmoid(output)
            
            # Binary mask: 1 where probability > 0.5
            mask = (prediction > 0.5).cpu().numpy().squeeze().astype(np.uint8) 
            mask_resized = cv2.resize(mask, (orig_w, orig_h))

        # 3. Display Results
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Patient Scan", use_container_width=True)
        
        with col2:
            # Create a 3-channel overlay (Red color for tumor)
            overlay = cv2.merge([image, image, image])
            overlay[mask_resized > 0] = [255, 0, 0] # Highlight tumor in Red
            
            st.image(overlay, caption="AI-Detected Oncology Signature", use_container_width=True)
            
        st.success("✅ Analysis Complete. Tumor region segmented successfully.")
        
        # --- Analytics for the Jury ---
        tumor_pixel_count = np.sum(mask_resized)
        total_pixels = orig_h * orig_w
        coverage = (tumor_pixel_count / total_pixels) * 100
        st.metric("Estimated Tumor Density", f"{coverage:.2f} %")