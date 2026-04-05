import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp

# --- UI CONFIG ---
st.set_page_config(page_title="BOSS - Oncology AI", layout="wide")
st.title("🎗️ Breast Oncology Surveillance System (BOSS)")
st.sidebar.header("Settings")

MODE = st.sidebar.selectbox("Select Modality", ["Ultrasound (US)", "Mammography (MG)"])
MODEL_PATH = "models/us_resnet50_best.pth" if "US" in MODE else "models/mg_resnet50_best.pth"

# --- MODEL LOADING ---
@st.cache_resource
def load_boss_model(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights=None,
        in_channels=1,
        classes=1,
        decoder_attention_type='scse'
    )
    try:
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device).eval()
        return model, device
    except:
        st.error(f"Model file not found at {path}. Please train the model first!")
        return None, None

# --- PROCESSING ---
uploaded_file = st.file_uploader("Upload Medical Scan...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    model, device = load_boss_model(MODEL_PATH)
    
    if model:
        # Load and Preprocess
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        orig_h, orig_w = image.shape
        
        input_tensor = cv2.resize(image, (512, 512))
        input_tensor = torch.from_numpy(input_tensor).float().unsqueeze(0).unsqueeze(0) / 255.0
        
        with torch.no_grad():
            prediction = torch.sigmoid(model(input_tensor.to(device)))
            # .astype(np.uint8) converts True/False to 1s and 0s
            mask = (prediction > 0.5).cpu().numpy().squeeze().astype(np.uint8) 
            mask_resized = cv2.resize(mask, (orig_w, orig_h))

        # Display Results
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Scan", use_column_width=True)
        with col2:
            # Create overlay
            overlay = cv2.merge([image, image, image])
            overlay[mask_resized > 0] = [255, 0, 0] # Red highlights for tumor
            st.image(overlay, caption="AI Tumor Detection", use_column_width=True)
            
        st.success("Analysis Complete. Tumor progression signature extracted.")