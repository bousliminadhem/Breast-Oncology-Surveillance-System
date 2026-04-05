import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from dataset import MultiModalityDataset
import argparse

# --- CLOUD CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 512
BATCH_SIZE = 8  # Optimized for Tesla T4 16GB
EPOCHS = 80
LEARNING_RATE = 1e-4

def train_model(mode):
    print(f"🚀 BOSS Training Initialized: {mode.upper()} on {torch.cuda.get_device_name(0)}")
    
    # 1. Setup Paths
    data_dir = f"data/processed_{mode}/train"
    save_path = f"models/{mode}_resnet50_best.pth"
    os.makedirs("models", exist_ok=True)

    # 2. Dataset & Loader
    dataset = MultiModalityDataset(
        image_dir=os.path.join(data_dir, "images"),
        mask_dir=os.path.join(data_dir, "masks"),
        img_size=IMG_SIZE,
        mode=mode
    )
    
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # 3. Model: ResNet-50 + scSE (Spatial & Channel Squeeze & Excitation)
    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=1,
        classes=1,
        decoder_attention_type='scse'
    ).to(DEVICE)

    # 4. Loss & Optimizer
    criterion = smp.losses.DiceLoss(mode='binary')
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. Training Loop
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for images, masks in train_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f}")
        
        # Save every 10 epochs or at the end
        if (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
            # Around line 56 in train.py
            torch.save(model.state_dict(), save_path)
            print(f"✅ Model checkpoint saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["us", "mg"])
    args = parser.parse_args()
    train_model(args.mode)