import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from dataset import MultiModalityDataset
import argparse

# --- SYSTEM CONFIG ---
torch.backends.cudnn.benchmark = True # Speeds up Tesla T4 processing
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 512
BATCH_SIZE = 8 
EPOCHS = 80
INITIAL_LR = 1e-4

def train_model(mode):
    print(f"🧬 BOSS Elite Training Initialized: {mode.upper()} on {torch.cuda.get_device_name(0)}")
    
    # 1. Specialist Model Selection
    if mode == "us":
        encoder = "efficientnet-b4"
        print("🔍 Mode: Ultrasound (Optimizing for Speckle Noise & Texture)")
    else:
        encoder = "resnet101"
        print("📍 Mode: Mammography (Optimizing for High-Res Calcifications)")

    # 2. Setup Paths
    data_dir = f"data/processed_{mode}/train"
    save_path = f"models/{mode}_elite_best.pth"
    os.makedirs("models", exist_ok=True)

    # 3. Dataset & Loader
    dataset = MultiModalityDataset(
        image_dir=os.path.join(data_dir, "images"),
        mask_dir=os.path.join(data_dir, "masks"),
        img_size=IMG_SIZE,
        mode=mode
    )
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    # 4. Model Architecture (U-Net + Specialist Encoder + scSE Attention)
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights="imagenet",
        in_channels=1,
        classes=1,
        decoder_attention_type='scse'
    ).to(DEVICE)

    # 5. Hybrid Loss & Optimizer
    # Combining Dice and BCE is the "Gold Standard" for medical segmentation
    dice_loss = smp.losses.DiceLoss(mode='binary')
    bce_loss = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=INITIAL_LR)
    
    # Scheduler: Reduces LR by 10x if loss doesn't improve for 5 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    # 6. Training Loop
    best_loss = float('inf')
    model.train()
    
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for images, masks in train_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            # Calculate Hybrid Loss
            loss = dice_loss(outputs, masks) + bce_loss(outputs, masks)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        
        # Step the scheduler based on average loss
        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")
        
        # Save the absolute "Best" version
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)
            print(f"⭐ New Elite Best Saved (Loss: {best_loss:.4f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["us", "mg"])
    args = parser.parse_args()
    train_model(args.mode)