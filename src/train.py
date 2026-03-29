import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import segmentation_models_pytorch as smp
from dataset import MultiModalityDataset
import os
import argparse

# --- HYPER-PARAMETERS FOR PRECISION ---
IMG_SIZE = 512
BATCH_SIZE = 2  # Lowered for Attention U-Net memory overhead
EPOCHS = 60     # Increased to allow the "Attention" to fine-tune
LEARNING_RATE = 1e-4

def train_modality(mode):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Precision Training: {mode.upper()} on {torch.cuda.get_device_name(0)}")

    # 1. ARCHITECTURE: Attention U-Net
    # decoder_attention_type="scse" adds 'Spatial and Channel Squeeze & Excitation'
    # This helps the model "shrink-wrap" the mask to the tumor boundaries.
    model = smp.Unet(
        encoder_name="resnet34", 
        in_channels=1, 
        classes=1,
        decoder_attention_type="scse" 
    ).to(device)

    # 2. DATA SPLIT (80/20)
    full_dataset = MultiModalityDataset(
        image_dir=f"data/processed_{mode}/train/images",
        mask_dir=f"data/processed_{mode}/train/masks",
        is_train=True
    )
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # 3. LOSS FUNCTION: Tversky + BCE
    # Tversky is better than Dice for "unbalanced" data (small tumors in large images)
    tversky_loss = smp.losses.TverskyLoss(mode='binary', alpha=0.7, beta=0.3) 
    bce_loss = nn.BCEWithLogitsLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7)

    # 4. TRAINING LOOP
    best_val_dice = 0.0
    model_save_path = f"models/{mode}_best.pth"
    os.makedirs("models", exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            
            outputs = model(imgs)
            # Combine losses: Tversky for boundary, BCE for stability
            loss = 0.6 * tversky_loss(outputs, masks) + 0.4 * bce_loss(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # VALIDATION PHASE
        model.eval()
        total_val_dice = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                # Apply Sigmoid to get probabilities [0, 1]
                preds = torch.sigmoid(model(imgs))
                preds = (preds > 0.5).float() # Threshold for Dice calc
                
                # Dice Calculation
                intersect = (preds * masks).sum()
                union = preds.sum() + masks.sum()
                total_val_dice += (2. * intersect / (union + 1e-7)).item()
        
        avg_val_dice = total_val_dice / len(val_loader)
        avg_loss = epoch_loss / len(train_loader)
        scheduler.step(avg_loss)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Val Dice: {avg_val_dice:.4f}")

        # ONLY SAVE IF PRECISION IMPROVES
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            torch.save(model.state_dict(), model_save_path)
            print(f"🌟 PRECISION BOOST: New Best Val Dice: {best_val_dice:.4f}")

    print(f"✅ Training Complete. Best precision model saved to {model_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, help="us or mg")
    args = parser.parse_args()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    train_modality(args.mode)