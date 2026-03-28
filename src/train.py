import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import segmentation_models_pytorch as smp
from dataset import MultiModalityDataset
import os

def train_modality(mode):
    device = torch.device("cuda")
    
    # 1. Model
    model = smp.Unet(encoder_name="resnet34", in_channels=1, classes=1).to(device)

    # 2. Data Splitting (The "Best Result" Strategy)
    full_dataset = MultiModalityDataset(
        image_dir=f"data/processed_{mode}/train/images",
        mask_dir=f"data/processed_{mode}/train/masks",
        is_train=True
    )
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)

    # 3. Hybrid Loss for Medical Imaging
    dice_loss = smp.losses.DiceLoss(mode='binary')
    bce_loss = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 4. Training Loop with "Best-Only" Saving
    best_val_dice = 0.0
    for epoch in range(40):
        model.train()
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            # Combined Loss: 50% Dice, 50% BCE
            loss = 0.5 * dice_loss(outputs, masks) + 0.5 * bce_loss(outputs, masks)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation Phase (Every Epoch)
        model.eval()
        total_val_dice = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                preds = (torch.sigmoid(model(imgs)) > 0.5).float()
                # Simple Dice Calculation
                intersect = (preds * masks).sum()
                union = preds.sum() + masks.sum()
                total_val_dice += (2. * intersect / (union + 1e-7)).item()
        
        avg_val_dice = total_val_dice / len(val_loader)
        print(f"Epoch {epoch+1} | Val Dice: {avg_val_dice:.4f}")

        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            torch.save(model.state_dict(), f"models/{mode}_best.pth")
            print("🌟 New Best Model Saved!")