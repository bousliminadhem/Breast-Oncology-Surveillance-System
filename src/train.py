import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import MammographyDatasetV3 # Using the new PNG + Augmentation class
import segmentation_models_pytorch as smp
from tqdm import tqdm
import os

def train_model():
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting Training on: {device}")
    os.makedirs("models", exist_ok=True)

    # 2. Initialize Model
    model = smp.Unet(
        encoder_name="resnet34",        
        encoder_weights="imagenet",     
        in_channels=1,                  
        classes=1,                      
    ).to(device)

    # 3. Data Loaders (Using the full 210-image set)
    train_ds = MammographyDatasetV3("data/processed/train/images", "data/processed/train/masks", is_train=True)
    test_ds = MammographyDatasetV3("data/processed/test/images", "data/processed/test/masks", is_train=False)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    print(f"📊 Training on {len(train_ds)} images | Testing on {len(test_ds)} images")

    # 4. Loss & Optimizer
    criterion = smp.losses.DiceLoss(mode='binary')
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 5. Training Loop
    epochs = 50 # Increased for the smaller, augmented dataset
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # --- TRAINING ---
        model.train()
        running_train_loss = 0.0
        train_loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]", leave=False)
        
        for images, masks in train_loop:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())

        avg_train_loss = running_train_loss / len(train_loader)

        # --- VALIDATION ---
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for images, masks in test_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                v_loss = criterion(outputs, masks)
                running_val_loss += v_loss.item()

        avg_val_loss = running_val_loss / len(test_loader)

        # 6. Checkpoint
        print(f"Epoch [{epoch+1}/{50}] - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "models/best_malignant_model.pth")
            print("⭐ New Best Malignant Model Saved!")

    print(f"✅ DONE! Best model saved to models/best_malignant_model.pth")

if __name__ == "__main__":
    train_model()