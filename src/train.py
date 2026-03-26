import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import MammographyDataset
import segmentation_models_pytorch as smp # Your pre-installed U-Net library

def train_model():
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on: {device}")

    # 2. Initialize Model (U-Net with ResNet34 backbone)
    model = smp.Unet(
        encoder_name="resnet34",        
        encoder_weights="imagenet",     
        in_channels=1,                  
        classes=1,                      
    ).to(device)

    # 3. Data Loaders
    dataset = MammographyDataset("data/processed/train/images", "data/processed/train/masks")
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

    # 4. Loss & Optimizer
    # DiceLoss is "medical grade"—it's better than standard loss for small tumors
    criterion = smp.losses.DiceLoss(mode='binary')
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 5. Training Loop
    epochs = 20
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Backward pass (The learning part)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss/len(train_loader):.4f}")

    # 6. Save the Brain
    torch.save(model.state_dict(), "models/unet_mammogram.pth")
    print("✅ Training Complete. Model saved to models/unet_mammogram.pth")

if __name__ == "__main__":
    train_model()