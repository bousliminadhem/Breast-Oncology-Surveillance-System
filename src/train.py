import torch
import torch.optim as optim
import argparse
import os
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from dataset import MultiModalityDataset
from tqdm import tqdm

def train_modality(mode):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("models", exist_ok=True)
    
    # 1. SETTINGS
    if mode == "us":
        data_path, model_save_path = "data/processed_us", "models/ultrasound_best.pth"
        initial_lr, epochs = 1e-4, 50
    else:
        data_path, model_save_path = "data/processed_mg", "models/mammography_best.pth"
        initial_lr, epochs = 1e-4, 80 # MG needs more time to converge

    # 2. DATA
    train_loader = DataLoader(
        MultiModalityDataset(f"{data_path}/train/images", f"{data_path}/train/masks", is_train=True),
        batch_size=8, shuffle=True
    )
    val_loader = DataLoader(
        MultiModalityDataset(f"{data_path}/test/images", f"{data_path}/test/masks", is_train=False),
        batch_size=8, shuffle=False
    )

    # 3. MODEL & OPTIMIZER
    model = smp.Unet(encoder_name="resnet34", in_channels=1, classes=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    criterion = smp.losses.DiceLoss(mode='binary')

    # --- THE SCHEDULER ---
    # mode='max' because we want to maximize the Dice Score
    # factor=0.1 means multiply LR by 0.1 (1e-4 becomes 1e-5)
    # patience=5 means wait 5 epochs of no improvement before dropping
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)

    best_dice = 0.0
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        running_dice = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                tp, fp, fn, tn = smp.metrics.get_stats(outputs, (masks > 0.5).int(), mode='binary', threshold=0.5)
                running_dice += smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
        
        avg_dice = running_dice / len(val_loader)
        
        # --- UPDATE SCHEDULER ---
        # It looks at the Dice score and decides if it needs to slow down
        scheduler.step(avg_dice)

        print(f"Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f} | Dice={avg_dice:.4f} | LR={optimizer.param_groups[0]['lr']}")

        if avg_dice > best_dice:
            best_dice = avg_dice
            torch.save(model.state_dict(), model_save_path)
            print("⭐ Best Weights Updated!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["us", "mg"], required=True)
    args = parser.parse_args()
    train_modality(args.mode)