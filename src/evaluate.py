import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from dataset import MultiModalityDataset
import segmentation_models_pytorch as smp
from sklearn.metrics import confusion_matrix
import os

def evaluate_expert(mode):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = f"data/processed_{mode}/test"
    model_path = f"models/{'ultrasound' if mode == 'us' else 'mammography'}_best.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found for {mode}")
        return None

    # Load Model
    model = smp.Unet(encoder_name="resnet34", in_channels=1, classes=1).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    test_loader = DataLoader(
        MultiModalityDataset(f"{data_path}/images", f"{data_path}/masks", is_train=False),
        batch_size=1, shuffle=False
    )

    metrics = {"dice": [], "iou": [], "precision": [], "recall": []}
    all_preds, all_masks = [], []

    print(f"🔬 Evaluating {mode.upper()} Model...")

    with torch.no_grad():
        for i, (img, mask) in enumerate(test_loader):
            img, mask = img.to(device), mask.to(device)
            output = model(img)
            
            # Binary Thresholding
            pred = (torch.sigmoid(output) > 0.5).int()
            mask = (mask > 0.5).int()

            # Calculate Stats
            tp, fp, fn, tn = smp.metrics.get_stats(pred, mask, mode='binary')
            
            metrics["dice"].append(smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro").item())
            metrics["iou"].append(smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro").item())
            metrics["precision"].append(smp.metrics.precision(tp, fp, fn, tn, reduction="micro").item())
            metrics["recall"].append(smp.metrics.recall(tp, fp, fn, tn, reduction="micro").item())

            # Save first 3 results for visualization
            if i < 3:
                save_visual_comparison(img[0], mask[0], pred[0], mode, i)

    # Summary Stats
    summary = {k: np.mean(v) for k, v in metrics.items()}
    summary['modality'] = mode.upper()
    return summary

def save_visual_comparison(img, mask, pred, mode, idx):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.imshow(img.cpu().squeeze(), cmap='gray'); plt.title("Original")
    plt.subplot(1, 3, 2); plt.imshow(mask.cpu().squeeze(), cmap='gray'); plt.title("Ground Truth")
    plt.subplot(1, 3, 3); plt.imshow(pred.cpu().squeeze(), cmap='gray'); plt.title("AI Prediction")
    plt.suptitle(f"{mode.upper()} Test Sample {idx}")
    plt.savefig(f"models/results_{mode}_{idx}.png")
    plt.close()

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    results = []
    for m in ["us", "mg"]:
        res = evaluate_expert(m)
        if res: results.append(res)
    
    if results:
        df = pd.DataFrame(results)
        print("\n🏆 FINAL PROJECT PERFORMANCE")
        print(df.to_string(index=False))
        df.to_csv("models/final_evaluation.csv")