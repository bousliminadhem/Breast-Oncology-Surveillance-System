from dataset import MammographyDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Setup paths
img_dir = "data/processed/train/images"
mask_dir = "data/processed/train/masks"

# Initialize dataset and loader
dataset = MammographyDataset(img_dir, mask_dir)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Pull one pair
image, mask = next(iter(loader))

print(f"Image shape: {image.shape}") # Should be [1, 1, 512, 512]
print(f"Mask shape: {mask.shape}")   # Should be [1, 1, 512, 512]

# Visualize
plt.subplot(1, 2, 1)
plt.title("AI Input (Image)")
plt.imshow(image[0][0], cmap='gray')

plt.subplot(1, 2, 2)
plt.title("AI Target (Mask)")
plt.imshow(mask[0][0], cmap='gray')

plt.show()