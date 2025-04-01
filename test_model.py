# visualise_trained_model.py

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from torchvision.transforms.functional import to_tensor
from train_unet_semantic_segmentation_module import UNet, SegmentationDataset, tile_image
from PIL import Image
import rasterio
from utils import evaluate_model, load_rgb_tif

# ========== Step 1: Load Trained Model ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(n_classes=8)

# # UNet model as baseline
# model.load_state_dict(torch.load("Baseline/unet_model_new.pth", map_location=device))

# UNet model with data augmentation
model.load_state_dict(torch.load("data_augmentation/unet_model_data_augmented_new.pth", map_location=device))

model.to(device)
model.eval()

# ========== Step 2: Load and Tile the Same Images ==========

image_paths = [r"C:\cartography\project\data\rgb_TA_124_1918.tif",
               r"C:\cartography\project\data\rgb_TA_138_1876.tif"]
mask_paths = [r"C:\cartography\project\data\testing_target_s1918.png",
              r"C:\cartography\project\data\testing_target_s1876.png"]
tile_size = 500

image_tiles, mask_tiles = [], []

for img_path, msk_path in zip(image_paths, mask_paths):
    image = load_rgb_tif(img_path)
    mask = Image.open(msk_path).convert("L")
    mask = np.array(mask)
    print(f"{msk_path}'s Mask unique values: {np.unique(mask)}")
    image_tiles += tile_image(image, tile_size)
    mask_tiles += tile_image(mask, tile_size)

dataset = SegmentationDataset(image_tiles, mask_tiles)

# ========== Step 3: Randomly Visualize 2 Predictions ==========
indices = random.sample(range(len(image_tiles)), 2)
for i, idx in enumerate(indices):
    img_np = image_tiles[idx]
    true_mask = mask_tiles[idx]

    img_tensor = to_tensor(img_np).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_mask = model(img_tensor).argmax(1).squeeze().cpu()

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(img_np)
    ax[0].set_title(f"Image (Tile {idx})")
    ax[1].imshow(pred_mask, cmap='tab20')
    ax[1].set_title("Prediction")
    ax[2].imshow(true_mask, cmap='tab20')
    ax[2].set_title("Ground Truth")
    for a in ax: a.axis("off")
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    # plt.savefig(f"Baseline/prediction_compare_{i}.png")
    plt.savefig(f"data_augmentation/prediction_compare_{i}_new.png")
    plt.close()


# Calculate metrics
evaluate_model(
    model=model,
    image_tiles=image_tiles,
    mask_tiles=mask_tiles,
    # save_path="Baseline/evaluation_metrics.txt"
    save_path="data_augmentation/evaluation_metrics_augmented_new.txt"
)
