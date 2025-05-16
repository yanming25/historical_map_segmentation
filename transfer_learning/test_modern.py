import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from torchvision.transforms.functional import to_tensor
from train_unet_semantic_segmentation_module import UNet, SegmentationDataset, tile_image
import cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import rasterio
from utils import evaluate_model

# ========== Step 1: Load Trained Model ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(n_classes=8)
model.load_state_dict(torch.load("unet_trained_on_modern_maps_new_add_data.pth", map_location=device))
model.to(device)
model.eval()

# ========== Step 2: Load and Tile the Same Images ==========
def load_rgb_tif(filepath):
    with rasterio.open(filepath) as src:
        img = src.read([1, 2, 3])
        img = np.transpose(img, (1, 2, 0))
    return img.astype(np.uint8)

image_paths = [r"D:\ETH_Master\02SpringSemester2025\Research Topics in Cartography\project\Baseline\data\swiss-map-raster25_2020_1184_krel_1.25_2056.tif",
               r"D:\ETH_Master\02SpringSemester2025\Research Topics in Cartography\project\Baseline\data\swiss-map-raster25_2021_1165_krel_1.25_2056.tif"]
mask_paths = [r"D:\ETH_Master\02SpringSemester2025\Research Topics in Cartography\project\Baseline\data\test_target_1184_modern1.png",
              r"D:\ETH_Master\02SpringSemester2025\Research Topics in Cartography\project\Baseline\data\test_target_1165_modern2.png"]
# image_paths = [
#         r"D:\ETH_Master\02SpringSemester2025\Research Topics in Cartography\project\Baseline\data\rgb_TA_138_1930.tif",
#         r"D:\ETH_Master\02SpringSemester2025\Research Topics in Cartography\project\Baseline\data\rgb_TA_316_1918.tif"
# ]
# mask_paths = [
#         r"D:\ETH_Master\02SpringSemester2025\Research Topics in Cartography\project\Baseline\data\training_target_s1930.png",
#         r"D:\ETH_Master\02SpringSemester2025\Research Topics in Cartography\project\Baseline\data\training_target_s1918.png"
# ]

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
    ax[0].set_title("Image")
    ax[1].imshow(pred_mask, cmap='tab20')
    ax[1].set_title("Prediction")
    ax[2].imshow(true_mask, cmap='tab20')
    ax[2].set_title("Ground Truth")
    for a in ax: a.axis("off")
    plt.tight_layout()
    plt.savefig(f"prediction_Siegfried_on_modernmodel_testing_ADDdata_{i}.png")
    plt.close()

evaluate_model(model,image_tiles,mask_tiles,"modern_unet_evaluation_new_add_data.txt")