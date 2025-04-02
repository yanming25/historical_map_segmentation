import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import evaluate_model, tile_image, load_rgb_tif, SegmentationDataset
from PIL import Image


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 工具函数
def extract_class_tiles(images, masks, class_ids, extra_copies=1):
    selected_imgs, selected_masks = [], []
    for img, msk in zip(images, masks):
        unique_vals = np.unique(msk)
        if any(cid in unique_vals for cid in class_ids):
            for _ in range(extra_copies):
                selected_imgs.append(img.copy())
                selected_masks.append(msk.copy())
    return selected_imgs, selected_masks

# 增强器
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
    # A.RandomSizedCrop(min_max_height=(400, 500), size=(500, 500), p=0.3),
    A.CoarseDropout(max_holes=8, max_height=40, max_width=40, fill_value=0, mask_fill_value=0, p=0.3),
    A.Normalize(),
    ToTensorV2()
])


# U-Net模型结构
class UNet(nn.Module):
    def __init__(self, n_classes):
        super(UNet, self).__init__()
        def CBR(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        self.enc1 = nn.Sequential(CBR(3, 64), CBR(64, 64))
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(CBR(64, 128), CBR(128, 128))
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(CBR(128, 256), CBR(256, 256))
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = nn.Sequential(CBR(256, 128), CBR(128, 64))
        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec1 = CBR(128, 64)
        self.final = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        return self.final(d1)

# 数据加载
image_paths = [r"C:\cartography\project\data\rgb_TA_138_1930.tif",
               r"C:\cartography\project\data\rgb_TA_316_1918.tif"]
mask_paths = [r"C:\cartography\project\data\training_target_s1930.png",
              r"C:\cartography\project\data\training_target_s1918.png"]
tile_size = 500

# crop images
image_tiles, mask_tiles = [], []
for img_path, msk_path in zip(image_paths, mask_paths):
    image = load_rgb_tif(img_path)
    mask = Image.open(msk_path).convert("L")
    mask = np.array(mask)
    print(f"{msk_path}'s Mask unique values: {np.unique(mask)}")
    image_tiles += tile_image(image, tile_size)
    mask_tiles += tile_image(mask, tile_size)

# 复制湖泊和湿地 tiles
stream_imgs, stream_msks = extract_class_tiles(image_tiles, mask_tiles, class_ids=[2], extra_copies=2)
lake_imgs, lake_msks = extract_class_tiles(image_tiles, mask_tiles, class_ids=[4], extra_copies=3)
wetland_imgs, wetland_msks = extract_class_tiles(image_tiles, mask_tiles, class_ids=[6], extra_copies=8)
building_imgs, building_msks = extract_class_tiles(image_tiles, mask_tiles, class_ids=[7], extra_copies=3)
image_tiles += lake_imgs + wetland_imgs + stream_imgs + building_imgs
mask_tiles += lake_msks + wetland_msks + stream_msks + building_msks


# # Visualize some image and mask samples to check
# import random
# indices_to_check = random.sample(range(len(image_tiles)), 50)
# for idx in indices_to_check:
#     img = image_tiles[idx]
#     mask = mask_tiles[idx]
#
#     fig, ax = plt.subplots(1, 2, figsize=(10, 4))
#     ax[0].imshow(img)
#     ax[0].set_title(f"Image Tile {idx}")
#     ax[1].imshow(mask, cmap='tab20')
#     ax[1].set_title(f"Mask Tile {idx}")
#     for a in ax: a.axis('off')
#     plt.tight_layout()
#     plt.savefig(f"sample_check/sample_tile_{idx}.png")
#     plt.close()


n_classes = 8
# class_weights = compute_class_weights(mask_tiles, n_classes).to(device)
# print("Computed Class Weights:", class_weights)


# 构造数据集和加载器
dataset = SegmentationDataset(image_tiles, mask_tiles, transform=train_transform)
loader = DataLoader(dataset, batch_size=8, shuffle=True)


# Visualize to check the samples after augmentation
import random
random.seed(42)
indices_to_check = random.sample(range(len(dataset)), 50)
for idx in indices_to_check:
    img_tensor, mask_tensor = dataset[idx]

    # # ------- 检查 mask 是否为整数类别 -------
    # unique_mask_vals = torch.unique(mask_tensor).cpu().numpy()
    # print(f"[{idx}] Mask unique values:", unique_mask_vals)
    #
    # # 检查是否存在非整数类别
    # if not torch.all(mask_tensor == mask_tensor.long()):
    #     print(f"⚠️ WARNING: Non-integer values found in mask for idx {idx}")
    #
    # # ------- 检查 image 是否归一化 -------
    # img_min, img_max = img_tensor.min().item(), img_tensor.max().item()
    # print(f"[{idx}] Image value range: min={img_min:.3f}, max={img_max:.3f}")

    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    mask = mask_tensor.cpu().numpy()

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].imshow(img)
    ax[0].set_title(f"Augmented Image {idx}")
    ax[1].imshow(mask, cmap='tab20')
    ax[1].set_title(f"Augmented Mask {idx}")
    for a in ax: a.axis('off')
    plt.tight_layout()
    plt.savefig(f"augmented_sample_check/augmented_tile_{idx}.png")
    plt.close()


# Initialize model and hyperparameters
model = UNet(n_classes).to(device)

#criterion = nn.CrossEntropyLoss(weight=class_weights)
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-5)


n_epochs = 30
for epoch in range(n_epochs):
    model.train()
    running_loss = 0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(loader)
    print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")
    scheduler.step(avg_loss)

# save model
torch.save(model.state_dict(), "unet_model_data_augmented_aug_copy.pth")


# # visualize two prediction samples from training set (with augmentation)
# model.eval()
# indices = [179,195]
# for i, idx in enumerate(indices):
#     #idx = random.randint(0, len(dataset) - 1)
#     img, true_mask = dataset[idx]
#     with torch.no_grad():
#         pred_mask = model(img.unsqueeze(0).to(device)).argmax(1).squeeze().cpu()
#
#     print(f"Tile {idx} - Predicted classes:", np.unique(pred_mask.numpy()))
#     print(f"Tile {idx} - Ground truth classes:", np.unique(true_mask.numpy()))
#
#     fig, ax = plt.subplots(1, 3, figsize=(12, 4))
#     ax[0].imshow(img.permute(1, 2, 0))
#     ax[0].set_title(f"Image (Tile {idx})")
#     ax[1].imshow(pred_mask, cmap='tab20')
#     ax[1].set_title("Prediction")
#     ax[2].imshow(true_mask, cmap='tab20')
#     ax[2].set_title("Ground Truth")
#     for a in ax: a.axis("off")
#     plt.tight_layout()
#     plt.subplots_adjust(top=0.9)
#     plt.savefig(f"data_augmentation/training_compare_{i}_new.png")
#     plt.close()
#
#
# # visualize two prediction samples from training set (without augmentation)
# dataset_no_aug = SegmentationDataset(image_tiles, mask_tiles, transform=None)
# indices = [179, 195]
#
# for i, idx in enumerate(indices):
#     img_np, true_mask_np = dataset_no_aug[idx]
#     img_tensor = img_np.unsqueeze(0).to(device)
#
#     with torch.no_grad():
#         pred_mask = model(img_tensor).argmax(1).squeeze().cpu()
#
#     print(f"Tile {idx} - Predicted classes:", np.unique(pred_mask.numpy()))
#     print(f"Tile {idx} - Ground truth classes:", np.unique(true_mask_np))
#
#     fig, ax = plt.subplots(1, 3, figsize=(12, 4))
#     ax[0].imshow(img_np.permute(1, 2, 0))
#     ax[0].set_title(f"Original Image (Tile {idx})")
#     ax[1].imshow(pred_mask, cmap='tab20')
#     ax[1].set_title("Prediction")
#     ax[2].imshow(true_mask_np, cmap='tab20')
#     ax[2].set_title("Ground Truth")
#     for a in ax:
#         a.axis("off")
#     plt.tight_layout()
#     plt.subplots_adjust(top=0.9)
#     plt.savefig(f"data_augmentation/training_compare_{idx}_raw_no_aug.png")
#     plt.close()


# Compute metrics
model.eval()
evaluate_model(
    model=model,
    image_tiles=image_tiles,
    mask_tiles=mask_tiles,
    save_path="data_augmentation/evaluation_metrics_aug_copy.txt"
)

