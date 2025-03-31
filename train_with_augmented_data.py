import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor
import matplotlib.pyplot as plt
import rasterio
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import Counter
from utils import evaluate_model

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 工具函数
def tile_image(image, size):
    tiles = []
    h, w = image.shape[:2]
    for y in range(0, h, size):
        for x in range(0, w, size):
            if y + size <= h and x + size <= w:
                tiles.append(image[y:y+size, x:x+size])
    return tiles

def load_rgb_tif(filepath):
    with rasterio.open(filepath) as src:
        img = src.read([1, 2, 3])
        img = np.transpose(img, (1, 2, 0))
    return img.astype(np.uint8)

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
    A.RandomSizedCrop(min_max_height=(400, 500), size=(500, 500), p=0.3),
    A.CoarseDropout(max_holes=8, max_height=40, max_width=40, fill_value=0, mask_fill_value=0, p=0.3),
    A.Normalize(),
    ToTensorV2()
])

# 数据集类
class SegmentationDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask'].long()
        else:
            image = to_tensor(image)
            mask = torch.from_numpy(mask).long()
        return image, mask

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
image_paths = [r"C:\cartography\project\Baseline\data\rgb_TA_138_1930.tif",
               r"C:\cartography\project\Baseline\data\rgb_TA_316_1918.tif"]
mask_paths = [r"C:\cartography\project\Baseline\data\training_target_s1930.png",
              r"C:\cartography\project\Baseline\data\training_target_s1918.png"]
tile_size = 500

# crop images
image_tiles, mask_tiles = [], []
for img_path, msk_path in zip(image_paths, mask_paths):
    image = load_rgb_tif(img_path)
    mask = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
    image_tiles += tile_image(image, tile_size)
    mask_tiles += tile_image(mask, tile_size)

# 复制湖泊和湿地 tiles
lake_imgs, lake_msks = extract_class_tiles(image_tiles, mask_tiles, class_ids=[4], extra_copies=3)
wetland_imgs, wetland_msks = extract_class_tiles(image_tiles, mask_tiles, class_ids=[6], extra_copies=5)
image_tiles += lake_imgs + wetland_imgs
mask_tiles += lake_msks + wetland_msks


# Compute class weights from class frequency
def compute_class_weights(masks, n_classes):
    pixel_counter = Counter()
    total_pixels = 0
    for mask in masks:
        unique, counts = np.unique(mask, return_counts=True)
        for u, c in zip(unique, counts):
            pixel_counter[u] += c
            total_pixels += c
    freqs = np.array([pixel_counter[i] / total_pixels if i in pixel_counter else 0.0 for i in range(n_classes)])
    weights = 1.0 / (freqs + 1e-6)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)

n_classes = 8
class_weights = compute_class_weights(mask_tiles, n_classes).to(device)
print("Computed Class Weights:", class_weights)


# 构造数据集和加载器
dataset = SegmentationDataset(image_tiles, mask_tiles, transform=train_transform)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# 初始化模型和训练组件
model = UNet(n_classes=8).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=False, min_lr=1e-5)

# 正式训练
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

# 保存模型
torch.save(model.state_dict(), "unet_model_data_augmented.pth")


# visualization the results
model.eval()
indices = [179,195]
for i, idx in enumerate(indices):
    #idx = random.randint(0, len(dataset) - 1)
    img, true_mask = dataset[idx]
    with torch.no_grad():
        pred_mask = model(img.unsqueeze(0).to(device)).argmax(1).squeeze().cpu()

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(img.permute(1, 2, 0))
    ax[0].set_title(f"Image (Tile {idx})")
    ax[1].imshow(pred_mask, cmap='tab20')
    ax[1].set_title("Prediction")
    ax[2].imshow(true_mask, cmap='tab20')
    ax[2].set_title("Ground Truth")
    for a in ax: a.axis("off")
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(f"data_augmentation/augmented_compare_{i}.png")
    plt.close()


# Compute metrics
evaluate_model(
    model=model,
    image_tiles=image_tiles,
    mask_tiles=mask_tiles,
    save_path="data_augmentation/evaluation_metrics_augmented.txt"
)


