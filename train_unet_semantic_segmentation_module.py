import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor
import matplotlib.pyplot as plt
import random
import rasterio
from PIL import Image
from torch.optim.lr_scheduler import ReduceLROnPlateau


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ========= 可被外部 import 的模块 =========

def tile_image(image, size):
    tiles = []
    h, w = image.shape[:2]
    for y in range(0, h, size):
        for x in range(0, w, size):
            if y + size <= h and x + size <= w:
                tiles.append(image[y:y+size, x:x+size])
    return tiles

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
            image = self.transform(image)
        else:
            image = to_tensor(image)
        mask = torch.from_numpy(mask).long()
        return image, mask

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

# ========= 主训练流程 =========

if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    image_paths = [
        r"D:\ETH_Master\02SpringSemester2025\Research Topics in Cartography\project\Baseline\data\training_image_s1930.tif",
        r"D:\ETH_Master\02SpringSemester2025\Research Topics in Cartography\project\Baseline\data\training_image_s1918.tif"
    ]
    mask_paths = [
        r"D:\ETH_Master\02SpringSemester2025\Research Topics in Cartography\project\Baseline\data\training_target_s1930.png",
        r"D:\ETH_Master\02SpringSemester2025\Research Topics in Cartography\project\Baseline\data\training_target_s1918.png"
    ]
    tile_size = 500

    def load_rgb_tif(filepath):
        with rasterio.open(filepath) as src:
            img = src.read([1, 2, 3])
            img = np.transpose(img, (1, 2, 0))
        return img.astype(np.uint8)

    image_tiles, mask_tiles = [], []
    for img_path, msk_path in zip(image_paths, mask_paths):
        image = load_rgb_tif(img_path)
        mask = Image.open(msk_path).convert("L")
        mask = np.array(mask)
        print(f"{msk_path}'s Mask unique values: {np.unique(mask)}")
        image_tiles += tile_image(image, tile_size)
        mask_tiles += tile_image(mask, tile_size)

    dataset = SegmentationDataset(image_tiles, mask_tiles)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(n_classes=8).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-5)

    n_epochs = 10
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
        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}")
        scheduler.step(avg_loss)

    torch.save(model.state_dict(), "unet_model.pth")

    # 随机可视化两组结果
    model.eval()
    indices = random.sample(range(len(dataset)), 2)
    for i, idx in enumerate(indices):
        img, true_mask = dataset[idx]
        with torch.no_grad():
            pred_mask = model(img.unsqueeze(0).to(device)).argmax(1).squeeze().cpu()

        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax[0].imshow(img.permute(1, 2, 0))
        ax[0].set_title("Image")
        ax[1].imshow(pred_mask, cmap='tab20')
        ax[1].set_title("Prediction")
        ax[2].imshow(true_mask, cmap='tab20')
        ax[2].set_title("Ground Truth")
        plt.tight_layout()
        plt.savefig(f"compare_{i}.png")
        plt.close()