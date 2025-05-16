import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import rasterio
from torchvision.transforms.functional import to_tensor
from PIL import Image

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- Utilities ---
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

# --- Dataset ---
class SegmentationDataset(Dataset):
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = to_tensor(self.images[idx])
        mask = torch.from_numpy(self.masks[idx]).long()
        return image, mask

# --- U-Net Model ---
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

# Load pretrained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(n_classes=8).to(device)
model.load_state_dict(torch.load("unet_finetuned_on_siegfried_newtarget2.pth"))

# Load Siegfried data
image_paths = [
        r"D:\ETH_Master\02SpringSemester2025\Research Topics in Cartography\project\Baseline\data\rgb_TA_138_1930.tif",
        r"D:\ETH_Master\02SpringSemester2025\Research Topics in Cartography\project\Baseline\data\rgb_TA_316_1918.tif"
]
mask_paths = [
        r"D:\ETH_Master\02SpringSemester2025\Research Topics in Cartography\project\Baseline\data\training_target_s1930.png",
        r"D:\ETH_Master\02SpringSemester2025\Research Topics in Cartography\project\Baseline\data\training_target_s1918.png"
]
image_tiles, mask_tiles = [], []

for img_path, msk_path in zip(image_paths, mask_paths):
    img = load_rgb_tif(img_path)
    msk = Image.open(msk_path).convert("L")
    msk = np.array(msk)
    image_tiles += tile_image(img, 500)
    mask_tiles += tile_image(msk, 500)

siegfried_dataset = SegmentationDataset(image_tiles, mask_tiles)
siegfried_loader = DataLoader(siegfried_dataset, batch_size=8, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Fine-tuning
for epoch in range(3):
    model.train()
    running_loss = 0
    for imgs, masks in siegfried_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        output = model(imgs)
        loss = criterion(output, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"[Finetune] Epoch {epoch+1}, Loss: {running_loss/len(siegfried_loader):.4f}")

torch.save(model.state_dict(), "unet_finetuned_on_siegfried_newtarget2.pth")

# Visualize 2 predictions
model.eval()
for i in range(2):
    img, true_mask = siegfried_dataset[i]
    with torch.no_grad():
        pred_mask = model(img.unsqueeze(0).to(device)).argmax(1).squeeze().cpu()
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(img.permute(1, 2, 0))
    ax[0].set_title("Image")
    ax[1].imshow(pred_mask, cmap='tab20')
    ax[1].set_title("Prediction")
    ax[2].imshow(true_mask, cmap='tab20')
    ax[2].set_title("Ground Truth")
    for a in ax: a.axis('off')
    plt.tight_layout()
    plt.savefig(f"finetune_result_newtarget1_{i}.png")
    plt.close()