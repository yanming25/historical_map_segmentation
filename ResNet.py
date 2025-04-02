import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.models as models
from torchvision.transforms.functional import to_tensor
import matplotlib.pyplot as plt
import rasterio
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import Counter
from PIL import Image

import utils
from utils import evaluate_model

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class UNetResNet(nn.Module):
    def __init__(self, n_classes, backbone="resnet34"):
        super(UNetResNet, self).__init__()

        # Load a pretrained ResNet model
        if backbone == "resnet34":
            self.encoder = models.resnet34(pretrained=True)
            filters = [64, 128, 256, 512]  # ResNet34 feature sizes
        elif backbone == "resnet50":
            self.encoder = models.resnet50(pretrained=True)
            filters = [64, 256, 512, 1024, 2048]  # ResNet50 feature sizes
        else:
            raise ValueError("Supports resnet34 and resnet50")

        self.encoder.fc = nn.Identity() # Remove original fully connected layer

        # U-Net decoder
        self.up4 = self._upsample(filters[-1], filters[-2])
        self.up3 = self._upsample(filters[-2], filters[-3])
        self.up2 = self._upsample(filters[-3], filters[-4])
        self.up1 = self._upsample(filters[-4], 64)

        self.final = nn.Conv2d(64, n_classes, kernel_size=1) # Output layer

    def _upsample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True))

    def forward(self, x):
        # Encoder (ResNet)
        x1 = self.encoder.conv1(x)  # Initial conv layer
        x1 = self.encoder.bn1(x1)
        x1 = self.encoder.relu(x1)
        x1 = self.encoder.maxpool(x1)

        x2 = self.encoder.layer1(x1)  # 64 filters
        x3 = self.encoder.layer2(x2)  # 128 filters
        x4 = self.encoder.layer3(x3)  # 256 filters
        x5 = self.encoder.layer4(x4)  # 512 filters (ResNet34) / 1024 (ResNet50)

        # Decoder (upsampling)
        d4 = self.up4(x5)
        d4 = torch.cat([d4, x4], dim=1)
        d3 = self.up3(d4)
        d3 = torch.cat([d3, x3], dim=1)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, x2], dim=1)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, x1], dim=1)

        return self.final(d1)

image_paths = [r"C:\cartography\project\Baseline\data\rgb_TA_138_1930.tif",
               r"C:\cartography\project\Baseline\data\rgb_TA_316_1918.tif"]
mask_paths = [r"C:\cartography\project\Baseline\data\training_target_s1930.png",
              r"C:\cartography\project\Baseline\data\training_target_s1918.png"]
tile_size = 500
n_classes = 8

image_tiles, mask_tiles = [], []
for img_path, msk_path in zip(image_paths, mask_paths):
    image = utils.load_rgb_tif(img_path)
    mask = Image.open(msk_path).convert("L")
    mask = np.array(mask)
    print(f"{msk_path}'s Mask unique values: {np.unique(mask)}")

    image_tiles += utils.tile_image(image, tile_size)
    mask_tiles += utils.tile_image(mask, tile_size)

# Load the dataset
dataset = utils.SegmentationDataset(image_tiles, mask_tiles, transform=ToTensorV2())
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Initial model
model = UNetResNet(n_classes=8, backbone="resnet34").to(device) # ResNet34 backbone
# backbone="resnet50" for ResNet50

# 1. Freeze encoder layers (ResNet)
for param in model.encoder.parameters():
    param.requires_grad = False  # Freeze ResNet weights

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# Training loop (Decoder only)
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

# Save the trained model
torch.save(model.state_dict(), "unet_resnet_decoder.pth")

utils.evaluate_model(model, image_tiles, mask_tiles, save_path="unet_resnet_decoder.txt")

# 2. Fine tuning (Unfreeze partial encoder layers)
for param in model.encoder.layer3.parameters():
    param.requires_grad = True  # Unfreeze ResNet layer3
for param in model.encoder.layer4.parameters():
    param.requires_grad = True  # Unfreeze ResNet layer4

# Reduce learning rate for fine-tuning
optimizer = optim.Adam(model.parameters(), lr=1e-4) # avoid overwriting

n_finetune_epochs = 10
for epoch in range(n_finetune_epochs):
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
    print(f"Fine-tune Epoch {epoch + 1}/{n_finetune_epochs}, Loss: {avg_loss:.4f}")
    scheduler.step(avg_loss)

utils.evaluate_model(model, image_tiles, mask_tiles, save_path="unet_resnet_encoder3_4.txt")

# 3. Fine tuning (Unfreeze encoder layers)
for param in model.encoder.parameters():
    param.requires_grad = True  # Unfreeze all ResNet layers

optimizer = optim.Adam(model.parameters(), lr=1e-5)  # very small LR

# Fine-tune entire model
n_finetune_epochs = 5 #(10)
for epoch in range(n_finetune_epochs):
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
    print(f"Fine-tune Epoch {epoch + 1}/{n_finetune_epochs}, Loss: {avg_loss:.4f}")
    scheduler.step(avg_loss)

torch.save(model.state_dict(), "unet_resnet_finetuned.pth")
utils.evaluate_model(model, image_tiles, mask_tiles, save_path="unet_resnet_finetuned.txt")

