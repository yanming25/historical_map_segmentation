from sklearn.metrics import f1_score, jaccard_score, accuracy_score, precision_score, recall_score
import numpy as np
from torchvision.transforms.functional import to_tensor
import torch
from torch.utils.data import Dataset
import rasterio
from torchvision import transforms
from skimage.morphology import skeletonize, binary_dilation, disk
from collections import Counter
import matplotlib.pyplot as plt
import random


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
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask'].long()
        else:
            image = to_tensor(image)
            mask = torch.from_numpy(mask).long()
        return image, mask


def load_rgb_tif(filepath):
    with rasterio.open(filepath) as src:
        img = src.read([1, 2, 3])
        img = np.transpose(img, (1, 2, 0))
    return img.astype(np.uint8)


# def evaluate_model_adjust(model, image_tiles, mask_tiles, save_path):
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     class_names = ['Background', 'Road', 'Stream', 'Forest', 'Lakes', 'Rivers', 'Wetlands', 'Buildings']
#     n_classes = len(class_names)
#
#     y_true = np.concatenate(np.array([m.flatten() for m in mask_tiles]))
#     y_pred = np.concatenate(np.array([
#         model(to_tensor(img).unsqueeze(0).to(device)).argmax(1).squeeze().cpu().numpy().flatten()
#         for img in image_tiles
#     ]))
#
#     f1 = f1_score(y_true, y_pred, average=None, labels=range(n_classes))
#     iou = jaccard_score(y_true, y_pred, average=None, labels=range(n_classes))
#     precision = precision_score(y_true, y_pred, average=None, labels=range(n_classes))
#     recall = recall_score(y_true, y_pred, average=None, labels=range(n_classes))
#
#     pixel_acc = accuracy_score(y_true, y_pred)
#     mean_f1 = f1.mean()
#     mean_iou = iou.mean()
#
#     with open(save_path, "w") as f:
#         f.write("===== Overall Metrics =====\n")
#         f.write(f"Pixel Accuracy: {pixel_acc:.4f}\n")
#         f.write(f"Mean F1 Score:  {mean_f1:.4f}\n")
#         f.write(f"Mean IoU:       {mean_iou:.4f}\n\n")
#
#         f.write("===== Per-Class Metrics =====\n")
#         for i in range(n_classes):
#             f.write(f"{class_names[i]:<10} | IoU: {iou[i]:.4f} | F1: {f1[i]:.4f} | "
#                     f"Precision: {precision[i]:.4f} | Recall: {recall[i]:.4f}\n")

def to_tensor(img):
    """Converts a numpy image to a normalized tensor"""
    return transforms.ToTensor()(img)

def binary_skeleton(arr):
    return skeletonize(arr > 0)

def buffer_match(pred_center, gt_center, buffer_radius=8):
    buffer = binary_dilation(gt_center, disk(buffer_radius))
    TP = np.logical_and(pred_center, buffer).sum()
    FP = np.logical_and(pred_center, np.logical_not(buffer)).sum()

    buffer_pred = binary_dilation(pred_center, disk(buffer_radius))
    FN = np.logical_and(gt_center, np.logical_not(buffer_pred)).sum()
    return TP, FP, FN

def evaluate_model(model, image_tiles, mask_tiles, save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ['Background', 'Road', 'Stream', 'Forest', 'Lakes', 'Rivers', 'Wetlands', 'Buildings']
    n_classes = len(class_names)

    # Predict all
    y_true = np.concatenate(np.array([m.flatten() for m in mask_tiles]))
    y_pred = np.concatenate(np.array([
        model(to_tensor(img).unsqueeze(0).to(device)).argmax(1).squeeze().cpu().numpy().flatten()
        for img in image_tiles
    ]))

    # Standard pixel-wise metrics
    f1 = f1_score(y_true, y_pred, average=None, labels=range(n_classes))
    iou = jaccard_score(y_true, y_pred, average=None, labels=range(n_classes))
    precision = precision_score(y_true, y_pred, average=None, labels=range(n_classes))
    recall = recall_score(y_true, y_pred, average=None, labels=range(n_classes))

    pixel_acc = accuracy_score(y_true, y_pred)
    mean_f1 = f1.mean()
    mean_iou = iou.mean()

    with open(save_path, "w") as f:
        f.write("===== Overall Metrics =====\n")
        f.write(f"Pixel Accuracy: {pixel_acc:.4f}\n")
        f.write(f"Mean F1 Score:  {mean_f1:.4f}\n")
        f.write(f"Mean IoU:       {mean_iou:.4f}\n\n")

        f.write("===== Per-Class Metrics (Standard) =====\n")
        for i in range(n_classes):
            f.write(f"{class_names[i]:<10} | IoU: {iou[i]:.4f} | F1: {f1[i]:.4f} | "
                    f"Precision: {precision[i]:.4f} | Recall: {recall[i]:.4f}\n")

        # Custom buffer-based metrics for Road (1) and Stream (2)
        f.write("\n===== Centerline-based Metrics (Buffer=8 pixels) =====\n")
        for class_id in [1, 2]:  # Road, Stream
            pred_mask = []
            gt_mask = []
            for img in image_tiles:
                pred = model(to_tensor(img).unsqueeze(0).to(device)).argmax(1).squeeze().cpu().numpy()
                pred_mask.append((pred == class_id).astype(np.uint8))
            for gt in mask_tiles:
                gt_mask.append((gt == class_id).astype(np.uint8))

            TP_total, FP_total, FN_total = 0, 0, 0
            for p, g in zip(pred_mask, gt_mask):
                pred_skel = binary_skeleton(p)
                gt_skel = binary_skeleton(g)
                TP, FP, FN = buffer_match(pred_skel, gt_skel, buffer_radius=8)
                TP_total += TP
                FP_total += FP
                FN_total += FN

            if TP_total + FP_total > 0:
                precision_val = TP_total / (TP_total + FP_total)
            else:
                precision_val = 0.0
            if TP_total + FN_total > 0:
                recall_val = TP_total / (TP_total + FN_total)
            else:
                recall_val = 0.0
            if precision_val + recall_val > 0:
                f1_val = 2 * precision_val * recall_val / (precision_val + recall_val)
            else:
                f1_val = 0.0
            if (TP_total + FP_total + FN_total) > 0:
                dice_val = 2 * TP_total / (2 * TP_total + FP_total + FN_total)
            else:
                dice_val = 0.0

            f.write(f"{class_names[class_id]:<10} | Dice: {dice_val:.4f} | F1: {f1_val:.4f} | "
                    f"Precision: {precision_val:.4f} | Recall: {recall_val:.4f}\n")


# Compute class weights from class frequency
def compute_class_weights(masks, n_classes):
    pixel_counter = Counter()
    total_pixels = 0
    for mask in masks:
        unique, counts = np.unique(mask, return_counts=True)
        for u, c in zip(unique, counts):
            pixel_counter[u] += c
            total_pixels += c

    freqs = np.array([pixel_counter[i] / total_pixels if i in pixel_counter else 1e-6 for i in range(n_classes)])
    weights = 1.0 / (np.log(1.02 + freqs))
    weights = weights / weights.mean()
    weights = np.clip(weights, 0.5, 2.0)
    return torch.tensor(weights, dtype=torch.float32)


def visualize_resnet(model, dataset, indices, device, save_path, model_name, setting="traning"):
    for i, idx in enumerate(indices):
        idx = random.randint(0, len(dataset) - 1)
        img, true_mask = dataset[idx]

        with torch.no_grad():
            pred_mask = model(img.unsqueeze(0).to(device)).argmax(1).squeeze().cpu()

        print(f"Tile {idx} - Predicted classes:", np.unique(pred_mask.numpy()))
        print(f"Tile {idx} - Ground truth classes:", np.unique(true_mask.numpy()))

        # Create a figure with 3 subplots: Image, Prediction, Ground Truth
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
        plt.savefig(f"{save_path}/{setting}_compare_{i}_{model_name}.png")
        plt.close()