from sklearn.metrics import f1_score, jaccard_score, accuracy_score, precision_score, recall_score
import numpy as np
from torchvision.transforms.functional import to_tensor
import torch
from torch.utils.data import Dataset
import rasterio


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


def load_rgb_tif(filepath):
    with rasterio.open(filepath) as src:
        img = src.read([1, 2, 3])
        img = np.transpose(img, (1, 2, 0))
    return img.astype(np.uint8)


def evaluate_model(model, image_tiles, mask_tiles, save_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ['Background', 'Road', 'Stream', 'Forest', 'Lakes', 'Rivers', 'Wetlands', 'Buildings']
    n_classes = len(class_names)

    y_true = np.concatenate(np.array([m.flatten() for m in mask_tiles]))
    y_pred = np.concatenate(np.array([
        model(to_tensor(img).unsqueeze(0).to(device)).argmax(1).squeeze().cpu().numpy().flatten()
        for img in image_tiles
    ]))

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

        f.write("===== Per-Class Metrics =====\n")
        for i in range(n_classes):
            f.write(f"{class_names[i]:<10} | IoU: {iou[i]:.4f} | F1: {f1[i]:.4f} | "
                    f"Precision: {precision[i]:.4f} | Recall: {recall[i]:.4f}\n")
