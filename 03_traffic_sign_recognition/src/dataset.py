"""
dataset.py
GTSRB dataset loader, class mappings, stratified splitting, and visualization utilities.
"""

import os
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

GTSRB_CLASSES = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing for vehicles > 3.5t",
    11: "Right-of-way at intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles > 3.5t prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve left",
    20: "Dangerous curve right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed & passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing > 3.5t",
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class GTSRBDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root_dir: str, transform=None, crop_roi: bool = True):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.crop_roi = crop_roi

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        img_rel_path = row["Path"]
        full_path = os.path.join(self.root_dir, img_rel_path)
        class_id = int(row["ClassId"])

        with Image.open(full_path) as img:
            img = img.convert("RGB")
            if self.crop_roi and "Roi.X1" in row:
                x1, y1 = int(row["Roi.X1"]), int(row["Roi.Y1"])
                x2, y2 = int(row["Roi.X2"]), int(row["Roi.Y2"])
                if x2 > x1 and y2 > y1:
                    img = img.crop((x1, y1, x2, y2))

            if self.transform is not None:
                tensor_img = self.transform(img)
            else:
                tensor_img = transforms.ToTensor()(img)

        return tensor_img, class_id


def get_gtsrb_dataloaders(
    raw_dir: str,
    batch_size: int = 64,
    img_size: int = 48,
    train_limit: Optional[int] = 10000,
    test_limit: Optional[int] = 2000,
    val_ratio: float = 0.15,
    seed: int = 42,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    train_csv_path = os.path.join(raw_dir, "Train.csv")
    test_csv_path = os.path.join(raw_dir, "Test.csv")

    if not os.path.exists(train_csv_path) or not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Missing Train.csv or Test.csv in {raw_dir}")

    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)

    total_raw_train = len(train_df)
    total_raw_test = len(test_df)

    # Subsample training set if limit provided (stratified sampling)
    np.random.seed(seed)
    if train_limit is not None and train_limit < len(train_df):
        sub_dfs = []
        for _, group in train_df.groupby("ClassId"):
            n = min(len(group), max(5, int(train_limit * len(group) / total_raw_train)))
            sub_dfs.append(group.sample(n, random_state=seed))
        train_df = pd.concat(sub_dfs, ignore_index=True)

    # Stratified Train/Val split
    train_df = train_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_dfs, tr_dfs = [], []
    for _, group in train_df.groupby("ClassId"):
        n_val = max(1, int(len(group) * val_ratio))
        val_sub = group.iloc[:n_val]
        tr_sub = group.iloc[n_val:]
        val_dfs.append(val_sub)
        tr_dfs.append(tr_sub)
    final_train_df = pd.concat(tr_dfs, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(val_dfs, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Subsample test set if limit provided
    if test_limit is not None and test_limit < len(test_df):
        test_sub_dfs = []
        for _, group in test_df.groupby("ClassId"):
            n = min(len(group), max(2, int(test_limit * len(group) / total_raw_test)))
            test_sub_dfs.append(group.sample(n, random_state=seed))
        test_df = pd.concat(test_sub_dfs, ignore_index=True).reset_index(drop=True)

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_ds = GTSRBDataset(final_train_df, raw_dir, transform=train_transform)
    val_ds = GTSRBDataset(val_df, raw_dir, transform=eval_transform)
    test_ds = GTSRBDataset(test_df, raw_dir, transform=eval_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    stats = {
        "raw_train_total": total_raw_train,
        "raw_test_total": total_raw_test,
        "used_train_samples": len(final_train_df),
        "used_val_samples": len(val_df),
        "used_test_samples": len(test_df),
        "class_distribution_raw": train_df["ClassId"].value_counts().to_dict(),
    }

    return train_loader, val_loader, test_loader, stats


def plot_gtsrb_class_distribution(raw_dir: str, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    train_csv = os.path.join(raw_dir, "Train.csv")
    df = pd.read_csv(train_csv)
    counts = df["ClassId"].value_counts().sort_index()

    plt.figure(figsize=(14, 6))
    bars = plt.bar(counts.index, counts.values, color="#1f77b4", edgecolor="black", alpha=0.85)
    plt.xlabel("Class ID (0 - 42)", fontsize=12)
    plt.ylabel("Number of Training Images", fontsize=12)
    plt.title("GTSRB Dataset Class Distribution (39,209 Training Images)", fontsize=14, fontweight="bold")
    plt.xticks(range(0, 43, 2))
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[✓] Saved GTSRB class distribution plot to: {output_path}")


def save_gtsrb_samples(raw_dir: str, output_path: str):
    """Saves a 4x4 visual grid of representative traffic signs with readable class labels."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    train_csv = os.path.join(raw_dir, "Train.csv")
    df = pd.read_csv(train_csv)

    selected_ids = [0, 1, 9, 11, 12, 13, 14, 17, 18, 25, 27, 33, 35, 38, 40, 41]
    fig, axes = plt.subplots(4, 4, figsize=(14, 12))
    axes = axes.flatten()

    for idx, cid in enumerate(selected_ids):
        ax = axes[idx]
        sample_row = df[df["ClassId"] == cid].iloc[0]
        full_p = os.path.join(raw_dir, sample_row["Path"])
        x1, y1, x2, y2 = int(sample_row["Roi.X1"]), int(sample_row["Roi.Y1"]), int(sample_row["Roi.X2"]), int(sample_row["Roi.Y2"])
        img = Image.open(full_p).convert("RGB").crop((x1, y1, x2, y2))
        ax.imshow(img)
        title = f"Class {cid}:\n{GTSRB_CLASSES[cid]}"
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.axis("off")

    plt.suptitle("German Traffic Sign Recognition Benchmark (GTSRB) Sample Visuals", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[✓] Saved GTSRB sample visuals to: {output_path}")
