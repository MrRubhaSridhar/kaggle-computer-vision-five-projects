"""
dataset.py
Dataset exploration, stratified split, transforms, and DataLoaders for Fruit Quality Inspection.
"""

import os
import glob
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

FRUIT_CLASSES = [
    "freshapples",
    "freshbanana",
    "freshoranges",
    "rottenapples",
    "rottenbanana",
    "rottenoranges",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class FruitDataset(Dataset):
    def __init__(self, file_paths: List[str], labels: List[int], transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        p = self.file_paths[idx]
        lbl = self.labels[idx]
        with Image.open(p) as img:
            img = img.convert("RGB")
            if self.transform is not None:
                tensor = self.transform(img)
            else:
                tensor = transforms.ToTensor()(img)
        return tensor, lbl


def discover_split_files(split_dir: str) -> Tuple[List[str], List[int], Dict[str, int]]:
    paths, labels = [], []
    class_counts = {}
    for cls_idx, cls_name in enumerate(FRUIT_CLASSES):
        c_dir = os.path.join(split_dir, cls_name)
        if os.path.exists(c_dir):
            files = sorted(
                glob.glob(os.path.join(c_dir, "*.jpg"))
                + glob.glob(os.path.join(c_dir, "*.png"))
                + glob.glob(os.path.join(c_dir, "*.jpeg"))
            )
            class_counts[cls_name] = len(files)
            for f in files:
                paths.append(f)
                labels.append(cls_idx)
        else:
            class_counts[cls_name] = 0
    return paths, labels, class_counts


def get_fruit_dataloaders(
    dataset_root: str,
    batch_size: int = 32,
    img_size: int = 224,
    train_limit: Optional[int] = 2000,
    test_limit: Optional[int] = 1000,
    val_ratio: float = 0.15,
    seed: int = 42,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    # Search for train and test folders
    train_dir = os.path.join(dataset_root, "train")
    test_dir = os.path.join(dataset_root, "test")

    if not os.path.exists(train_dir):
        # Check nested dataset folder
        nested = os.path.join(dataset_root, "dataset")
        if os.path.exists(os.path.join(nested, "train")):
            train_dir = os.path.join(nested, "train")
            test_dir = os.path.join(nested, "test")
        else:
            raise FileNotFoundError(f"Could not locate train directory in {dataset_root}")

    train_paths, train_labels, train_counts = discover_split_files(train_dir)
    test_paths, test_labels, test_counts = discover_split_files(test_dir)

    # Subsample training data if limit provided (stratified per class)
    np.random.seed(seed)
    if train_limit is not None and train_limit < len(train_paths):
        per_class_limit = max(1, train_limit // len(FRUIT_CLASSES))
        sub_paths, sub_labels = [], []
        for c_idx in range(len(FRUIT_CLASSES)):
            c_indices = [i for i, l in enumerate(train_labels) if l == c_idx]
            chosen = np.random.choice(c_indices, min(per_class_limit, len(c_indices)), replace=False)
            for ch in chosen:
                sub_paths.append(train_paths[ch])
                sub_labels.append(train_labels[ch])
        train_paths, train_labels = sub_paths, sub_labels

    # Stratified split into Train and Validation
    train_final_paths, train_final_labels = [], []
    val_final_paths, val_final_labels = [], []

    for c_idx in range(len(FRUIT_CLASSES)):
        c_indices = [i for i, l in enumerate(train_labels) if l == c_idx]
        np.random.shuffle(c_indices)
        n_val = max(1, int(len(c_indices) * val_ratio))
        val_indices = c_indices[:n_val]
        tr_indices = c_indices[n_val:]

        for vi in val_indices:
            val_final_paths.append(train_paths[vi])
            val_final_labels.append(train_labels[vi])
        for ti in tr_indices:
            train_final_paths.append(train_paths[ti])
            train_final_labels.append(train_labels[ti])

    # Subsample test set if test_limit specified
    if test_limit is not None and test_limit < len(test_paths):
        per_class_test = max(1, test_limit // len(FRUIT_CLASSES))
        sub_t_paths, sub_t_labels = [], []
        for c_idx in range(len(FRUIT_CLASSES)):
            c_indices = [i for i, l in enumerate(test_labels) if l == c_idx]
            chosen = np.random.choice(c_indices, min(per_class_test, len(c_indices)), replace=False)
            for ch in chosen:
                sub_t_paths.append(test_paths[ch])
                sub_t_labels.append(test_labels[ch])
        test_paths, test_labels = sub_t_paths, sub_t_labels

    # Data Augmentation & Normalization
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_ds = FruitDataset(train_final_paths, train_final_labels, transform=train_transform)
    val_ds = FruitDataset(val_final_paths, val_final_labels, transform=eval_transform)
    test_ds = FruitDataset(test_paths, test_labels, transform=eval_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    stats = {
        "raw_train_counts": train_counts,
        "raw_test_counts": test_counts,
        "used_train_samples": len(train_final_paths),
        "used_val_samples": len(val_final_paths),
        "used_test_samples": len(test_paths),
    }

    return train_loader, val_loader, test_loader, stats


def plot_class_distribution(stats: Dict[str, Any], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    classes = FRUIT_CLASSES
    train_vals = [stats["raw_train_counts"].get(c, 0) for c in classes]
    test_vals = [stats["raw_test_counts"].get(c, 0) for c in classes]

    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, train_vals, width, label="Train Set", color="#2b5c8f")
    ax.bar(x + width / 2, test_vals, width, label="Test Set", color="#d95f02")

    ax.set_ylabel("Number of Images")
    ax.set_title("Fruit Quality Inspection - Class Distribution")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=25, ha="right", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[✓] Saved class distribution plot to: {output_path}")


def save_fruit_samples(dataset_root: str, output_path: str):
    """Saves a 2x3 sample visual of each fruit category."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()

    train_dir = os.path.join(dataset_root, "train")
    if not os.path.exists(train_dir):
        train_dir = os.path.join(dataset_root, "dataset", "train")

    for idx, cname in enumerate(FRUIT_CLASSES):
        c_dir = os.path.join(train_dir, cname)
        img_files = glob.glob(os.path.join(c_dir, "*.jpg")) + glob.glob(os.path.join(c_dir, "*.png"))
        if img_files:
            img = Image.open(img_files[0]).convert("RGB")
            axes[idx].imshow(img)
            # Quality tag
            quality_tag = "FRESH" if "fresh" in cname else "ROTTEN"
            tag_color = "green" if quality_tag == "FRESH" else "red"
            axes[idx].set_title(f"{cname}\n[{quality_tag}]", fontsize=11, fontweight="bold", color=tag_color)
        axes[idx].axis("off")

    plt.suptitle("Fruit Quality Inspection - Dataset Exploration", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[✓] Saved dataset sample visualization to: {output_path}")
