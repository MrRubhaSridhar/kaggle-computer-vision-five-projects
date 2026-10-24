"""
dataset.py
Loads the raw CIFAR-10 python batches, provides PyTorch Datasets, transforms, and data exploration utilities.
"""

import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# Standard CIFAR-10 Normalization
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def load_cifar_batch(file_path: str) -> Tuple[np.ndarray, List[int]]:
    with open(file_path, "rb") as f:
        entry = pickle.load(f, encoding="bytes")
    data = entry[b"data"]
    labels = entry[b"labels"]
    # Reshape from (N, 3072) to (N, 32, 32, 3)
    data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return data, labels


class CIFAR10Dataset(Dataset):
    def __init__(self, images: np.ndarray, labels: List[int], transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = self.images[idx]
        label = self.labels[idx]
        pil_img = Image.fromarray(img)

        if self.transform is not None:
            tensor_img = self.transform(pil_img)
        else:
            tensor_img = transforms.ToTensor()(pil_img)

        return tensor_img, label


def get_cifar10_data(
    raw_dir: str,
    train_limit: Optional[int] = 10000,
    test_limit: Optional[int] = 2000,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, List[int], np.ndarray, List[int], np.ndarray, List[int]]:
    """
    Loads raw CIFAR-10 python batch files and splits into train, val, and test arrays.
    """
    batch_dir = os.path.join(raw_dir, "cifar-10-batches-py")
    if not os.path.exists(batch_dir):
        # Look if files are directly in raw_dir
        if os.path.exists(os.path.join(raw_dir, "data_batch_1")):
            batch_dir = raw_dir
        else:
            raise FileNotFoundError(f"CIFAR-10 batches not found at {batch_dir}")

    train_images_list, train_labels_list = [], []
    for i in range(1, 6):
        batch_file = os.path.join(batch_dir, f"data_batch_{i}")
        if os.path.exists(batch_file):
            d, l = load_cifar_batch(batch_file)
            train_images_list.append(d)
            train_labels_list.extend(l)

    all_train_images = np.concatenate(train_images_list, axis=0)
    all_train_labels = np.array(train_labels_list)

    # Subsample if train_limit specified
    np.random.seed(seed)
    if train_limit is not None and train_limit < len(all_train_labels):
        indices = np.random.choice(len(all_train_labels), train_limit, replace=False)
        all_train_images = all_train_images[indices]
        all_train_labels = all_train_labels[indices]

    # Validation split
    n_total = len(all_train_labels)
    n_val = int(n_total * val_ratio)
    val_idx = np.random.choice(n_total, n_val, replace=False)
    train_mask = np.ones(n_total, dtype=bool)
    train_mask[val_idx] = False

    train_imgs = all_train_images[train_mask]
    train_lbls = all_train_labels[train_mask].tolist()
    val_imgs = all_train_images[val_idx]
    val_lbls = all_train_labels[val_idx].tolist()

    # Test batch
    test_file = os.path.join(batch_dir, "test_batch")
    test_imgs, test_lbls = load_cifar_batch(test_file)
    test_lbls = np.array(test_lbls)

    if test_limit is not None and test_limit < len(test_lbls):
        t_indices = np.random.choice(len(test_lbls), test_limit, replace=False)
        test_imgs = test_imgs[t_indices]
        test_lbls = test_lbls[t_indices].tolist()
    else:
        test_lbls = test_lbls.tolist()

    return train_imgs, train_lbls, val_imgs, val_lbls, test_imgs, test_lbls


def get_dataloaders(
    raw_dir: str,
    batch_size: int = 64,
    train_limit: Optional[int] = 10000,
    test_limit: Optional[int] = 2000,
    val_ratio: float = 0.1,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_imgs, train_lbls, val_imgs, val_lbls, test_imgs, test_lbls = get_cifar10_data(
        raw_dir, train_limit, test_limit, val_ratio
    )

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    train_dataset = CIFAR10Dataset(train_imgs, train_lbls, transform=train_transform)
    val_dataset = CIFAR10Dataset(val_imgs, val_lbls, transform=eval_transform)
    test_dataset = CIFAR10Dataset(test_imgs, test_lbls, transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


def save_dataset_samples(images: np.ndarray, labels: List[int], output_path: str):
    """Saves a 2x5 grid of dataset samples representing all 10 CIFAR-10 classes."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()

    found_classes = {}
    for img, lbl in zip(images, labels):
        if lbl not in found_classes:
            found_classes[lbl] = img
        if len(found_classes) == 10:
            break

    for cls_idx in range(10):
        ax = axes[cls_idx]
        if cls_idx in found_classes:
            ax.imshow(found_classes[cls_idx])
            ax.set_title(CIFAR10_CLASSES[cls_idx], fontsize=11, fontweight="bold")
        ax.axis("off")

    plt.suptitle("CIFAR-10 Dataset Class Exploration", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[✓] Saved dataset sample visualization to: {output_path}")
