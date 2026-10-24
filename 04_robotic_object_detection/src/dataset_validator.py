"""
dataset_validator.py
Validates YOLO annotations, counts bounding box class distributions, and creates exploration visualisations.
"""

import os
import glob
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
import matplotlib.pyplot as plt

SAFETY_CLASSES = [
    "Hardhat",
    "Mask",
    "NO-Hardhat",
    "NO-Mask",
    "NO-Safety Vest",
    "Person",
    "Safety Cone",
    "Safety Vest",
    "machinery",
    "vehicle",
]

# Distinct colors for each class (BGR format)
CLASS_COLORS = [
    (0, 255, 0),    # Hardhat: Green
    (255, 255, 0),  # Mask: Cyan
    (0, 0, 255),    # NO-Hardhat: Red
    (0, 140, 255),  # NO-Mask: Orange
    (0, 69, 255),   # NO-Safety Vest: Red-Orange
    (255, 0, 255),  # Person: Magenta
    (0, 215, 255),  # Safety Cone: Yellow
    (50, 205, 50),  # Safety Vest: Lime Green
    (255, 144, 30), # machinery: Blue-Steel
    (180, 105, 255),# vehicle: Pink
]


def validate_yolo_dataset(data_dir: str) -> Dict[str, Any]:
    stats = {}
    class_counts = {c: 0 for c in SAFETY_CLASSES}
    total_boxes = 0
    invalid_lines = 0

    for split in ["train", "valid", "test"]:
        img_dir = os.path.join(data_dir, split, "images")
        lbl_dir = os.path.join(data_dir, split, "labels")

        imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png")))
        lbls = sorted(glob.glob(os.path.join(lbl_dir, "*.txt")))

        stats[f"{split}_images"] = len(imgs)
        stats[f"{split}_labels"] = len(lbls)

        for l_path in lbls:
            with open(l_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        try:
                            cid = int(parts[0])
                            xc, yc, bw, bh = [float(x) for x in parts[1:]]
                            if 0 <= cid < len(SAFETY_CLASSES):
                                class_counts[SAFETY_CLASSES[cid]] += 1
                                total_boxes += 1
                            else:
                                invalid_lines += 1
                        except ValueError:
                            invalid_lines += 1
                    else:
                        invalid_lines += 1

    stats["total_bounding_boxes"] = total_boxes
    stats["invalid_annotations"] = invalid_lines
    stats["class_counts"] = class_counts
    return stats


def plot_dataset_exploration(data_dir: str, output_path: str, num_samples: int = 4):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img_dir = os.path.join(data_dir, "train", "images")
    lbl_dir = os.path.join(data_dir, "train", "labels")

    img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png")))
    if not img_files:
        return

    # Select images that contain multiple annotations
    selected_pairs = []
    for ip in img_files:
        bp = os.path.splitext(os.path.basename(ip))[0]
        lp = os.path.join(lbl_dir, f"{bp}.txt")
        if os.path.exists(lp):
            with open(lp) as f:
                lines = f.readlines()
            if len(lines) >= 3:
                selected_pairs.append((ip, lp))
        if len(selected_pairs) >= num_samples:
            break

    fig, axes = plt.subplots(1, len(selected_pairs), figsize=(5 * len(selected_pairs), 5))
    if len(selected_pairs) == 1:
        axes = [axes]

    for ax_idx, (ip, lp) in enumerate(selected_pairs):
        img = cv2.imread(ip)
        h, w = img.shape[:2]

        with open(lp) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cid = int(parts[0])
                    xc, yc, bw, bh = [float(x) for x in parts[1:]]
                    x1 = int((xc - bw / 2) * w)
                    y1 = int((yc - bh / 2) * h)
                    x2 = int((xc + bw / 2) * w)
                    y2 = int((yc + bh / 2) * h)

                    color = CLASS_COLORS[cid % len(CLASS_COLORS)]
                    cname = SAFETY_CLASSES[cid]
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        img,
                        cname,
                        (x1, max(15, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        color,
                        1,
                        cv2.LINE_AA,
                    )

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[ax_idx].imshow(img_rgb)
        axes[ax_idx].set_title(f"Sample {ax_idx+1}: {os.path.basename(ip)[:15]}...", fontsize=9)
        axes[ax_idx].axis("off")

    plt.suptitle("Construction Site Safety - Ground Truth Bounding Box Visualisation", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[✓] Saved dataset exploration visualisation to: {output_path}")
