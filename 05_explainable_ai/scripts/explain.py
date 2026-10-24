#!/usr/bin/env python3
"""
05_explainable_ai/scripts/explain.py
Generates Grad-CAM visualisations for the CIFAR-10 model:
- Produces Original, Heatmap, and Overlay triplets
- Extracts both Correct and Misclassified examples to explain model reasoning
- Demonstrates single-image inference and visualization
"""

import os
import sys
import argparse
from typing import List, Dict

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model import CIFAR10CNN
from src.dataset import get_cifar10_data, CIFAR10_CLASSES, CIFAR_MEAN, CIFAR_STD
from src.gradcam import GradCAM, create_gradcam_overlay


def run_explainability(
    raw_dir: str,
    output_dir: str,
    checkpoint_path: str,
    device_name: str = "auto",
    num_samples_per_category: int = 4,
):
    gradcam_out_dir = os.path.join(output_dir, "gradcam")
    os.makedirs(gradcam_out_dir, exist_ok=True)

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    print(f"[+] Running Explainability on device: {device}")

    # Load model
    model = CIFAR10CNN(num_classes=10).to(device)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[✓] Model checkpoint loaded from: {checkpoint_path}")

    target_layer = model.get_target_layer()
    gradcam = GradCAM(model, target_layer)

    # Load test data
    _, _, _, _, test_imgs, test_lbls = get_cifar10_data(raw_dir, test_limit=1000)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    correct_examples: List[Dict] = []
    incorrect_examples: List[Dict] = []

    print("[+] Collecting correct and misclassified examples...")
    for idx in range(len(test_imgs)):
        img_np = test_imgs[idx]  # (32, 32, 3) uint8
        true_label = test_lbls[idx]

        pil_img = Image.fromarray(img_np)
        input_tensor = transform(pil_img).unsqueeze(0).to(device)

        heatmap, pred_class, pred_prob = gradcam.generate(input_tensor, target_class=None)
        heatmap_rgb, overlay_rgb = create_gradcam_overlay(img_np, heatmap)

        item = {
            "index": idx,
            "orig_rgb": img_np,
            "heatmap": heatmap,
            "heatmap_rgb": heatmap_rgb,
            "overlay_rgb": overlay_rgb,
            "true_class": true_label,
            "true_name": CIFAR10_CLASSES[true_label],
            "pred_class": pred_class,
            "pred_name": CIFAR10_CLASSES[pred_class],
            "pred_prob": pred_prob,
        }

        if pred_class == true_label and len(correct_examples) < num_samples_per_category:
            correct_examples.append(item)
            # Save individual example
            out_fname = f"correct_{len(correct_examples)}_{item['true_name']}.png"
            _save_triplet(item, os.path.join(gradcam_out_dir, out_fname))
        elif pred_class != true_label and len(incorrect_examples) < num_samples_per_category:
            incorrect_examples.append(item)
            # Save individual example
            out_fname = f"incorrect_{len(incorrect_examples)}_true_{item['true_name']}_pred_{item['pred_name']}.png"
            _save_triplet(item, os.path.join(gradcam_out_dir, out_fname))

        if len(correct_examples) >= num_samples_per_category and len(incorrect_examples) >= num_samples_per_category:
            break

    gradcam.remove_hooks()

    # Generate master comparison grid
    grid_path = os.path.join(gradcam_out_dir, "gradcam_comparison_grid.png")
    _save_comparison_grid(correct_examples, incorrect_examples, grid_path)
    print(f"[✓] Saved comprehensive Grad-CAM grid to: {grid_path}")


def _save_triplet(item: Dict, path: str):
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].imshow(item["orig_rgb"])
    axes[0].set_title(f"True: {item['true_name']}", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(item["heatmap_rgb"])
    axes[1].set_title("Grad-CAM Heatmap", fontsize=10)
    axes[1].axis("off")

    axes[2].imshow(item["overlay_rgb"])
    axes[2].set_title(f"Pred: {item['pred_name']} ({item['pred_prob']*100:.1f}%)", fontsize=10, fontweight="bold")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _save_comparison_grid(correct_list: List[Dict], incorrect_list: List[Dict], path: str):
    n_rows = len(correct_list) + len(incorrect_list)
    fig, axes = plt.subplots(n_rows, 3, figsize=(10, 2.8 * n_rows))

    row = 0
    # Correct section
    for item in correct_list:
        axes[row, 0].imshow(item["orig_rgb"])
        axes[row, 0].set_title(f"[CORRECT] True: {item['true_name']}", color="green", fontweight="bold")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(item["heatmap_rgb"])
        axes[row, 1].set_title("Grad-CAM Heatmap")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(item["overlay_rgb"])
        axes[row, 2].set_title(f"Pred: {item['pred_name']} ({item['pred_prob']*100:.1f}%)", color="green", fontweight="bold")
        axes[row, 2].axis("off")
        row += 1

    # Incorrect section
    for item in incorrect_list:
        axes[row, 0].imshow(item["orig_rgb"])
        axes[row, 0].set_title(f"[MISCLASSIFIED] True: {item['true_name']}", color="red", fontweight="bold")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(item["heatmap_rgb"])
        axes[row, 1].set_title("Grad-CAM Heatmap")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(item["overlay_rgb"])
        axes[row, 2].set_title(f"Pred: {item['pred_name']} ({item['pred_prob']*100:.1f}%)", color="red", fontweight="bold")
        axes[row, 2].axis("off")
        row += 1

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM Visualizations for CIFAR-10")
    parser.add_argument("--raw_dir", type=str, default="05_explainable_ai/data/raw", help="Path to raw CIFAR-10 data")
    parser.add_argument("--output_dir", type=str, default="05_explainable_ai/results", help="Directory for artifacts")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="05_explainable_ai/results/best_model.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu, cuda, auto)")
    args = parser.parse_args()

    run_explainability(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
