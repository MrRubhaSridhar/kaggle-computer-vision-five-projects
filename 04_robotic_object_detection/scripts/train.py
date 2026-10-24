#!/usr/bin/env python3
"""
04_robotic_object_detection/scripts/train.py
Configures and trains YOLOv8n on the Construction Site Safety Roboflow dataset.
"""

import os
import sys
import shutil
import argparse
import yaml
import torch
from ultralytics import YOLO

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataset_validator import (
    validate_yolo_dataset,
    plot_dataset_exploration,
    SAFETY_CLASSES,
)


def prepare_data_yaml(raw_dir: str, output_yaml_path: str) -> str:
    css_dir = os.path.join(raw_dir, "css-data")
    if not os.path.exists(css_dir):
        css_dir = raw_dir

    abs_css_dir = os.path.abspath(css_dir)

    data_config = {
        "path": abs_css_dir,
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": {i: name for i, name in enumerate(SAFETY_CLASSES)},
    }

    os.makedirs(os.path.dirname(output_yaml_path), exist_ok=True)
    with open(output_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_config, f, sort_keys=False)

    print(f"[✓] Generated data configuration at: {output_yaml_path}")
    return output_yaml_path


def train_yolo_model(
    raw_dir: str,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 16,
    img_size: int = 640,
    device_name: str = "auto",
):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Dataset validation
    css_dir = os.path.join(raw_dir, "css-data")
    if not os.path.exists(css_dir):
        css_dir = raw_dir
    print("[+] Validating YOLO dataset annotations and generating exploration visualisations...")
    stats = validate_yolo_dataset(css_dir)
    print(f"    - Train images: {stats['train_images']}, Valid images: {stats['valid_images']}, Test images: {stats['test_images']}")
    print(f"    - Total Bounding Boxes: {stats['total_bounding_boxes']}, Invalid Annotations: {stats['invalid_annotations']}")

    plot_dataset_exploration(css_dir, os.path.join(output_dir, "dataset_exploration.png"))

    # 2. Data YAML
    yaml_path = prepare_data_yaml(raw_dir, os.path.join(output_dir, "data.yaml"))

    # 3. Determine device
    if device_name == "auto":
        device = 0 if torch.cuda.is_available() else "cpu"
    else:
        device = device_name
    print(f"[+] Training on device: {device}")

    # 4. Initialize YOLOv8n
    model = YOLO("yolov8n.pt")

    # 5. Execute Training
    runs_dir = os.path.join(output_dir, "runs")
    print(f"\n--- Starting YOLOv8n Training for {epochs} Epochs ---")
    train_results = model.train(
        data=yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=runs_dir,
        name="train",
        exist_ok=True,
        workers=2,
        verbose=True,
    )

    # Copy best weights and summary figures
    best_weights_src = os.path.join(runs_dir, "train", "weights", "best.pt")
    if os.path.exists(best_weights_src):
        best_dest = os.path.join(output_dir, "best.pt")
        shutil.copy2(best_weights_src, best_dest)
        print(f"[✓] Best model weights saved to: {best_dest}")

    # Copy plots if present
    for fig_name in ["results.png", "confusion_matrix.png", "F1_curve.png"]:
        src_fig = os.path.join(runs_dir, "train", fig_name)
        if os.path.exists(src_fig):
            shutil.copy2(src_fig, os.path.join(output_dir, fig_name))

    print(f"[✓] YOLOv8n training completed successfully.")


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8n for Robotic Object Detection")
    parser.add_argument(
        "--raw_dir",
        type=str,
        default="04_robotic_object_detection/data/raw",
        help="Path to raw dataset directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="04_robotic_object_detection/results",
        help="Directory to save training outputs and weights",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--img_size", type=int, default=640, help="Image resolution")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu, 0, auto)")
    args = parser.parse_args()

    train_yolo_model(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
