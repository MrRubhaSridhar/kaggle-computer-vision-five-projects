#!/usr/bin/env python3
"""
03_traffic_sign_recognition/scripts/train.py
Trains TrafficSignCNN on the German Traffic Sign Recognition Benchmark (GTSRB).
"""

import os
import sys
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataset import (
    get_gtsrb_dataloaders,
    plot_gtsrb_class_distribution,
    save_gtsrb_samples,
)
from src.model import TrafficSignCNN


def train_traffic_sign_model(
    raw_dir: str,
    output_dir: str,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
    train_limit: int = 10000,
    test_limit: int = 2000,
    device_name: str = "auto",
):
    os.makedirs(output_dir, exist_ok=True)

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    print(f"[+] Using device: {device}")

    print("[+] Plotting class distribution and generating dataset exploration samples...")
    plot_gtsrb_class_distribution(raw_dir, os.path.join(output_dir, "class_distribution.png"))
    save_gtsrb_samples(raw_dir, os.path.join(output_dir, "dataset_samples.png"))

    train_loader, val_loader, test_loader, stats = get_gtsrb_dataloaders(
        raw_dir,
        batch_size=batch_size,
        train_limit=train_limit,
        test_limit=test_limit,
    )
    print(f"[+] Loaded splits: {stats['used_train_samples']} train, {stats['used_val_samples']} val, {stats['used_test_samples']} test")

    model = TrafficSignCNN(num_classes=43).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_ckpt_path = os.path.join(output_dir, "best_model.pth")

    print(f"\n--- Training Traffic Sign Classifier for {epochs} Epochs ---")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for images, targets in pbar:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct/total:.3f}"})

        scheduler.step()
        epoch_tr_loss = running_loss / total
        epoch_tr_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total

        history["train_loss"].append(epoch_tr_loss)
        history["train_acc"].append(epoch_tr_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} -> "
            f"Train Loss: {epoch_tr_loss:.4f} | Train Acc: {epoch_tr_acc*100:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.2f}%"
        )

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": epoch_val_acc,
                    "history": history,
                    "stats": stats,
                },
                best_ckpt_path,
            )
            print(f"    [*] Checkpoint saved (New Best Val Acc: {best_val_acc*100:.2f}%)")

    total_time = time.time() - start_time
    print(f"\n[✓] Training completed in {total_time:.1f}s. Best Val Acc: {best_val_acc*100:.2f}%")

    # Plot Curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(range(1, epochs + 1), history["train_loss"], label="Train Loss", marker="o")
    axes[0].plot(range(1, epochs + 1), history["val_loss"], label="Val Loss", marker="s")
    axes[0].set_title("Loss vs Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross Entropy Loss")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()

    axes[1].plot(range(1, epochs + 1), [a * 100 for a in history["train_acc"]], label="Train Acc", marker="o")
    axes[1].plot(range(1, epochs + 1), [a * 100 for a in history["val_acc"]], label="Val Acc", marker="s")
    axes[1].set_title("Accuracy vs Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    curve_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(curve_path, dpi=150)
    plt.close()
    print(f"[✓] Saved training curves to: {curve_path}")


def main():
    parser = argparse.ArgumentParser(description="Train GTSRB Traffic Sign Classifier")
    parser.add_argument(
        "--raw_dir",
        type=str,
        default="03_traffic_sign_recognition/data/raw",
        help="Path to GTSRB raw directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="03_traffic_sign_recognition/results",
        help="Directory for artifacts",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--train_limit", type=int, default=10000, help="Max train samples")
    parser.add_argument("--test_limit", type=int, default=2000, help="Max test samples")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu, cuda, auto)")
    args = parser.parse_args()

    train_traffic_sign_model(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        train_limit=args.train_limit,
        test_limit=args.test_limit,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
