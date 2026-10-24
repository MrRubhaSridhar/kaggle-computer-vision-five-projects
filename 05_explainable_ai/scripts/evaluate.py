#!/usr/bin/env python3
"""
05_explainable_ai/scripts/evaluate.py
Evaluates the trained CIFAR-10 model on the test split.
Computes Accuracy, Macro Precision, Macro Recall, Macro F1-Score, and Confusion Matrix.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score
from tqdm import tqdm

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model import CIFAR10CNN
from src.dataset import get_dataloaders, CIFAR10_CLASSES


def evaluate_test_set(
    raw_dir: str,
    output_dir: str,
    checkpoint_path: str,
    test_limit: int = 2000,
    device_name: str = "auto",
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    print(f"[+] Evaluating on device: {device}")

    # Load dataloaders
    _, _, test_loader = get_dataloaders(raw_dir, batch_size=64, test_limit=test_limit)

    model = CIFAR10CNN(num_classes=10).to(device)
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        print(f"[✓] Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"[!] Warning: Checkpoint {checkpoint_path} not found. Using untrained weights.")

    model.eval()
    all_preds = []
    all_targets = []

    print(f"[+] Running evaluation on test set ({len(test_loader.dataset)} samples)...")
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Calculate metrics
    acc = accuracy_score(all_targets, all_preds)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="macro", zero_division=0
    )

    class_prec, class_rec, class_f1, class_supp = precision_recall_fscore_support(
        all_targets, all_preds, average=None, zero_division=0
    )

    class_metrics = {}
    for idx, cname in enumerate(CIFAR10_CLASSES):
        class_metrics[cname] = {
            "precision": float(round(class_prec[idx], 4)),
            "recall": float(round(class_rec[idx], 4)),
            "f1_score": float(round(class_f1[idx], 4)),
            "support": int(class_supp[idx]),
        }

    cm = confusion_matrix(all_targets, all_preds)

    results = {
        "dataset": "ekaakurniawan/the-cifar10-dataset",
        "split": "test",
        "total_test_samples": len(all_targets),
        "overall_metrics": {
            "accuracy": float(round(acc, 4)),
            "macro_precision": float(round(prec_macro, 4)),
            "macro_recall": float(round(rec_macro, 4)),
            "macro_f1_score": float(round(f1_macro, 4)),
        },
        "class_wise_metrics": class_metrics,
        "confusion_matrix": cm.tolist(),
    }

    # Save metrics JSON
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[✓] Evaluation metrics saved to: {metrics_path}")

    # Plot Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CIFAR10_CLASSES,
        yticklabels=CIFAR10_CLASSES,
    )
    plt.title("CIFAR-10 Test Confusion Matrix", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"[✓] Confusion matrix plot saved to: {cm_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate CIFAR-10 CNN")
    parser.add_argument("--raw_dir", type=str, default="05_explainable_ai/data/raw", help="Path to raw CIFAR-10 data")
    parser.add_argument("--output_dir", type=str, default="05_explainable_ai/results", help="Directory for artifacts")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="05_explainable_ai/results/best_model.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument("--test_limit", type=int, default=2000, help="Max test images for evaluation")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu, cuda, auto)")
    args = parser.parse_args()

    results = evaluate_test_set(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint,
        test_limit=args.test_limit,
        device_name=args.device,
    )

    print("\n--- Evaluation Summary ---")
    print(f"Accuracy:        {results['overall_metrics']['accuracy']*100:.2f}%")
    print(f"Macro Precision: {results['overall_metrics']['macro_precision']:.4f}")
    print(f"Macro Recall:    {results['overall_metrics']['macro_recall']:.4f}")
    print(f"Macro F1-Score:  {results['overall_metrics']['macro_f1_score']:.4f}")


if __name__ == "__main__":
    main()
