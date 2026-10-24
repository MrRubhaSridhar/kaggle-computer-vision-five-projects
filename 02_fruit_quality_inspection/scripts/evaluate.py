#!/usr/bin/env python3
"""
02_fruit_quality_inspection/scripts/evaluate.py
Evaluates the Fruit Quality MobileNetV2 classifier on the test split.
Generates metrics JSON, confusion matrix plot, and sample prediction grids (correct & misclassified).
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, List

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataset import get_fruit_dataloaders, FRUIT_CLASSES
from src.model import FruitMobileNetV2


def evaluate_fruit_model(
    dataset_root: str,
    output_dir: str,
    checkpoint_path: str,
    test_limit: int = 1000,
    device_name: str = "auto",
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    pred_dir = os.path.join(output_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    print(f"[+] Evaluating on device: {device}")

    _, _, test_loader, stats = get_fruit_dataloaders(
        dataset_root, batch_size=32, test_limit=test_limit
    )

    model = FruitMobileNetV2(num_classes=len(FRUIT_CLASSES), pretrained=False).to(device)
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        print(f"[✓] Loaded model checkpoint from: {checkpoint_path}")
    else:
        print(f"[!] Checkpoint not found at {checkpoint_path}")

    model.eval()
    all_preds, all_targets, all_probs = [], [], []

    print(f"[+] Running evaluation on test set ({len(test_loader.dataset)} samples)...")
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_preds.extend(preds)
            all_targets.extend(targets.numpy())
            all_probs.extend(probs)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    # Metrics
    acc = accuracy_score(all_targets, all_preds)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="macro", zero_division=0
    )

    c_prec, c_rec, c_f1, c_supp = precision_recall_fscore_support(
        all_targets, all_preds, average=None, zero_division=0
    )

    class_metrics = {}
    for idx, cname in enumerate(FRUIT_CLASSES):
        class_metrics[cname] = {
            "precision": float(round(c_prec[idx], 4)),
            "recall": float(round(c_rec[idx], 4)),
            "f1_score": float(round(c_f1[idx], 4)),
            "support": int(c_supp[idx]),
        }

    cm = confusion_matrix(all_targets, all_preds)

    results = {
        "dataset": "sriramr/fruits-fresh-and-rotten-for-classification",
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
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=FRUIT_CLASSES,
        yticklabels=FRUIT_CLASSES,
    )
    plt.title("Fruit Quality Inspection - Confusion Matrix", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"[✓] Saved confusion matrix plot to: {cm_path}")

    # Save visual predictions (correct vs misclassified grids)
    test_file_paths = test_loader.dataset.file_paths
    correct_indices = np.where(all_preds == all_targets)[0]
    incorrect_indices = np.where(all_preds != all_targets)[0]

    _save_prediction_grid(
        test_file_paths,
        all_targets,
        all_preds,
        all_probs,
        correct_indices[:6],
        os.path.join(pred_dir, "correct_samples.png"),
        title="Correctly Classified Fruit Samples",
        is_correct=True,
    )

    if len(incorrect_indices) > 0:
        _save_prediction_grid(
            test_file_paths,
            all_targets,
            all_preds,
            all_probs,
            incorrect_indices[:6],
            os.path.join(pred_dir, "misclassified_samples.png"),
            title="Misclassified Fruit Samples",
            is_correct=False,
        )

    return results


def _save_prediction_grid(paths, targets, preds, probs, indices, out_path, title, is_correct):
    if len(indices) == 0:
        return
    n = len(indices)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()

    for i, idx in enumerate(indices):
        ax = axes[i]
        img = Image.open(paths[idx]).convert("RGB")
        ax.imshow(img)
        true_name = FRUIT_CLASSES[targets[idx]]
        pred_name = FRUIT_CLASSES[preds[idx]]
        conf = probs[idx][preds[idx]] * 100.0

        if is_correct:
            color = "green"
            ax.set_title(f"True: {true_name}\nPred: {pred_name} ({conf:.1f}%)", color=color, fontweight="bold", fontsize=10)
        else:
            color = "red"
            ax.set_title(f"True: {true_name}\nPred: {pred_name} ({conf:.1f}%)", color=color, fontweight="bold", fontsize=10)
        ax.axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[✓] Saved prediction grid: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Fruit Quality Inspection Model")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="02_fruit_quality_inspection/data/raw/dataset",
        help="Path to dataset root",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="02_fruit_quality_inspection/results",
        help="Directory for artifacts",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="02_fruit_quality_inspection/results/best_model.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument("--test_limit", type=int, default=1000, help="Max test samples")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu, cuda, auto)")
    args = parser.parse_args()

    results = evaluate_fruit_model(
        dataset_root=args.dataset_root,
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
