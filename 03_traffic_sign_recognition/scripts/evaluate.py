#!/usr/bin/env python3
"""
03_traffic_sign_recognition/scripts/evaluate.py
Evaluates the TrafficSignCNN model on the GTSRB test split.
Computes Accuracy, Macro Precision, Macro Recall, Macro F1-Score, and Confusion Matrix.
Saves genuine prediction visualisations with ground truth and predicted labels.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any

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
from src.dataset import get_gtsrb_dataloaders, GTSRB_CLASSES
from src.model import TrafficSignCNN


def evaluate_traffic_signs(
    raw_dir: str,
    output_dir: str,
    checkpoint_path: str,
    test_limit: int = 2000,
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

    _, _, test_loader, stats = get_gtsrb_dataloaders(raw_dir, batch_size=64, test_limit=test_limit)

    model = TrafficSignCNN(num_classes=43).to(device)
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        print(f"[✓] Loaded model checkpoint from: {checkpoint_path}")
    else:
        print(f"[!] Warning: Checkpoint not found at {checkpoint_path}")

    model.eval()
    all_preds, all_targets, all_probs = [], [], []

    print(f"[+] Evaluating on test set ({len(test_loader.dataset)} samples)...")
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Testing"):
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
    for idx in range(43):
        class_metrics[str(idx)] = {
            "name": GTSRB_CLASSES.get(idx, "Unknown"),
            "precision": float(round(c_prec[idx], 4)) if idx < len(c_prec) else 0.0,
            "recall": float(round(c_rec[idx], 4)) if idx < len(c_rec) else 0.0,
            "f1_score": float(round(c_f1[idx], 4)) if idx < len(c_f1) else 0.0,
            "support": int(c_supp[idx]) if idx < len(c_supp) else 0,
        }

    cm = confusion_matrix(all_targets, all_preds, labels=list(range(43)))

    results = {
        "dataset": "meowmeowmeowmeowmeow/gtsrb-german-traffic-sign",
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
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, cmap="Blues", cbar=True, annot=False)
    plt.title("GTSRB 43-Class Test Confusion Matrix", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Class ID", fontsize=12)
    plt.ylabel("True Class ID", fontsize=12)
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"[✓] Saved confusion matrix plot to: {cm_path}")

    # Visualizations of predictions
    test_df = test_loader.dataset.df
    correct_indices = np.where(all_preds == all_targets)[0]
    incorrect_indices = np.where(all_preds != all_targets)[0]

    _save_gtsrb_pred_grid(
        raw_dir,
        test_df,
        all_targets,
        all_preds,
        all_probs,
        correct_indices[:8],
        os.path.join(pred_dir, "sample_predictions.png"),
        title="Correctly Recognized Traffic Signs",
        is_correct=True,
    )

    if len(incorrect_indices) > 0:
        _save_gtsrb_pred_grid(
            raw_dir,
            test_df,
            all_targets,
            all_preds,
            all_probs,
            incorrect_indices[:8],
            os.path.join(pred_dir, "misclassified_signs.png"),
            title="Misclassified Traffic Signs Analysis",
            is_correct=False,
        )

    return results


def _save_gtsrb_pred_grid(raw_dir, df, targets, preds, probs, indices, out_path, title, is_correct):
    if len(indices) == 0:
        return
    n = len(indices)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4.5 * rows))
    axes = np.array(axes).flatten()

    for i, idx in enumerate(indices):
        ax = axes[i]
        row = df.iloc[idx]
        img_p = os.path.join(raw_dir, row["Path"])
        x1, y1 = int(row["Roi.X1"]), int(row["Roi.Y1"])
        x2, y2 = int(row["Roi.X2"]), int(row["Roi.Y2"])
        img = Image.open(img_p).convert("RGB")
        if x2 > x1 and y2 > y1:
            img = img.crop((x1, y1, x2, y2))

        ax.imshow(img)
        t_name = GTSRB_CLASSES.get(targets[idx], "Unknown")
        p_name = GTSRB_CLASSES.get(preds[idx], "Unknown")
        conf = probs[idx][preds[idx]] * 100.0

        if is_correct:
            color = "green"
            lbl = f"True: {t_name}\nPred: {p_name}\nConf: {conf:.1f}%"
        else:
            color = "red"
            lbl = f"True: {t_name}\nPred: {p_name}\nConf: {conf:.1f}%"

        ax.set_title(lbl, fontsize=9, fontweight="bold", color=color)
        ax.axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[✓] Saved prediction visual: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Traffic Sign Classifier")
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
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="03_traffic_sign_recognition/results/best_model.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument("--test_limit", type=int, default=2000, help="Max test samples")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu, cuda, auto)")
    args = parser.parse_args()

    results = evaluate_traffic_signs(
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
