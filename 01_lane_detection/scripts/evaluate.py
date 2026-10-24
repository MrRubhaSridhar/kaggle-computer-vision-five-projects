#!/usr/bin/env python3
"""
01_lane_detection/scripts/evaluate.py
Evaluates the Lane Detection pipeline against the test set ground truth masks.
Calculates Precision, Recall, F1-Score, and Intersection-over-Union (IoU).
Saves quantitative metrics to JSON and qualitative visualizations to disk.
"""

import os
import sys
import glob
import json
import argparse
from typing import Dict, Any, List

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.lane_detector import LaneDetector
from src.mask_converter import yolo_seg_to_mask


def evaluate_dataset(
    dataset_dir: str,
    output_dir: str,
    max_samples: int = 100,
    save_visuals: int = 10,
) -> Dict[str, Any]:
    test_img_dir = os.path.join(dataset_dir, "test", "images")
    test_lbl_dir = os.path.join(dataset_dir, "test", "labels")

    if not os.path.exists(test_img_dir):
        raise FileNotFoundError(f"Test image directory not found: {test_img_dir}")

    img_files = sorted(glob.glob(os.path.join(test_img_dir, "*.jpg")) + glob.glob(os.path.join(test_img_dir, "*.png")))
    if not img_files:
        raise RuntimeError(f"No test images found in {test_img_dir}")

    img_files = img_files[:max_samples]
    print(f"[+] Evaluating Lane Detector on {len(img_files)} test images...")

    detector = LaneDetector()
    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, "overlays")
    inter_dir = os.path.join(output_dir, "intermediate")
    plots_dir = os.path.join(output_dir, "summary_plots")
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(inter_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    per_image_ious: List[float] = []
    per_image_f1s: List[float] = []
    per_image_precisions: List[float] = []
    per_image_recalls: List[float] = []

    departure_counts = {"IN_LANE": 0, "DEPARTURE_LEFT": 0, "DEPARTURE_RIGHT": 0, "NO_LANES_DETECTED": 0}

    saved_count = 0
    sample_visuals = []

    for idx, img_path in enumerate(tqdm(img_files, desc="Evaluating")):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(test_lbl_dir, f"{base_name}.txt")

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue

        h, w = img_bgr.shape[:2]
        # Ground truth binary mask
        gt_mask = yolo_seg_to_mask(lbl_path, h, w)

        # Run pipeline
        res = detector.process_frame(img_bgr, return_intermediates=(saved_count < save_visuals))
        pred_mask = res["pred_mask"]
        overlay = res["overlay"]
        dep_info = res["departure_info"]
        departure_counts[dep_info["status"]] = departure_counts.get(dep_info["status"], 0) + 1

        # Pixel-level metrics on lane region
        tp = int(np.sum((pred_mask > 0) & (gt_mask > 0)))
        fp = int(np.sum((pred_mask > 0) & (gt_mask == 0)))
        fn = int(np.sum((pred_mask == 0) & (gt_mask > 0)))
        tn = int(np.sum((pred_mask == 0) & (gt_mask == 0)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

        per_image_ious.append(iou)
        per_image_f1s.append(f1)
        per_image_precisions.append(precision)
        per_image_recalls.append(recall)

        if saved_count < save_visuals:
            cv2.imwrite(os.path.join(vis_dir, f"overlay_{base_name}.jpg"), overlay)
            if "intermediates" in res:
                for k, v in res["intermediates"].items():
                    cv2.imwrite(os.path.join(inter_dir, f"{base_name}_{k}.jpg"), v)

            sample_visuals.append({
                "orig": cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                "edges": res["intermediates"]["edges"] if "intermediates" in res else None,
                "gt": gt_mask,
                "pred": pred_mask,
                "overlay": cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                "iou": iou,
                "f1": f1,
                "status": dep_info["status"],
                "name": base_name,
            })
            saved_count += 1

    # Overall dataset-level micro metrics
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = (
        (2 * overall_precision * overall_recall) / (overall_precision + overall_recall)
        if (overall_precision + overall_recall) > 0
        else 0.0
    )
    overall_iou = (
        total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0.0
    )

    metrics = {
        "dataset": "princekhunt19/road-lane-segmentation-imgs-and-labels",
        "split": "test",
        "total_evaluated_samples": len(per_image_ious),
        "dataset_level_micro_metrics": {
            "iou": float(round(overall_iou, 4)),
            "precision": float(round(overall_precision, 4)),
            "recall": float(round(overall_recall, 4)),
            "f1_score": float(round(overall_f1, 4)),
            "tp_pixels": total_tp,
            "fp_pixels": total_fp,
            "fn_pixels": total_fn,
            "tn_pixels": total_tn,
        },
        "per_image_macro_averages": {
            "mean_iou": float(round(float(np.mean(per_image_ious)), 4)),
            "mean_precision": float(round(float(np.mean(per_image_precisions)), 4)),
            "mean_recall": float(round(float(np.mean(per_image_recalls)), 4)),
            "mean_f1_score": float(round(float(np.mean(per_image_f1s)), 4)),
        },
        "departure_warning_distribution": departure_counts,
    }

    # Save metrics JSON
    metrics_path = os.path.join(output_dir, "evaluation_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[✓] Metrics saved to {metrics_path}")

    # Generate visual figure comparison for report
    if sample_visuals:
        fig, axes = plt.subplots(len(sample_visuals[:4]), 4, figsize=(16, 4 * len(sample_visuals[:4])))
        if len(sample_visuals[:4]) == 1:
            axes = np.expand_dims(axes, 0)
        for r, item in enumerate(sample_visuals[:4]):
            axes[r, 0].imshow(item["orig"])
            axes[r, 0].set_title(f"Input: {item['name']}")
            axes[r, 0].axis("off")

            if item["edges"] is not None:
                axes[r, 1].imshow(item["edges"], cmap="gray")
            axes[r, 1].set_title("Canny Edges")
            axes[r, 1].axis("off")

            axes[r, 2].imshow(item["gt"], cmap="Blues")
            axes[r, 2].set_title("Ground Truth Mask")
            axes[r, 2].axis("off")

            axes[r, 3].imshow(item["overlay"])
            axes[r, 3].set_title(f"Overlay | IoU: {item['iou']:.2f} | {item['status']}")
            axes[r, 3].axis("off")

        plt.tight_layout()
        plot_path = os.path.join(plots_dir, "lane_detection_comparison.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"[✓] Qualitative plot saved to {plot_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Lane Detection on GTSRB/Road Lane Dataset")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="01_lane_detection/data/raw/dataset",
        help="Path to extracted dataset folder containing train/val/test splits",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="01_lane_detection/results",
        help="Directory to save evaluation results and metrics",
    )
    parser.add_argument("--max_samples", type=int, default=103, help="Max test samples to evaluate")
    args = parser.parse_args()

    metrics = evaluate_dataset(args.dataset_dir, args.output_dir, args.max_samples)
    print("\n--- Evaluation Summary ---")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
