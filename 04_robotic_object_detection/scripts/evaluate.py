#!/usr/bin/env python3
"""
04_robotic_object_detection/scripts/evaluate.py
Evaluates YOLOv8n detector on the test split.
Measures mAP@0.5, mAP@0.5:0.95, Precision, Recall, Inference Latency (ms), and FPS.
Saves annotated detection visualisations and metrics JSON.
"""

import os
import sys
import glob
import json
import argparse
import time
from typing import Dict, Any

import cv2
import numpy as np
from ultralytics import YOLO

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataset_validator import SAFETY_CLASSES
from src.detector import RoboticSafetyDetector


def evaluate_yolo(
    weights_path: str,
    data_yaml: str,
    raw_dir: str,
    output_dir: str,
    device: str = "auto",
    save_annotated_count: int = 10,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    annotated_out_dir = os.path.join(output_dir, "annotated_detections")
    os.makedirs(annotated_out_dir, exist_ok=True)

    import torch
    if device == "auto":
        device = 0 if torch.cuda.is_available() else "cpu"

    print(f"[+] Loading YOLO model from: {weights_path}")
    model = YOLO(weights_path)

    # 1. Ultralytics validation on test split
    print("[+] Running Ultralytics validation on test split...")
    val_results = model.val(data=data_yaml, split="test", device=device, verbose=False)

    metrics_dict = val_results.results_dict
    precision = float(round(metrics_dict.get("metrics/precision(B)", 0.0), 4))
    recall = float(round(metrics_dict.get("metrics/recall(B)", 0.0), 4))
    map50 = float(round(metrics_dict.get("metrics/mAP50(B)", 0.0), 4))
    map50_95 = float(round(metrics_dict.get("metrics/mAP50-95(B)", 0.0), 4))

    # 2. Benchmarking inference latency & FPS over test images
    test_img_dir = os.path.join(raw_dir, "css-data", "test", "images")
    if not os.path.exists(test_img_dir):
        test_img_dir = os.path.join(raw_dir, "test", "images")

    test_imgs = sorted(glob.glob(os.path.join(test_img_dir, "*.jpg")) + glob.glob(os.path.join(test_img_dir, "*.png")))
    detector = RoboticSafetyDetector(model_path=weights_path, device=device)

    latencies = []
    saved_count = 0

    print(f"[+] Benchmarking inference latency on {len(test_imgs)} test images...")
    for idx, imp in enumerate(test_imgs):
        frame = cv2.imread(imp)
        if frame is None:
            continue
        res = detector.detect(frame)
        latencies.append(res["latency_ms"])

        if saved_count < save_annotated_count:
            fname = os.path.basename(imp)
            out_p = os.path.join(annotated_out_dir, f"detected_{fname}")
            cv2.imwrite(out_p, res["annotated_image"])
            saved_count += 1

    avg_latency = float(round(float(np.mean(latencies)), 2)) if latencies else 0.0
    p95_latency = float(round(float(np.percentile(latencies, 95)), 2)) if latencies else 0.0
    avg_fps = float(round(1000.0 / avg_latency, 1)) if avg_latency > 0 else 0.0

    eval_summary = {
        "dataset": "snehilsanyal/construction-site-safety-image-dataset-roboflow",
        "split": "test",
        "total_test_images": len(test_imgs),
        "classes": SAFETY_CLASSES,
        "detection_metrics": {
            "mAP50": map50,
            "mAP50_95": map50_95,
            "precision": precision,
            "recall": recall,
        },
        "performance_telemetry": {
            "average_latency_ms": avg_latency,
            "p95_latency_ms": p95_latency,
            "fps": avg_fps,
            "hardware_device": str(detector.device),
        },
        "ros2_integration": {
            "message_type": "vision_msgs/msg/Detection2DArray",
            "topic": "/robot/vision/safety_detections",
            "description": "Ready for publication into ROS 2 node. Coordinates normalized relative to camera frame.",
        },
    }

    metrics_file = os.path.join(output_dir, "metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)

    print(f"[✓] Evaluation metrics saved to: {metrics_file}")
    print(f"[✓] Saved {saved_count} annotated sample detections to: {annotated_out_dir}")
    return eval_summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8n on Construction Site Safety Dataset")
    parser.add_argument(
        "--weights",
        type=str,
        default="04_robotic_object_detection/results/best.pt",
        help="Path to YOLO weights file",
    )
    parser.add_argument(
        "--data_yaml",
        type=str,
        default="04_robotic_object_detection/results/data.yaml",
        help="Path to data.yaml",
    )
    parser.add_argument(
        "--raw_dir",
        type=str,
        default="04_robotic_object_detection/data/raw",
        help="Path to raw dataset folder",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="04_robotic_object_detection/results",
        help="Directory to save evaluation artifacts",
    )
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu, 0, auto)")
    args = parser.parse_args()

    # Fallback to yolov8n.pt if best.pt does not exist
    weights_path = args.weights
    if not os.path.exists(weights_path):
        print(f"[!] {weights_path} not found. Falling back to pretrained base yolov8n.pt")
        weights_path = "yolov8n.pt"

    eval_summary = evaluate_yolo(
        weights_path=weights_path,
        data_yaml=args.data_yaml,
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        device=args.device,
    )

    print("\n--- Evaluation Summary ---")
    print(f"mAP@0.5:      {eval_summary['detection_metrics']['mAP50']:.4f}")
    print(f"mAP@0.5:0.95: {eval_summary['detection_metrics']['mAP50_95']:.4f}")
    print(f"Precision:    {eval_summary['detection_metrics']['precision']:.4f}")
    print(f"Recall:       {eval_summary['detection_metrics']['recall']:.4f}")
    print(f"Avg Latency:  {eval_summary['performance_telemetry']['average_latency_ms']} ms ({eval_summary['performance_telemetry']['fps']} FPS)")


if __name__ == "__main__":
    main()
