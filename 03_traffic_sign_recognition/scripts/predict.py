#!/usr/bin/env python3
"""
03_traffic_sign_recognition/scripts/predict.py
Inference runner for Traffic Sign Recognition.
Supports single-image prediction and batch predictions.
"""

import os
import sys
import argparse
import json
import cv2

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.predictor import TrafficSignPredictor


def main():
    parser = argparse.ArgumentParser(description="Traffic Sign Predictor CLI")
    parser.add_argument("--image", type=str, default=None, help="Path to single input image")
    parser.add_argument("--input_dir", type=str, default=None, help="Directory containing images")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="03_traffic_sign_recognition/results/predictions",
        help="Directory to save visual overlays or CSV",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="03_traffic_sign_recognition/results/best_model.pth",
        help="Path to trained checkpoint",
    )
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu, cuda, auto)")
    args = parser.parse_args()

    predictor = TrafficSignPredictor(checkpoint_path=args.checkpoint, device_name=args.device)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.image:
        print(f"[+] Predicting single image: {args.image}")
        res = predictor.predict_image(args.image)
        overlay = predictor.render_overlay(args.image, res)
        out_path = os.path.join(args.output_dir, f"pred_{os.path.basename(args.image)}")
        cv2.imwrite(out_path, overlay)

        print(f"[✓] Predicted: Class {res['predicted_class_id']} - {res['predicted_name']} ({res['confidence']*100:.2f}%)")
        print(f"[✓] Saved overlay to: {out_path}")

    elif args.input_dir:
        print(f"[+] Predicting batch from directory: {args.input_dir}")
        results = predictor.predict_batch(args.input_dir, max_images=50)
        out_json = os.path.join(args.output_dir, "batch_predictions.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"[✓] Processed {len(results)} images. Results saved to: {out_json}")

    else:
        # Default test sample
        default_sample = "03_traffic_sign_recognition/data/raw/Test/00000.png"
        if os.path.exists(default_sample):
            print(f"[+] No input specified. Running on sample test image: {default_sample}")
            res = predictor.predict_image(default_sample)
            overlay = predictor.render_overlay(default_sample, res)
            out_path = os.path.join(args.output_dir, f"pred_{os.path.basename(default_sample)}")
            cv2.imwrite(out_path, overlay)
            print(f"[✓] Predicted: Class {res['predicted_class_id']} - {res['predicted_name']} ({res['confidence']*100:.2f}%)")
            print(f"[✓] Saved overlay to: {out_path}")
        else:
            print("Please specify --image or --input_dir.")


if __name__ == "__main__":
    main()
