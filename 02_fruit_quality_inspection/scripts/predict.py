#!/usr/bin/env python3
"""
02_fruit_quality_inspection/scripts/predict.py
Inference runner for single-image and batch fruit quality inspection.
"""

import os
import sys
import argparse
import json
import cv2

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.predictor import FruitQualityPredictor


def main():
    parser = argparse.ArgumentParser(description="Fruit Quality Predictor CLI")
    parser.add_argument("--image", type=str, default=None, help="Path to single input image")
    parser.add_argument("--input_dir", type=str, default=None, help="Directory containing images")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="02_fruit_quality_inspection/results/predictions",
        help="Directory to save visual overlays or CSV",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="02_fruit_quality_inspection/results/best_model.pth",
        help="Path to trained checkpoint",
    )
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu, cuda, auto)")
    args = parser.parse_args()

    predictor = FruitQualityPredictor(checkpoint_path=args.checkpoint, device_name=args.device)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.image:
        print(f"[+] Running single-image prediction: {args.image}")
        res = predictor.predict_image(args.image)
        overlay = predictor.render_prediction_overlay(args.image, res)
        out_img_path = os.path.join(args.output_dir, f"pred_{os.path.basename(args.image)}")
        cv2.imwrite(out_img_path, overlay)

        print(f"[✓] Predicted Class: {res['predicted_class']} (Confidence: {res['confidence']*100:.2f}%)")
        print(f"[✓] Quality:         {res['quality']}")
        print(f"[✓] Saved overlay to: {out_img_path}")

    elif args.input_dir:
        print(f"[+] Running batch predictions on: {args.input_dir}")
        results = predictor.predict_batch(args.input_dir, max_images=50)
        out_json_path = os.path.join(args.output_dir, "batch_predictions.json")
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"[✓] Completed batch predictions for {len(results)} images. Saved results to: {out_json_path}")
    else:
        # Default test sample
        test_sample = "02_fruit_quality_inspection/data/raw/dataset/test/freshapples"
        if os.path.exists(test_sample):
            files = os.listdir(test_sample)
            if files:
                sample_p = os.path.join(test_sample, files[0])
                print(f"[+] No input specified. Running on default test sample: {sample_p}")
                res = predictor.predict_image(sample_p)
                overlay = predictor.render_prediction_overlay(sample_p, res)
                out_img_path = os.path.join(args.output_dir, f"pred_{os.path.basename(sample_p)}")
                cv2.imwrite(out_img_path, overlay)
                print(f"[✓] Predicted: {res['predicted_class']} ({res['confidence']*100:.2f}%) -> {res['quality']}")
                print(f"[✓] Saved overlay to: {out_img_path}")
        else:
            print("Please specify --image or --input_dir.")


if __name__ == "__main__":
    main()
