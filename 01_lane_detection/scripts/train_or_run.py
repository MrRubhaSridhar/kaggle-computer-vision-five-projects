#!/usr/bin/env python3
"""
01_lane_detection/scripts/train_or_run.py
Inference runner for Lane Detection and Departure Warning.
Supports:
- Single image inference
- Batch directory inference
- Video file inference
"""

import os
import sys
import argparse
import cv2

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.lane_detector import LaneDetector
from src.video_pipeline import process_image_batch, process_video


def main():
    parser = argparse.ArgumentParser(description="Lane Detection & Departure Warning CLI")
    parser.add_argument("--image", type=str, default=None, help="Path to single input image")
    parser.add_argument("--input_dir", type=str, default=None, help="Directory containing images")
    parser.add_argument("--video", type=str, default=None, help="Path to input video file")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="01_lane_detection/results/overlays",
        help="Directory to save output visualisations",
    )
    parser.add_argument(
        "--departure_threshold",
        type=float,
        default=35.0,
        help="Pixel offset threshold for lane departure warning",
    )
    parser.add_argument("--max_frames", type=int, default=None, help="Max video frames to process")
    parser.add_argument(
        "--save_intermediates",
        action="store_true",
        help="Save Canny, ROI, Hough intermediate stages",
    )
    args = parser.parse_args()

    detector = LaneDetector(departure_threshold_px=args.departure_threshold)

    if args.image:
        print(f"[+] Processing single image: {args.image}")
        img = cv2.imread(args.image)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {args.image}")

        res = detector.process_frame(img, return_intermediates=args.save_intermediates)
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, f"out_{os.path.basename(args.image)}")
        cv2.imwrite(out_path, res["overlay"])
        print(f"[✓] Saved overlay to {out_path}")
        print(f"[i] Departure Info: {res['departure_info']}")

        if args.save_intermediates and "intermediates" in res:
            inter_dir = os.path.join(args.output_dir, "intermediate")
            os.makedirs(inter_dir, exist_ok=True)
            for k, v in res["intermediates"].items():
                p = os.path.join(inter_dir, f"{k}.jpg")
                cv2.imwrite(p, v)
            print(f"[✓] Saved intermediate images to {inter_dir}")

    elif args.video:
        print(f"[+] Processing video: {args.video}")
        out_video_path = os.path.join(args.output_dir, f"detected_{os.path.basename(args.video)}")
        process_video(detector, args.video, out_video_path, max_frames=args.max_frames)

    elif args.input_dir:
        print(f"[+] Processing batch from directory: {args.input_dir}")
        records = process_image_batch(
            detector,
            args.input_dir,
            args.output_dir,
            save_intermediates=args.save_intermediates,
        )
        print(f"[✓] Processed {len(records)} images.")
    else:
        # Default behavior: run on test images from data/raw/dataset/test/images
        default_dir = "01_lane_detection/data/raw/dataset/test/images"
        if os.path.exists(default_dir):
            print(f"[+] No input specified. Processing default test set: {default_dir}")
            records = process_image_batch(
                detector,
                default_dir,
                args.output_dir,
                max_images=15,
                save_intermediates=True,
            )
            print(f"[✓] Processed {len(records)} test images.")
        else:
            print("Please specify --image, --input_dir, or --video.")


if __name__ == "__main__":
    main()
