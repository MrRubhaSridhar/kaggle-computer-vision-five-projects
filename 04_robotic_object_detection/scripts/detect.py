#!/usr/bin/env python3
"""
04_robotic_object_detection/scripts/detect.py
Inference runner for real-time robotic object detection on images and videos.
"""

import os
import sys
import argparse
import json
import cv2

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.detector import RoboticSafetyDetector
from src.video_pipeline import process_detection_video


def main():
    parser = argparse.ArgumentParser(description="Robotic Object Detection Inference CLI")
    parser.add_argument("--image", type=str, default=None, help="Path to input image")
    parser.add_argument("--video", type=str, default=None, help="Path to input video")
    parser.add_argument("--input_dir", type=str, default=None, help="Directory containing images")
    parser.add_argument(
        "--weights",
        type=str,
        default="04_robotic_object_detection/results/best.pt",
        help="Path to YOLO weights file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="04_robotic_object_detection/results/annotated_detections",
        help="Directory to save annotated detections",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU NMS threshold")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu, 0, auto)")
    parser.add_argument("--max_frames", type=int, default=150, help="Max frames for video inference")
    args = parser.parse_args()

    weights_path = args.weights
    if not os.path.exists(weights_path):
        print(f"[!] {weights_path} not found. Falling back to pretrained base yolov8n.pt")
        weights_path = "yolov8n.pt"

    detector = RoboticSafetyDetector(
        model_path=weights_path,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        device=args.device,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    if args.image:
        print(f"[+] Running detection on image: {args.image}")
        img = cv2.imread(args.image)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {args.image}")

        res = detector.detect(img)
        out_path = os.path.join(args.output_dir, f"out_{os.path.basename(args.image)}")
        cv2.imwrite(out_path, res["annotated_image"])

        print(f"[✓] Detected {len(res['detections'])} objects in {res['latency_ms']:.1f}ms ({res['fps']:.1f} FPS)")
        for det in res["detections"]:
            print(f"    - {det['class_name']}: {det['confidence']*100:.1f}% at {det['bbox_xyxy']}")
        print(f"[✓] Saved annotated image to: {out_path}")

    elif args.video:
        print(f"[+] Running detection on video: {args.video}")
        out_video_path = os.path.join(args.output_dir, f"detected_{os.path.basename(args.video)}")
        process_detection_video(detector, args.video, out_video_path, max_frames=args.max_frames)

    elif args.input_dir:
        print(f"[+] Running batch detection on folder: {args.input_dir}")
        img_files = sorted(os.listdir(args.input_dir))[:20]
        batch_results = []
        for fn in img_files:
            fp = os.path.join(args.input_dir, fn)
            frame = cv2.imread(fp)
            if frame is None:
                continue
            res = detector.detect(frame)
            out_p = os.path.join(args.output_dir, f"detected_{fn}")
            cv2.imwrite(out_p, res["annotated_image"])
            batch_results.append({
                "file": fn,
                "objects": len(res["detections"]),
                "latency_ms": res["latency_ms"],
            })
        print(f"[✓] Processed {len(batch_results)} images. Saved annotated frames to {args.output_dir}")

    else:
        # Default test sample
        sample_img = "04_robotic_object_detection/data/raw/css-data/test/images/000005_jpg.rf.96e9379ccae638140c4a90fc4b700a2b.jpg"
        if os.path.exists(sample_img):
            print(f"[+] No input specified. Running detection on sample image: {sample_img}")
            img = cv2.imread(sample_img)
            res = detector.detect(img)
            out_path = os.path.join(args.output_dir, f"out_sample.jpg")
            cv2.imwrite(out_path, res["annotated_image"])
            print(f"[✓] Detected {len(res['detections'])} objects in {res['latency_ms']:.1f}ms ({res['fps']:.1f} FPS)")
            print(f"[✓] Saved annotated image to: {out_path}")
        else:
            print("Please specify --image, --video, or --input_dir.")


if __name__ == "__main__":
    main()
