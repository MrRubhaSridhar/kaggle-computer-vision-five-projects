"""
video_pipeline.py
Handles batch image processing and video inference for Lane Detection and Departure Warning.
"""

import os
import glob
from typing import Optional, List
import cv2
import numpy as np
from tqdm import tqdm

from .lane_detector import LaneDetector


def process_image_batch(
    detector: LaneDetector,
    input_dir: str,
    output_dir: str,
    max_images: Optional[int] = None,
    save_intermediates: bool = False,
) -> List[dict]:
    """
    Processes all images in input_dir, saves rendered lane overlays to output_dir,
    and optionally saves intermediate processing steps.
    """
    os.makedirs(output_dir, exist_ok=True)
    if save_intermediates:
        inter_dir = os.path.join(output_dir, "intermediate")
        os.makedirs(inter_dir, exist_ok=True)

    img_paths = sorted(
        glob.glob(os.path.join(input_dir, "*.jpg"))
        + glob.glob(os.path.join(input_dir, "*.png"))
    )
    if max_images:
        img_paths = img_paths[:max_images]

    records = []
    for path in tqdm(img_paths, desc="Processing Image Batch"):
        fname = os.path.basename(path)
        img = cv2.imread(path)
        if img is None:
            continue

        result = detector.process_frame(img, return_intermediates=save_intermediates)
        out_path = os.path.join(output_dir, f"detected_{fname}")
        cv2.imwrite(out_path, result["overlay"])

        if save_intermediates and "intermediates" in result:
            base_no_ext = os.path.splitext(fname)[0]
            for step_name, step_img in result["intermediates"].items():
                cv2.imwrite(
                    os.path.join(inter_dir, f"{base_no_ext}_{step_name}.jpg"),
                    step_img,
                )

        records.append({
            "image": fname,
            "status": result["departure_info"]["status"],
            "offset_px": result["departure_info"]["offset_px"],
            "has_left_lane": result["left_lane"] is not None,
            "has_right_lane": result["right_lane"] is not None,
        })

    return records


def process_video(
    detector: LaneDetector,
    video_path: str,
    output_path: str,
    max_frames: Optional[int] = None,
) -> None:
    """
    Processes an input video file, drawing lane detection and departure warning,
    and writing the annotated video to output_path.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if max_frames and max_frames < total_frames:
        total_frames = max_frames

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    pbar = tqdm(total=total_frames, desc="Processing Video")
    while cap.isOpened() and (max_frames is None or frame_idx < max_frames):
        ret, frame = cap.read()
        if not ret:
            break

        result = detector.process_frame(frame, return_intermediates=False)
        out.write(result["overlay"])
        frame_idx += 1
        pbar.update(1)

    cap.release()
    out.release()
    pbar.close()
    print(f"[✓] Video saved to: {output_path}")
