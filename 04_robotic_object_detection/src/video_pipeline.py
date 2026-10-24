"""
video_pipeline.py
Video processing pipeline for real-time robotic object detection and performance benchmarking.
"""

import os
from typing import Optional
import cv2
from tqdm import tqdm

from .detector import RoboticSafetyDetector


def process_detection_video(
    detector: RoboticSafetyDetector,
    video_path: str,
    output_path: str,
    max_frames: Optional[int] = 150,
) -> None:
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

    frame_count = 0
    latencies = []
    pbar = tqdm(total=total_frames, desc="Processing Robotic Video")

    while cap.isOpened() and (max_frames is None or frame_count < max_frames):
        ret, frame = cap.read()
        if not ret:
            break

        res = detector.detect(frame)
        out.write(res["annotated_image"])
        latencies.append(res["latency_ms"])
        frame_count += 1
        pbar.update(1)

    cap.release()
    out.release()
    pbar.close()

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    avg_fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0
    print(f"[✓] Processed {frame_count} frames. Average Latency: {avg_latency:.2f}ms (~{avg_fps:.1f} FPS)")
    print(f"[✓] Annotated video saved to: {output_path}")
