"""
detector.py
Real-Time YOLO Object Detector engineered for robotic vision pipelines.
Includes per-frame latency measurement, FPS calculation, bounding box rendering,
and architectural specifications for ROS 2 node integration.
"""

import os
import time
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
from ultralytics import YOLO

from .dataset_validator import SAFETY_CLASSES, CLASS_COLORS


class RoboticSafetyDetector:
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
        device: str = "auto",
    ):
        import torch
        if device == "auto":
            device = 0 if torch.cuda.is_available() else "cpu"
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.model = YOLO(model_path)
        self.device = device
        # Warmup
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model(dummy, verbose=False)

    def detect(self, img_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Runs object detection on a single BGR frame.
        Measures inference latency and calculates instantaneous FPS.
        """
        start_time = time.perf_counter()
        results = self.model.predict(
            source=img_bgr,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            device=self.device,
            verbose=False,
        )
        elapsed_sec = time.perf_counter() - start_time
        latency_ms = elapsed_sec * 1000.0
        fps = 1.0 / elapsed_sec if elapsed_sec > 0 else 0.0

        detections = []
        res = results[0]
        boxes = res.boxes

        annotated_img = img_bgr.copy()
        h, w = img_bgr.shape[:2]

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = [int(v) for v in xyxy]
                conf = float(box.conf[0].cpu().numpy())
                cid = int(box.cls[0].cpu().numpy())
                cname = SAFETY_CLASSES[cid] if cid < len(SAFETY_CLASSES) else f"Class_{cid}"
                color = CLASS_COLORS[cid % len(CLASS_COLORS)]

                # Bounding box
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)

                # Label badge
                badge_text = f"{cname} {conf*100:.1f}%"
                (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(annotated_img, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, max(th + 6, y1)), color, -1)
                cv2.putText(
                    annotated_img,
                    badge_text,
                    (x1 + 2, max(th, y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255) if color[1] < 180 else (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

                # Robotic bounding box representation (normalized center + dims)
                detections.append({
                    "class_id": cid,
                    "class_name": cname,
                    "confidence": float(round(conf, 4)),
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "robotic_center_norm": [
                        float(round(((x1 + x2) / 2.0) / w, 4)),
                        float(round(((y1 + y2) / 2.0) / h, 4)),
                    ],
                })

        # Telemetry HUD banner (Latency, FPS, Detection Count)
        hud_text = f"Inference: {latency_ms:.1f}ms | FPS: {fps:.1f} | Objects: {len(detections)}"
        cv2.rectangle(annotated_img, (0, 0), (w, 32), (20, 20, 20), -1)
        cv2.putText(
            annotated_img,
            hud_text,
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return {
            "detections": detections,
            "latency_ms": float(round(latency_ms, 2)),
            "fps": float(round(fps, 1)),
            "annotated_image": annotated_img,
        }

    @staticmethod
    def get_ros2_schema_documentation() -> str:
        """
        Explains how detection results map to standard ROS 2 message types
        for robotic integration.
        """
        return """
# ROS 2 Robotic Vision Integration Architecture:
#
# Standard Message Type: vision_msgs/msg/Detection2DArray
# Topic: /robot/vision/safety_detections
#
# For each detected object:
# - detection.results[0].hypothesis.class_id = str(cid) # e.g. "Hardhat" or "Person"
# - detection.results[0].hypothesis.score = conf
# - detection.bbox.center.position.x = (x1 + x2) / 2.0
# - detection.bbox.center.position.y = (y1 + y2) / 2.0
# - detection.bbox.size_x = x2 - x1
# - detection.bbox.size_y = y2 - y1
#
# Published at robot camera frame rate (e.g. 30 Hz).
# Downstream navigation node (Nav2) can query /robot/vision/safety_detections
# to stop robot or trigger acoustic alarms when NO-Hardhat or Person is in hazardous machinery zones.
"""
