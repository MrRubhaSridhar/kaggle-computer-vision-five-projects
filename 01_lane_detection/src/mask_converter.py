"""
mask_converter.py
Converts polygon-based YOLO segmentation annotations into binary ground-truth lane masks.
"""

import os
from typing import List, Tuple
import cv2
import numpy as np


def yolo_seg_to_mask(label_path: str, height: int, width: int) -> np.ndarray:
    """
    Reads a YOLO segmentation txt file containing polygon coordinates
    and rasterizes it into a binary mask of shape (height, width).
    
    Format in label file:
        <class_id> <x1> <y1> <x2> <y2> ... (normalized 0 to 1)
        
    Returns:
        np.ndarray of shape (height, width), uint8, where 255 represents lane and 0 is background.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    if not os.path.exists(label_path):
        return mask

    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    polygons: List[np.ndarray] = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        # parts[0] is class_id
        coords = [float(x) for x in parts[1:]]
        # Pair into (x, y)
        poly = []
        for i in range(0, len(coords) - 1, 2):
            px = int(coords[i] * width)
            py = int(coords[i + 1] * height)
            poly.append([px, py])
        if len(poly) >= 3:
            polygons.append(np.array(poly, dtype=np.int32))

    if polygons:
        cv2.fillPoly(mask, polygons, color=255)

    return mask
