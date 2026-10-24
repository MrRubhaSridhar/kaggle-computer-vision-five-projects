"""
predictor.py
Inference pipeline for Traffic Sign Recognition with confidence scores and class name mapping.
"""

import os
import glob
from typing import Dict, List, Optional, Any

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

from .dataset import GTSRB_CLASSES, IMAGENET_MEAN, IMAGENET_STD
from .model import TrafficSignCNN


class TrafficSignPredictor:
    def __init__(self, checkpoint_path: str, device_name: str = "auto", img_size: int = 48):
        if device_name == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_name)

        self.img_size = img_size
        self.model = TrafficSignCNN(num_classes=43).to(self.device)

        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            self.model.load_state_dict(state_dict)
            self.model.eval()
        else:
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def predict_image(self, img_path: str, crop_box: Optional[tuple] = None) -> Dict[str, Any]:
        with Image.open(img_path) as pil_img:
            img = pil_img.convert("RGB")
            if crop_box is not None:
                img = img.crop(crop_box)
            tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        top_indices = np.argsort(probs)[::-1][:3]
        pred_id = int(top_indices[0])
        confidence = float(probs[pred_id])

        top_candidates = [
            {"class_id": int(cid), "name": GTSRB_CLASSES.get(cid, "Unknown"), "confidence": float(probs[cid])}
            for cid in top_indices
        ]

        return {
            "image_path": img_path,
            "filename": os.path.basename(img_path),
            "predicted_class_id": pred_id,
            "predicted_name": GTSRB_CLASSES.get(pred_id, "Unknown"),
            "confidence": confidence,
            "top_candidates": top_candidates,
        }

    def predict_batch(self, input_dir: str, max_images: Optional[int] = None) -> List[Dict[str, Any]]:
        img_paths = sorted(
            glob.glob(os.path.join(input_dir, "*", "*.png"))
            + glob.glob(os.path.join(input_dir, "*.png"))
            + glob.glob(os.path.join(input_dir, "*.jpg"))
        )
        if max_images:
            img_paths = img_paths[:max_images]

        results = []
        for p in img_paths:
            try:
                results.append(self.predict_image(p))
            except Exception as e:
                print(f"[!] Error predicting {p}: {e}")
        return results

    def render_overlay(self, img_path: str, pred_info: Dict[str, Any]) -> np.ndarray:
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")

        # Resize display frame for clear inspection if original is tiny
        h, w = img_bgr.shape[:2]
        display_scale = max(1, 200 // max(h, w))
        display_img = cv2.resize(img_bgr, (w * display_scale, h * display_scale), interpolation=cv2.INTER_NEAREST)

        dh, dw = display_img.shape[:2]
        canvas = np.full((dh + 60, dw, 3), 25, dtype=np.uint8)
        canvas[:dh, :] = display_img

        text = f"{pred_info['predicted_name'][:22]}"
        conf_text = f"Conf: {pred_info['confidence']*100:.1f}%"

        cv2.putText(canvas, text, (8, dh + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, conf_text, (8, dh + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 1, cv2.LINE_AA)

        return canvas
