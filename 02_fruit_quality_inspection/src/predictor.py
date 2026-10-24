"""
predictor.py
Inference engine for single images and image batches with confidence scores and quality categorization.
"""

import os
import glob
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

from .dataset import FRUIT_CLASSES, IMAGENET_MEAN, IMAGENET_STD
from .model import FruitMobileNetV2


class FruitQualityPredictor:
    def __init__(self, checkpoint_path: str, device_name: str = "auto", img_size: int = 224):
        if device_name == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_name)

        self.img_size = img_size
        self.model = FruitMobileNetV2(num_classes=len(FRUIT_CLASSES), pretrained=False).to(self.device)

        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            self.model.load_state_dict(state_dict)
            self.model.eval()
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def predict_image(self, img_path: str) -> Dict[str, Any]:
        with Image.open(img_path) as pil_img:
            rgb_img = pil_img.convert("RGB")
            tensor = self.transform(rgb_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        pred_class = FRUIT_CLASSES[pred_idx]
        confidence = float(probs[pred_idx])

        quality = "FRESH" if "fresh" in pred_class else "ROTTEN"
        fruit_type = pred_class.replace("fresh", "").replace("rotten", "")

        return {
            "image_path": img_path,
            "filename": os.path.basename(img_path),
            "predicted_class": pred_class,
            "quality": quality,
            "fruit_type": fruit_type,
            "confidence": confidence,
            "all_probabilities": {FRUIT_CLASSES[i]: float(probs[i]) for i in range(len(FRUIT_CLASSES))},
        }

    def predict_batch(self, input_dir: str, max_images: Optional[int] = None) -> List[Dict[str, Any]]:
        img_paths = sorted(
            glob.glob(os.path.join(input_dir, "*", "*.jpg"))
            + glob.glob(os.path.join(input_dir, "*", "*.png"))
            + glob.glob(os.path.join(input_dir, "*.jpg"))
            + glob.glob(os.path.join(input_dir, "*.png"))
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

    def render_prediction_overlay(self, img_path: str, pred_info: Dict[str, Any]) -> np.ndarray:
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")

        h, w = img_bgr.shape[:2]
        overlay = img_bgr.copy()

        quality = pred_info["quality"]
        banner_color = (0, 160, 0) if quality == "FRESH" else (0, 0, 220)
        text = f"{pred_info['predicted_class'].upper()} ({pred_info['confidence']*100:.1f}%)"

        # Header banner
        banner_h = max(36, int(h * 0.10))
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (20, 20, 20), -1)
        cv2.rectangle(overlay, (0, 0), (12, banner_h), banner_color, -1)
        font_scale = max(0.5, w / 600.0)
        cv2.putText(
            overlay,
            text,
            (20, int(banner_h * 0.7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return overlay
