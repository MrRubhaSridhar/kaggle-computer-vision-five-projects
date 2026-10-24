"""
gradcam.py
Self-contained, production-grade Grad-CAM (Gradient-weighted Class Activation Mapping) implementation.
Hooks activations and gradients of a target convolutional layer to explain model predictions.
"""

from typing import Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.handlers = []

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self.handlers.extend([h1, h2])

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, float]:
        """
        Generates Grad-CAM heatmap for input_tensor (1, C, H, W).
        Returns:
            heatmap: (H, W) float32 in [0, 1]
            pred_class: int
            pred_prob: float
        """
        self.model.eval()
        self.model.zero_grad()

        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1)
        pred_class = int(torch.argmax(probs, dim=1).item())
        pred_prob = float(probs[0, pred_class].item())

        if target_class is None:
            target_class = pred_class

        target_score = logits[0, target_class]
        target_score.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations were not captured by hooks.")

        # Global average pooling on gradients: alpha_k = 1/Z sum_i sum_j (grad)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of forward activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)  # Only positive influence

        # Upsample to input tensor spatial dimensions
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Min-max normalization
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            heatmap = (cam - cam_min) / (cam_max - cam_min)
        else:
            heatmap = np.zeros_like(cam)

        return heatmap, pred_class, pred_prob

    def remove_hooks(self):
        for h in self.handlers:
            h.remove()
        self.handlers.clear()


def create_gradcam_overlay(
    orig_img_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Overlays a Grad-CAM heatmap onto an RGB image.
    orig_img_rgb: uint8 array of shape (H, W, 3) in [0, 255]
    heatmap: float32 array of shape (H, W) in [0.0, 1.0]
    
    Returns:
        heatmap_colored_rgb: (H, W, 3) uint8
        overlay_rgb: (H, W, 3) uint8
    """
    # Resize heatmap if shape mismatch
    h, w = orig_img_rgb.shape[:2]
    if heatmap.shape[:2] != (h, w):
        heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

    heatmap_uint8 = np.uint8(255 * heatmap)
    colored_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)

    overlay_float = (1.0 - alpha) * orig_img_rgb.astype(np.float32) + alpha * heatmap_colored_rgb.astype(np.float32)
    overlay_rgb = np.clip(overlay_float, 0, 255).astype(np.uint8)

    return heatmap_colored_rgb, overlay_rgb
