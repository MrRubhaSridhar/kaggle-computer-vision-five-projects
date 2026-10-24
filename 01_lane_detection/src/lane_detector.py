"""
lane_detector.py
Classical Computer Vision pipeline for Lane Detection and Lane Departure Warning.
Includes:
- Grayscale conversion
- Gaussian blur
- Canny edge detection
- Trapezoidal Region of Interest (ROI) masking
- Hough Line Transform (cv2.HoughLinesP)
- Slope filtering & clustering into Left/Right lane lines
- Lane line extrapolation and lane surface polygon overlay
- Lane Departure Warning (LDW) approximation based on vehicle center offset
"""

import os
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np


class LaneDetector:
    def __init__(
        self,
        canny_low: int = 50,
        canny_high: int = 150,
        blur_kernel: int = 5,
        hough_threshold: int = 25,
        min_line_len: int = 20,
        max_line_gap: int = 30,
        departure_threshold_px: float = 35.0,
        roi_top_pct: float = 0.60,
        roi_bottom_pct: float = 0.95,
        target_size: Optional[Tuple[int, int]] = (640, 640),
    ):
        """
        Parameters:
            canny_low: Lower hysteresis threshold for Canny edge detector
            canny_high: Upper hysteresis threshold for Canny edge detector
            blur_kernel: Kernel size for Gaussian blur (odd integer)
            hough_threshold: Hough accumulator threshold
            min_line_len: Minimum line segment length in HoughLinesP
            max_line_gap: Maximum allowed gap between points on the same line
            departure_threshold_px: Pixel offset threshold for lane departure warning
            roi_top_pct: Fraction of image height where trapezoid top begins
            roi_bottom_pct: Fraction of image height where trapezoid bottom ends
            target_size: (width, height) to resize input images (or None to keep original)
        """
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        self.hough_threshold = hough_threshold
        self.min_line_len = min_line_len
        self.max_line_gap = max_line_gap
        self.departure_threshold_px = departure_threshold_px
        self.roi_top_pct = roi_top_pct
        self.roi_bottom_pct = roi_bottom_pct
        self.target_size = target_size

    def preprocess(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Resizes (if target_size set), converts to grayscale, and applies Gaussian blur."""
        if self.target_size is not None and (img_bgr.shape[1], img_bgr.shape[0]) != self.target_size:
            img_bgr = cv2.resize(img_bgr, self.target_size, interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)
        return img_bgr, gray, blurred

    def detect_edges(self, blurred: np.ndarray) -> np.ndarray:
        """Applies Canny edge detection."""
        return cv2.Canny(blurred, self.canny_low, self.canny_high)

    def get_roi_mask(self, h: int, w: int) -> np.ndarray:
        """Creates a trapezoidal Region of Interest (ROI) mask focused on the road ahead."""
        mask = np.zeros((h, w), dtype=np.uint8)
        # Trapezoid vertices focusing on driving corridor
        top_y = int(h * self.roi_top_pct)
        bottom_y = int(h * self.roi_bottom_pct)

        vertices = np.array(
            [
                [
                    (int(w * 0.05), bottom_y),
                    (int(w * 0.42), top_y),
                    (int(w * 0.58), top_y),
                    (int(w * 0.95), bottom_y),
                ]
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, vertices, 255)
        return mask

    def apply_roi(self, edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Masks edges to the Region of Interest."""
        h, w = edges.shape[:2]
        roi_mask = self.get_roi_mask(h, w)
        masked_edges = cv2.bitwise_and(edges, roi_mask)
        return masked_edges, roi_mask

    def detect_hough_lines(self, masked_edges: np.ndarray) -> Optional[np.ndarray]:
        """Runs Probabilistic Hough Line Transform."""
        return cv2.HoughLinesP(
            masked_edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_len,
            maxLineGap=self.max_line_gap,
        )

    def separate_and_fit_lanes(
        self, lines: Optional[np.ndarray], h: int, w: int
    ) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[Tuple[int, int, int, int]]]:
        """
        Separates Hough line segments into left and right lanes by slope.
        Left lane: negative slope in image coordinates (slope < -0.3)
        Right lane: positive slope in image coordinates (slope > 0.3)
        Fits linear models (x = m*y + b) for robust extrapolation across the ROI.
        """
        if lines is None or len(lines) == 0:
            return None, None

        left_x: List[int] = []
        left_y: List[int] = []
        right_x: List[int] = []
        right_y: List[int] = []

        mid_x = w // 2
        flat_lines = lines.reshape(-1, 4)

        for x1, y1, x2, y2 in flat_lines:
            if x1 == x2:
                continue  # skip vertical lines to prevent division by zero
            slope = (y2 - y1) / (x2 - x1)

            # Left lane in image space: slope is negative, segment predominantly on the left side
            if slope < -0.3 and (x1 < mid_x + 50 or x2 < mid_x + 50):
                left_x.extend([x1, x2])
                left_y.extend([y1, y2])
            # Right lane in image space: slope is positive, segment predominantly on the right side
            elif slope > 0.3 and (x1 > mid_x - 50 or x2 > mid_x - 50):
                right_x.extend([x1, x2])
                right_y.extend([y1, y2])

        y_bottom = int(h * self.roi_bottom_pct)
        y_top = int(h * self.roi_top_pct)

        def fit_line(x_pts: List[int], y_pts: List[int]) -> Optional[Tuple[int, int, int, int]]:
            if len(x_pts) < 2:
                return None
            try:
                # Fit x = f(y) polynomial degree 1
                poly = np.polyfit(y_pts, x_pts, deg=1)
                x_bottom = int(poly[0] * y_bottom + poly[1])
                x_top = int(poly[0] * y_top + poly[1])
                # Boundary clamping
                x_bottom = max(-w, min(2 * w, x_bottom))
                x_top = max(0, min(w, x_top))
                return (x_bottom, y_bottom, x_top, y_top)
            except Exception:
                return None

        left_lane = fit_line(left_x, left_y)
        right_lane = fit_line(right_x, right_y)

        return left_lane, right_lane

    def compute_departure_warning(
        self,
        left_lane: Optional[Tuple[int, int, int, int]],
        right_lane: Optional[Tuple[int, int, int, int]],
        w: int,
    ) -> Dict[str, Any]:
        """
        Approximates lane departure warning:
        - Computes lane center at the bottom of the image
        - Compares with vehicle center (assumed w / 2)
        - Computes offset = vehicle_center - lane_center
        """
        camera_center = w / 2.0
        status = "IN_LANE"
        offset_px = 0.0
        confidence = "HIGH"

        if left_lane is not None and right_lane is not None:
            left_bottom_x = left_lane[0]
            right_bottom_x = right_lane[0]
            lane_center = (left_bottom_x + right_bottom_x) / 2.0
            offset_px = camera_center - lane_center
            if offset_px > self.departure_threshold_px:
                status = "DEPARTURE_RIGHT"
            elif offset_px < -self.departure_threshold_px:
                status = "DEPARTURE_LEFT"
            else:
                status = "IN_LANE"
        elif left_lane is not None:
            # Only left lane detected
            offset_px = camera_center - (left_lane[0] + 180)
            confidence = "MEDIUM"
            if offset_px > self.departure_threshold_px:
                status = "DEPARTURE_RIGHT"
            else:
                status = "IN_LANE"
        elif right_lane is not None:
            # Only right lane detected
            offset_px = camera_center - (right_lane[0] - 180)
            confidence = "MEDIUM"
            if offset_px < -self.departure_threshold_px:
                status = "DEPARTURE_LEFT"
            else:
                status = "IN_LANE"
        else:
            status = "NO_LANES_DETECTED"
            confidence = "LOW"
            offset_px = 0.0

        return {
            "status": status,
            "offset_px": float(offset_px),
            "camera_center": float(camera_center),
            "confidence": confidence,
        }

    def render_overlay(
        self,
        img_bgr: np.ndarray,
        left_lane: Optional[Tuple[int, int, int, int]],
        right_lane: Optional[Tuple[int, int, int, int]],
        departure_info: Dict[str, Any],
    ) -> np.ndarray:
        """Renders lane boundaries, lane corridor polygon, and status HUD banner."""
        h, w = img_bgr.shape[:2]
        overlay = img_bgr.copy()
        lane_surface = np.zeros_like(img_bgr)

        # Draw filled lane polygon if both lanes detected
        if left_lane is not None and right_lane is not None:
            pts = np.array(
                [
                    [left_lane[0], left_lane[1]],
                    [left_lane[2], left_lane[3]],
                    [right_lane[2], right_lane[3]],
                    [right_lane[0], right_lane[1]],
                ],
                dtype=np.int32,
            )
            # Safe corridor color: green if IN_LANE, amber if departure
            color = (0, 220, 0) if departure_info["status"] == "IN_LANE" else (0, 140, 255)
            cv2.fillPoly(lane_surface, [pts], color)
            # Blend lane surface
            cv2.addWeighted(lane_surface, 0.35, overlay, 1.0, 0, overlay)

        # Draw lane line strokes
        if left_lane is not None:
            cv2.line(overlay, (left_lane[0], left_lane[1]), (left_lane[2], left_lane[3]), (0, 0, 255), 5)
        if right_lane is not None:
            cv2.line(overlay, (right_lane[0], right_lane[1]), (right_lane[2], right_lane[3]), (255, 0, 0), 5)

        # HUD Banner
        status = departure_info["status"]
        offset = departure_info["offset_px"]
        if status == "IN_LANE":
            banner_color = (0, 160, 0)
            text = f"LANE TRACKING: NORMAL | Offset: {offset:+.1f}px"
        elif status == "DEPARTURE_LEFT":
            banner_color = (0, 0, 220)
            text = f"WARNING: DEPARTURE LEFT! | Offset: {offset:+.1f}px"
        elif status == "DEPARTURE_RIGHT":
            banner_color = (0, 0, 220)
            text = f"WARNING: DEPARTURE RIGHT! | Offset: {offset:+.1f}px"
        else:
            banner_color = (120, 120, 120)
            text = "WARNING: NO LANES DETECTED"

        # Top banner background
        cv2.rectangle(overlay, (0, 0), (w, 48), (20, 20, 20), -1)
        cv2.circle(overlay, (24, 24), 10, banner_color, -1)
        cv2.putText(
            overlay,
            text,
            (45, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return overlay

    def generate_pred_mask(
        self,
        left_lane: Optional[Tuple[int, int, int, int]],
        right_lane: Optional[Tuple[int, int, int, int]],
        h: int,
        w: int,
        line_thickness: int = 15,
    ) -> np.ndarray:
        """
        Generates a binary lane prediction mask (255 for lane markings, 0 for background)
        for quantitative evaluation against ground-truth masks.
        """
        mask = np.zeros((h, w), dtype=np.uint8)
        if left_lane is not None:
            cv2.line(
                mask,
                (left_lane[0], left_lane[1]),
                (left_lane[2], left_lane[3]),
                255,
                line_thickness,
            )
        if right_lane is not None:
            cv2.line(
                mask,
                (right_lane[0], right_lane[1]),
                (right_lane[2], right_lane[3]),
                255,
                line_thickness,
            )
        return mask

    def process_frame(
        self, img_bgr: np.ndarray, return_intermediates: bool = False
    ) -> Dict[str, Any]:
        """
        Full inference step on a single frame or image.
        Returns overlay, prediction mask, departure info, and optionally intermediate steps.
        """
        img, gray, blurred = self.preprocess(img_bgr)
        h, w = img.shape[:2]

        edges = self.detect_edges(blurred)
        masked_edges, roi_mask = self.apply_roi(edges)
        lines = self.detect_hough_lines(masked_edges)
        left_lane, right_lane = self.separate_and_fit_lanes(lines, h, w)
        departure_info = self.compute_departure_warning(left_lane, right_lane, w)
        overlay = self.render_overlay(img, left_lane, right_lane, departure_info)
        pred_mask = self.generate_pred_mask(left_lane, right_lane, h, w)

        res: Dict[str, Any] = {
            "overlay": overlay,
            "pred_mask": pred_mask,
            "departure_info": departure_info,
            "left_lane": left_lane,
            "right_lane": right_lane,
        }

        if return_intermediates:
            # Create Hough line visual
            hough_visual = cv2.cvtColor(masked_edges, cv2.COLOR_GRAY2BGR)
            if lines is not None and len(lines) > 0:
                for x1, y1, x2, y2 in lines.reshape(-1, 4):
                    cv2.line(hough_visual, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)

            res["intermediates"] = {
                "grayscale": gray,
                "blurred": blurred,
                "edges": edges,
                "roi_mask": roi_mask,
                "masked_edges": masked_edges,
                "hough_visual": hough_visual,
            }

        return res
