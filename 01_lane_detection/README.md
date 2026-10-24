# Project 1: Road Lane Detection and Departure Warning

An end-to-end Classical Computer Vision pipeline for detecting road lane boundaries, extrapolating vehicle driving corridors, and estimating real-time Lane Departure Warnings (LDW). Evaluated against genuine Kaggle ground-truth polygon masks.

---

## 📌 Dataset Information
- **Dataset Title**: Road Lane SEGMENTATION - IMGs & Labels
- **Kaggle Source**: [princekhunt19/road-lane-segmentation-imgs-and-labels](https://www.kaggle.com/datasets/princekhunt19/road-lane-segmentation-imgs-and-labels)
- **License**: CC0 1.0 (Public Domain)
- **Structure**: 640×640 RGB road images with YOLO polygon segmentation annotations (`0 x1 y1 x2 y2 ...`).
- **Splits**: Train (815), Validation (101), Test (103).

### Download Command
```bash
kaggle datasets download -d princekhunt19/road-lane-segmentation-imgs-and-labels -p 01_lane_detection/data/raw --unzip
```

---

## 🛠️ Pipeline Architecture
1. **Grayscale Conversion & Gaussian Filtering**: Eliminates sensor noise and tarmac grain.
2. **Canny Edge Detection**: Identifies sharp gradient boundaries ($T_{low}=50, T_{high}=150$).
3. **Trapezoidal ROI Masking**: Focuses computation strictly on the forward driving lane.
4. **Probabilistic Hough Line Transform**: Detects line segments with minimum length and gap constraints.
5. **Slope Partitioning & Extrapolation**: Clusters lines into left ($m < -0.3$) and right ($m > 0.3$) lanes and fits linear models ($x = m \cdot y + b$).
6. **Lane Departure Warning (LDW)**: Compares lane corridor midpoint against camera optical center; displays real-time HUD status and alerts.
7. **Quantitative Evaluation**: Converts YOLO polygons into binary ground-truth masks to measure Precision, Recall, F1, and IoU.

---

## 🚀 Reproduction & Usage

### 1. Evaluate Pipeline on Test Set
```bash
python 01_lane_detection/scripts/evaluate.py --dataset_dir 01_lane_detection/data/raw/dataset --output_dir 01_lane_detection/results --max_samples 103
```

### 2. Run Inference on a Single Image
```bash
python 01_lane_detection/scripts/train_or_run.py --image 01_lane_detection/data/raw/dataset/test/images/i789.jpg --save_intermediates
```

### 3. Run Inference on a Batch of Images
```bash
python 01_lane_detection/scripts/train_or_run.py --input_dir 01_lane_detection/data/raw/dataset/test/images --output_dir 01_lane_detection/results/overlays
```

### 4. Run Inference on a Video
```bash
python 01_lane_detection/scripts/train_or_run.py --video path/to/driving_video.mp4 --output_dir 01_lane_detection/results/sample_clips
```

---

## 📊 Evaluation Results (103 Test Images)

| Metric | Measured Value |
| :--- | :--- |
| **Precision** | **0.4583** |
| **Recall** | **0.0750** |
| **F1-Score** | **0.1290** |
| **IoU** | **0.0689** |
| **In-Lane Tracking** | **77 frames (74.8%)** |
| **Departure Left Warning** | **19 frames (18.4%)** |
| **Departure Right Warning** | **6 frames (5.8%)** |
| **No Lanes Detected** | **1 frame (1.0%)** |

All metrics and sample comparisons are saved in:
- Metrics: `results/evaluation_metrics.json`
- Plots: `results/summary_plots/lane_detection_comparison.png`
- Overlays: `results/overlays/`
- Intermediates: `results/intermediate/`

---

## ⚠️ Limitations & Future Work
- **Curved Roads**: Classical straight-line linear models struggle on sharp hairpin turns; higher-order splines or Deep Learning (e.g. U-Net / LaneNet) are required for non-linear geometry.
- **Occlusions & Weather**: Extreme rain, glare, or heavy traffic occlusion can disrupt edge detection.
