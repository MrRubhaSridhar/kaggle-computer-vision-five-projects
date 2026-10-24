# Five Applied Computer Vision Projects (Kaggle-Based Benchmark Suite)

A production-grade, student-friendly repository containing five complete Computer Vision and Deep Learning projects based on genuine Kaggle datasets. Implemented in Python 3.12 with PyTorch, TorchVision, Ultralytics YOLOv8, and OpenCV, accelerated by an NVIDIA GeForce RTX GPU.

---

## 📋 Table of Contents
1. [Repository Architecture](#-repository-architecture)
2. [Project Summaries & Kaggle Sources](#-project-summaries--kaggle-sources)
3. [Empirical Results Summary](#-empirical-results-summary)
4. [Environment Setup & Installation](#-environment-setup--installation)
5. [Dataset Download Guide](#-dataset-download-guide)
6. [Cross-Project Execution & Reproduction](#-cross-project-execution--reproduction)
7. [System Requirements & Limitations](#-system-requirements--limitations)
8. [License & Maintainers](#-license--maintainers)

---

## 🏛️ Repository Architecture

```
kaggle-computer-vision-five-projects/
├── README.md                                # Root repository guide (this file)
├── requirements.txt                         # Consolidated dependencies
├── LICENSE                                  # MIT Open Source License
├── .gitignore                               # Rigorous exclusions for data, weights, caches
├── docs/
│   └── final_project_summary.md             # Comprehensive cross-project engineering report
├── scripts/
│   ├── check_environment.py                 # System, CUDA, tool, and Kaggle auth verification
│   ├── download_datasets.sh                 # Sequential automated Kaggle dataset downloader
│   └── run_all_evaluations.sh               # Master script executing all 5 project evaluations
├── 01_lane_detection/                       # Project 1: Classical CV Lane & Departure Warning
│   ├── README.md, report.md, requirements.txt
│   ├── src/, scripts/, results/
├── 02_fruit_quality_inspection/             # Project 2: MobileNetV2 Fresh vs. Rotten Classifier
│   ├── README.md, report.md, requirements.txt
│   ├── src/, scripts/, results/
├── 03_traffic_sign_recognition/             # Project 3: 43-Class GTSRB Traffic Sign CNN
│   ├── README.md, report.md, requirements.txt
│   ├── src/, scripts/, results/
├── 04_robotic_object_detection/             # Project 4: YOLOv8n Construction Safety & ROS 2 Spec
│   ├── README.md, report.md, requirements.txt
│   ├── src/, scripts/, results/
└── 05_explainable_ai/                       # Project 5: CIFAR-10 Classifier with Custom Grad-CAM
    ├── README.md, report.md, requirements.txt
    ├── src/, scripts/, results/
```

---

## 🔍 Project Summaries & Kaggle Sources

### [Project 1: Road Lane Detection and Departure Warning](01_lane_detection/README.md)
- **Problem**: Detect road lane boundaries and estimate vehicle departure warnings in real time.
- **Approach**: Grayscale conversion, Gaussian smoothing, Canny edge detection, trapezoidal ROI masking, Probabilistic Hough Line Transform, slope-based clustering, linear extrapolation, and HUD departure warning.
- **Kaggle Dataset**: [`princekhunt19/road-lane-segmentation-imgs-and-labels`](https://www.kaggle.com/datasets/princekhunt19/road-lane-segmentation-imgs-and-labels)
- **Detailed Docs**: [01_lane_detection/README.md](01_lane_detection/README.md) | [Report](01_lane_detection/report.md)

### [Project 2: Fruit Quality Inspection](02_fruit_quality_inspection/README.md)
- **Problem**: Classify agricultural produce across 6 categories into fresh vs. rotten quality states.
- **Approach**: Transfer learning using MobileNetV2 with ImageNet pretrained feature representations, stratified validation partitioning, and real-time single-image/batch inference.
- **Kaggle Dataset**: [`sriramr/fruits-fresh-and-rotten-for-classification`](https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification)
- **Detailed Docs**: [02_fruit_quality_inspection/README.md](02_fruit_quality_inspection/README.md) | [Report](02_fruit_quality_inspection/report.md)

### [Project 3: Traffic Sign Detection and Recognition](03_traffic_sign_recognition/README.md)
- **Problem**: Classify 43 distinct road traffic signs under varying weather and illumination.
- **Approach**: ROI bounding-box cropping, photometric augmentation, 3-stage custom `TrafficSignCNN` with Batch Normalization and Dropout, and readable class label translation.
- **Kaggle Dataset**: [`meowmeowmeowmeowmeow/gtsrb-german-traffic-sign`](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
- **Detailed Docs**: [03_traffic_sign_recognition/README.md](03_traffic_sign_recognition/README.md) | [Report](03_traffic_sign_recognition/report.md)

### [Project 4: Real-Time Object Detection for Robotic Applications](04_robotic_object_detection/README.md)
- **Problem**: Detect personnel, construction equipment, and PPE compliance in real time for robot obstacle avoidance and safety interlocking.
- **Approach**: Annotation validation across 38,352 YOLO bounding boxes, fine-tuning YOLOv8n (anchor-free decoupled head), per-frame latency/FPS telemetry, and ROS 2 `vision_msgs` schema integration.
- **Kaggle Dataset**: [`snehilsanyal/construction-site-safety-image-dataset-roboflow`](https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow)
- **Detailed Docs**: [04_robotic_object_detection/README.md](04_robotic_object_detection/README.md) | [Report](04_robotic_object_detection/report.md)

### [Project 5: Explainable AI for Object Recognition](05_explainable_ai/README.md)
- **Problem**: Understand neural network predictions and visually verify influential image regions.
- **Approach**: Multi-stage CNN on CIFAR-10 with a from-scratch Grad-CAM engine capturing forward activations and backward target-class gradients, producing triplet diagnostic comparisons.
- **Kaggle Dataset**: [`ekaakurniawan/the-cifar10-dataset`](https://www.kaggle.com/datasets/ekaakurniawan/the-cifar10-dataset)
- **Detailed Docs**: [05_explainable_ai/README.md](05_explainable_ai/README.md) | [Report](05_explainable_ai/report.md)

---

## 📊 Empirical Results Summary

All metrics below reflect actual evaluation runs conducted on test partitions:

| # | Project | Test Split Scope | Primary Metrics | Latency / FPS | Key Outputs |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **01** | Road Lane Detection | 103 Test Images | Precision: **0.4583**, Recall: **0.0750**, F1: **0.1290**, IoU: **0.0689** | >120 FPS | [Comparison Plot](01_lane_detection/results/summary_plots/lane_detection_comparison.png) |
| **02** | Fruit Quality Inspection | 996 Test Images | Accuracy: **98.80%**, Macro F1: **0.9880**, Macro Prec: **0.9880** | ~14.5 ms | [Confusion Matrix](02_fruit_quality_inspection/results/confusion_matrix.png) |
| **03** | Traffic Sign Recognition | 1,983 Test Images | Accuracy: **98.23%**, Macro F1: **0.9739**, Macro Prec: **0.9782** | <2.0 ms | [Confusion Matrix](03_traffic_sign_recognition/results/confusion_matrix.png) |
| **04** | Robotic Object Detection | 82 Images (760 Objects) | mAP@0.5: **0.5319**, mAP@0.5:0.95: **0.2446**, Prec: **0.6791** | **6.57 ms** (152.2 FPS) | [Annotated Frames](04_robotic_object_detection/results/annotated_detections/) |
| **05** | Explainable AI (Grad-CAM) | 2,000 Test Images | Accuracy: **53.15%**, Macro F1: **0.5204**, Macro Prec: **0.5205** | ~5.5 ms | [Grad-CAM Grid](05_explainable_ai/results/gradcam/gradcam_comparison_grid.png) |

---

## ⚙️ Environment Setup & Installation

### 1. Prerequisites
- Linux OS (Ubuntu 22.04 or 24.04 recommended)
- Python 3.10 to 3.12
- NVIDIA GPU with driver supporting CUDA 11.8+ or 12.0+ (optional, CPU fallback supported)
- Kaggle CLI configured (`~/.kaggle/kaggle.json` or `~/.kaggle/credentials.json`)

### 2. Create Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Verify Environment
```bash
python scripts/check_environment.py
```

---

## 📥 Dataset Download Guide

Download individual datasets or all five sequentially using the provided script:

```bash
# Download all 5 datasets:
./scripts/download_datasets.sh all

# Or download individually:
./scripts/download_datasets.sh 1   # Road Lane Detection
./scripts/download_datasets.sh 2   # Fruit Quality
./scripts/download_datasets.sh 3   # Traffic Sign Recognition
./scripts/download_datasets.sh 4   # Robotic Object Detection
./scripts/download_datasets.sh 5   # CIFAR-10 Explainable AI
```

---

## 🚀 Cross-Project Execution & Reproduction

### Run Master Evaluation Suite
To execute evaluation across all five projects and reproduce the benchmark metrics:
```bash
./scripts/run_all_evaluations.sh
```

### Run Project 1: Lane Detection
```bash
# Evaluation on test set:
python 01_lane_detection/scripts/evaluate.py --max_samples 103

# Single-image inference:
python 01_lane_detection/scripts/train_or_run.py --image 01_lane_detection/data/raw/dataset/test/images/i789.jpg --save_intermediates

# Video inference:
python 01_lane_detection/scripts/train_or_run.py --video path/to/driving.mp4
```

### Run Project 2: Fruit Quality Inspection
```bash
# Train MobileNetV2:
python 02_fruit_quality_inspection/scripts/train.py --epochs 4 --train_limit 2000

# Evaluate on test set:
python 02_fruit_quality_inspection/scripts/evaluate.py --test_limit 1000

# Predict single image:
python 02_fruit_quality_inspection/scripts/predict.py --image 02_fruit_quality_inspection/data/raw/dataset/test/freshapples/rotated_by_15_Screen\ Shot\ 2018-06-08\ at\ 5.24.26\ PM.png
```

### Run Project 3: Traffic Sign Recognition
```bash
# Train TrafficSignCNN:
python 03_traffic_sign_recognition/scripts/train.py --epochs 5 --train_limit 10000

# Evaluate on test set:
python 03_traffic_sign_recognition/scripts/evaluate.py --test_limit 2000

# Predict single sign:
python 03_traffic_sign_recognition/scripts/predict.py --image 03_traffic_sign_recognition/data/raw/Test/00000.png
```

### Run Project 4: Robotic Object Detection
```bash
# Train YOLOv8n:
python 04_robotic_object_detection/scripts/train.py --epochs 3 --batch_size 16

# Evaluate on test split:
python 04_robotic_object_detection/scripts/evaluate.py

# Video inference demo:
python 04_robotic_object_detection/scripts/detect.py --video 04_robotic_object_detection/data/raw/source_files/source_files/hardhat.mp4
```

### Run Project 5: Explainable AI
```bash
# Train CNN classifier:
python 05_explainable_ai/scripts/train.py --epochs 5 --train_limit 10000

# Evaluate on test set:
python 05_explainable_ai/scripts/evaluate.py --test_limit 2000

# Generate Grad-CAM heatmaps:
python 05_explainable_ai/scripts/explain.py
```

---

## ⚠️ System Requirements & Limitations

1. **Hardware Constraints**: Heavy training runs (full 50,000 CIFAR-10 images or 39,209 GTSRB images) benefit from GPU acceleration. Default limits (e.g., 2,000–10,000 samples) ensure rapid reproducibility in resource-conscious environments while preserving full test splits for evaluation.
2. **Classical Lane Detection**: Straight-line Hough models assume standard highway corridors. Extreme curved roads require non-linear spline fitting or deep segmentation networks (e.g., U-Net).
3. **ROS 2 Integration Scope**: The robotic detector (Project 4) provides normalized bounding box coordinates and architectural schemas matching ROS 2 `vision_msgs/msg/Detection2DArray`. Physical hardware connection was not performed or claimed.
4. **Grad-CAM Interpretation**: Gradient-weighted heatmaps indicate feature correlations with output logits, not causal proof.

---

## 📜 License
This repository is licensed under the [MIT License](LICENSE). Datasets remain under their respective Kaggle source licenses (CC0, CC BY 4.0, or Open Source).
