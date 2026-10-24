# Project 4: Real-Time Object Detection for Robotic Applications

A real-time Computer Vision subsystem deploying an anchor-free YOLOv8n detector trained on the Construction Site Safety Roboflow dataset. Designed for robotic integration, featuring per-frame latency/FPS telemetry, video stream processing, and ROS 2 `vision_msgs` schema compatibility.

---

## 📌 Dataset Information
- **Dataset Title**: Construction Site Safety Image Dataset Roboflow
- **Kaggle Identifier**: [snehilsanyal/construction-site-safety-image-dataset-roboflow](https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow)
- **License**: CC BY 4.0
- **Volume**: 2,801 total images (2,605 train, 114 valid, 82 test) with 38,352 YOLO-format bounding box annotations.
- **Classes (10)**: `Hardhat`, `Mask`, `NO-Hardhat`, `NO-Mask`, `NO-Safety Vest`, `Person`, `Safety Cone`, `Safety Vest`, `machinery`, `vehicle`.

### Download Command
```bash
kaggle datasets download -d snehilsanyal/construction-site-safety-image-dataset-roboflow -p 04_robotic_object_detection/data/raw --unzip
```

---

## 🛠️ Robotic Vision Architecture
1. **Lightweight Detector**: YOLOv8n (3.0M parameters, 8.2 GFLOPs).
2. **Anchor-Free Decoupled Head**: Direct regression of bounding box coordinates without anchor priors.
3. **Telemetry Overlay**: Per-frame inference latency (ms) and instantaneous FPS displayed in real-time.
4. **ROS 2 Interface Ready**: Generates normalized centroid coordinates and class hypotheses conforming to `vision_msgs/msg/Detection2DArray`.

---

## 🚀 Reproduction Commands

### 1. Train YOLOv8n Detector
```bash
python 04_robotic_object_detection/scripts/train.py --epochs 3 --batch_size 16 --img_size 640
```

### 2. Evaluate on Test Split
```bash
python 04_robotic_object_detection/scripts/evaluate.py
```

### 3. Run Inference on a Single Image
```bash
python 04_robotic_object_detection/scripts/detect.py --image path/to/image.jpg
```

### 4. Run Inference on a Video Stream
```bash
python 04_robotic_object_detection/scripts/detect.py --video 04_robotic_object_detection/data/raw/source_files/source_files/hardhat.mp4
```

---

## 📊 Measured Test Performance (82 Test Images, 760 Objects)

| Metric | Measured Score |
| :--- | :--- |
| **mAP@0.5** | **0.5319** |
| **mAP@0.5:0.95** | **0.2446** |
| **Precision** | **0.6791** |
| **Recall** | **0.5119** |
| **Inference Latency (GPU)** | **6.57 ms / frame** |
| **Inference Throughput** | **152.2 FPS** |

### Output Visualisations
- **Metrics JSON**: `results/metrics.json`
- **Dataset Exploration**: `results/dataset_exploration.png`
- **Annotated Detections**: `results/annotated_detections/`
- **Annotated Video**: `results/sample_clips/detected_hardhat.mp4`

---

## 🤖 ROS 2 Integration Architecture
The detector extracts bounding box centers normalized to the camera frame:
```python
# ROS 2 Publisher schema: vision_msgs/msg/Detection2DArray
# Topic: /robot/vision/safety_detections
{
  "class_name": "Hardhat",
  "confidence": 0.895,
  "bbox_xyxy": [120, 45, 185, 110],
  "robotic_center_norm": [0.238, 0.121]
}
```
*Note: Evaluated as a standalone vision pipeline on local GPU; live physical robot deployment has not been claimed.*
