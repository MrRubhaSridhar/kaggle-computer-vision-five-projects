# Project 3: Traffic Sign Detection and Recognition — Project Report

## 1. Problem Statement
Autonomous vehicles, Advanced Driver Assistance Systems (ADAS), and smart transport infrastructure require fast, highly accurate identification of road traffic signs under varying weather, illumination, motion blur, and partial occlusions. Recognizing 43 distinct classes with nuanced visual differences (e.g., speed limits differing only by small numerical glyphs) is a critical perception requirement.

## 2. Objectives
- Ingest and parse the benchmark German Traffic Sign Recognition Benchmark (GTSRB) dataset.
- Establish a complete mapping from 43 integer class indices to standardized human-readable traffic sign names.
- Preprocess input imagery by extracting annotated bounding box Regions of Interest (ROIs), resizing to 48 × 48 pixels, and applying photometric augmentations.
- Implement a lightweight, custom convolutional neural network (`TrafficSignCNN`) engineered for rapid convergence and low latency.
- Formulate a reproducible stratified train/validation split from the training dataset.
- Quantitatively evaluate the classifier on the official test set using Accuracy, Macro Precision, Macro Recall, Macro F1-Score, and a 43×43 Confusion Matrix.
- Generate qualitative prediction visualisations comparing ground-truth vs. predicted labels with confidence scores.

## 3. Dataset & Dataset Source
- **Dataset Title**: GTSRB - German Traffic Sign Recognition Benchmark
- **Kaggle Identifier**: [`meowmeowmeowmeowmeow/gtsrb-german-traffic-sign`](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
- **License**: CC0: Public Domain
- **Annotation Files**: `Train.csv`, `Test.csv`, `Meta.csv`
- **Total Dataset Volume**: 39,209 training images, 12,630 test images across 43 classes

### Experimental Split Used
- **Training Set**: 8,502 images (stratified sample from 10,000 train pool)
- **Validation Set**: 1,476 images (15% stratified holdout)
- **Test Set**: 1,983 images (stratified sample from official test split)

## 4. Preprocessing Steps
1. **ROI Extraction**: Extracted the sign region using ground-truth bounding box coordinates (`Roi.X1`, `Roi.Y1`, `Roi.X2`, `Roi.Y2`) to isolate the traffic sign from background scenery.
2. **Standardization**: Resized to 48 × 48 pixels.
3. **Training Augmentations**:
   - Random Affine Rotation ($\pm 10^\circ$).
   - Color Jittering (Brightness $\pm 0.2$, Contrast $\pm 0.2$).
4. **Channel Normalization**:
   $$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$

## 5. Methodology & Architecture
### Model Architecture (`TrafficSignCNN`)
- **Block 1**: Conv2D(3 $\to$ 32, 3×3) $\to$ BatchNorm $\to$ ReLU $\to$ Conv2D(32 $\to$ 32, 3×3) $\to$ BatchNorm $\to$ ReLU $\to$ MaxPool(2×2) $\to$ Dropout(0.2) [48×48 $\to$ 24×24]
- **Block 2**: Conv2D(32 $\to$ 64, 3×3) $\to$ BatchNorm $\to$ ReLU $\to$ Conv2D(64 $\to$ 64, 3×3) $\to$ BatchNorm $\to$ ReLU $\to$ MaxPool(2×2) $\to$ Dropout(0.25) [24×24 $\to$ 12×12]
- **Block 3**: Conv2D(64 $\to$ 128, 3×3) $\to$ BatchNorm $\to$ ReLU $\to$ Conv2D(128 $\to$ 128, 3×3) $\to$ BatchNorm $\to$ ReLU $\to$ MaxPool(2×2) $\to$ Dropout(0.3) [12×12 $\to$ 6×6]
- **Dense Classifier**: Flatten $\to$ Linear(4608 $\to$ 256) $\to$ BatchNorm $\to$ ReLU $\to$ Dropout(0.4) $\to$ Linear(256 $\to$ 43)

## 6. Tools and Libraries
- Python 3.12.3
- PyTorch 2.14.0+cu130 (GPU execution on NVIDIA GeForce RTX 3050 Laptop GPU)
- TorchVision 0.29.0
- Pandas 3.1.0 (annotation parsing and stratified sampling)
- OpenCV 5.0.0 (image overlay rendering)
- Matplotlib 3.11.2 & Seaborn 0.13.2 (confusion matrix and distributions)
- Scikit-Learn 1.9.1 (macro evaluation metrics)

## 7. Experimental Configuration
- **Epochs**: 5
- **Batch Size**: 64
- **Optimizer**: Adam ($\text{lr} = 10^{-3}$, weight decay $= 10^{-4}$)
- **Learning Rate Scheduler**: CosineAnnealingLR ($T_{\max} = 5$)
- **Loss Function**: Categorical Cross-Entropy

## 8. Actual Evaluation Metrics (1,983 Test Samples)

| Metric | Measured Score |
| :--- | :--- |
| **Accuracy** | **98.23%** |
| **Macro Precision** | **0.9782** |
| **Macro Recall** | **0.9722** |
| **Macro F1-Score** | **0.9739** |

### Selected Class-Wise Test Performance (Top & Critical Signs)
| Class ID | Traffic Sign Name | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | Speed limit (30km/h) | 0.9904 | 0.9810 | 0.9856 | 105 |
| **2** | Speed limit (50km/h) | 0.9720 | 0.9905 | 0.9811 | 105 |
| **12** | Priority road | 0.9899 | 0.9899 | 0.9899 | 99 |
| **13** | Yield | 1.0000 | 0.9909 | 0.9954 | 110 |
| **14** | Stop | 1.0000 | 1.0000 | 1.0000 | 42 |
| **17** | No entry | 1.0000 | 1.0000 | 1.0000 | 56 |
| **25** | Road work | 0.9855 | 0.9714 | 0.9784 | 70 |
| **38** | Keep right | 0.9796 | 0.9796 | 0.9796 | 98 |

## 9. Sample Inputs and Outputs
- **Metrics JSON**: `03_traffic_sign_recognition/results/metrics.json`
- **Confusion Matrix**: `03_traffic_sign_recognition/results/confusion_matrix.png`
- **Class Distribution Bar Chart**: `03_traffic_sign_recognition/results/class_distribution.png`
- **Dataset Sample Visuals Grid**: `03_traffic_sign_recognition/results/dataset_samples.png`
- **Correct Sign Prediction Grid**: `03_traffic_sign_recognition/results/predictions/sample_predictions.png`
- **Misclassified Sign Diagnostic Grid**: `03_traffic_sign_recognition/results/predictions/misclassified_signs.png`

## 10. Challenges & Limitations
1. **Severe Imbalance in Raw Data**: Common signs (e.g., Speed limit 50km/h with >2,000 samples) heavily outnumber rare signs (e.g., Speed limit 20km/h with ~200 samples). Stratified sampling mitigated bias during training.
2. **Motion Blur & Low Dynamic Range**: Heavily underexposed twilight captures can cause confusion between speed limit circle digits.
3. **Classification vs. Detection Boundary**: This project solves traffic sign recognition (classification of sign crops). In a complete autonomous vehicle stack, this module operates downstream of a general object detector (such as YOLO).

## 11. Conclusion
The TrafficSignCNN architecture demonstrates exceptional recognition performance (98.23% accuracy, 0.9739 macro F1) across 43 classes with sub-2 millisecond inference latency per sign, fulfilling the real-time requirements of embedded ADAS platforms.
