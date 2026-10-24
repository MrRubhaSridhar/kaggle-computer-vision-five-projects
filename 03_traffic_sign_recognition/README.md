# Project 3: Traffic Sign Detection and Recognition (GTSRB)

A deep learning image classification system for recognizing 43 distinct road traffic signs based on the German Traffic Sign Recognition Benchmark (GTSRB). Evaluated on genuine test split data with readable class mappings, confusion matrix, and prediction visualisations.

---

## 📌 Dataset Information
- **Dataset Title**: GTSRB - German Traffic Sign Recognition Benchmark
- **Kaggle Source**: [meowmeowmeowmeowmeow/gtsrb-german-traffic-sign](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
- **Classes**: 43 traffic sign categories (Speed limits, Warnings, Mandatory, Prohibitory).
- **Volume**: 39,209 training images, 12,630 test images.
- **Annotations**: `Train.csv` and `Test.csv` with bounding box coordinates (`Roi.X1`, `Roi.Y1`, `Roi.X2`, `Roi.Y2`) and integer `ClassId`.

### Download Command
```bash
kaggle datasets download -d meowmeowmeowmeowmeow/gtsrb-german-traffic-sign -p 03_traffic_sign_recognition/data/raw --unzip
```

---

## 🛠️ Pipeline Architecture
1. **Bounding Box Extraction**: Crops the traffic sign from background scenery using ground-truth annotations.
2. **Preprocessing**: Resizes to 48×48 pixels and normalizes via ImageNet statistics.
3. **TrafficSignCNN Backbone**: 3 dual-convolution blocks (32, 64, 128 channels) with Batch Normalization, Max Pooling, and Dropout.
4. **Dense Classifier**: Fully connected projection with 43-class softmax output.
5. **Human-Readable Class Mapping**: Translates numerical IDs into clear traffic sign names.

---

## 🚀 Reproduction Commands

### 1. Train TrafficSignCNN
```bash
python 03_traffic_sign_recognition/scripts/train.py --epochs 5 --train_limit 10000 --batch_size 64
```

### 2. Evaluate on Test Split
```bash
python 03_traffic_sign_recognition/scripts/evaluate.py --test_limit 2000
```

### 3. Predict Single Traffic Sign
```bash
python 03_traffic_sign_recognition/scripts/predict.py --image path/to/sign.png
```

### 4. Predict Batch Directory
```bash
python 03_traffic_sign_recognition/scripts/predict.py --input_dir path/to/folder
```

---

## 📊 Measured Test Performance (1,983 Test Samples)

| Metric | Measured Score |
| :--- | :--- |
| **Accuracy** | **98.23%** |
| **Macro Precision** | **0.9782** |
| **Macro Recall** | **0.9722** |
| **Macro F1-Score** | **0.9739** |

### Output Visualisations
- **Confusion Matrix**: `results/confusion_matrix.png`
- **Class Distribution**: `results/class_distribution.png`
- **Dataset Sample Grid**: `results/dataset_samples.png`
- **Sample Predictions**: `results/predictions/sample_predictions.png`
- **Misclassified Analysis**: `results/predictions/misclassified_signs.png`
- **Metrics JSON**: `results/metrics.json`
