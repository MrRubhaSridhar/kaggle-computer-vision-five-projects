# Project 2: Fruit Quality Inspection (Fresh vs. Rotten)

A transfer-learning visual inspection system using MobileNetV2 to classify agricultural produce across 6 fresh and rotten fruit categories. Evaluated on genuine Kaggle test data with class-wise metrics, confusion matrix, and visual prediction overlays.

---

## 📌 Dataset Information
- **Dataset Title**: Fruits fresh and rotten for classification
- **Kaggle Source**: [sriramr/fruits-fresh-and-rotten-for-classification](https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification)
- **Categories (6)**: `freshapples`, `freshbanana`, `freshoranges`, `rottenapples`, `rottenbanana`, `rottenoranges`.
- **Dataset Volume**: 13,599 total RGB images (10,901 train, 2,698 test).

### Download Command
```bash
kaggle datasets download -d sriramr/fruits-fresh-and-rotten-for-classification -p 02_fruit_quality_inspection/data/raw --unzip
```

---

## 🛠️ Model Architecture & Methodology
- **Backbone**: MobileNetV2 with ImageNet pretrained feature representations.
- **Classification Head**: Dropout (0.3) $\to$ Linear (1280 $\to$ 256) $\to$ BatchNorm $\to$ ReLU $\to$ Dropout (0.2) $\to$ Linear (256 $\to$ 6).
- **Stratified Partitioning**: 15% validation split extracted from the training set with preserved class distributions.
- **Augmentation**: Random horizontal flips, affine rotations ($\pm 15^\circ$), and color jittering.

---

## 🚀 Reproduction Commands

### 1. Train Model
```bash
python 02_fruit_quality_inspection/scripts/train.py --epochs 4 --train_limit 2000 --batch_size 32
```

### 2. Evaluate on Test Split
```bash
python 02_fruit_quality_inspection/scripts/evaluate.py --test_limit 1000
```

### 3. Predict Single Image
```bash
python 02_fruit_quality_inspection/scripts/predict.py --image path/to/fruit.jpg
```

### 4. Predict Batch Directory
```bash
python 02_fruit_quality_inspection/scripts/predict.py --input_dir path/to/folder
```

---

## 📊 Measured Test Performance (996 Samples)

| Metric | Measured Score |
| :--- | :--- |
| **Accuracy** | **98.80%** |
| **Macro Precision** | **0.9880** |
| **Macro Recall** | **0.9880** |
| **Macro F1-Score** | **0.9880** |

### Output Visualisations
- **Confusion Matrix**: `results/confusion_matrix.png`
- **Class Distribution**: `results/class_distribution.png`
- **Dataset Samples**: `results/dataset_samples.png`
- **Correct Predictions**: `results/predictions/correct_samples.png`
- **Misclassified Predictions**: `results/predictions/misclassified_samples.png`
- **Metrics JSON**: `results/metrics.json`
