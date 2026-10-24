# Project 2: Fruit Quality Inspection — Project Report

## 1. Problem Statement
Automated visual inspection in agricultural logistics, food packaging, and smart retail requires rapid, reliable discrimination between fresh and spoiled produce. Manual sorting is labor-intensive, subjective, and prone to human error. An automated computer vision system must classify fruit species and evaluate quality status (Fresh vs. Rotten) with high confidence.

## 2. Objectives
- Ingest and structure the multi-class Kaggle fruit quality dataset.
- Implement stratified train/validation/test data splits preserving class distributions.
- Deploy a transfer-learning classifier using MobileNetV2 with ImageNet pretrained feature representations.
- Train and validate the model with checkpointing and loss/accuracy logging.
- Evaluate test set performance using Accuracy, Macro Precision, Macro Recall, Macro F1-score, and a complete Confusion Matrix.
- Provide single-image and batch inference utilities with confidence scores and annotated visual overlays.

## 3. Dataset & Dataset Source
- **Dataset Title**: Fruits fresh and rotten for classification
- **Kaggle Identifier**: [`sriramr/fruits-fresh-and-rotten-for-classification`](https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification)
- **License**: Kaggle Public Dataset
- **Total Images in Raw Dataset**: 13,599 images across 6 classes

### Raw Class Distribution
| Class Name | Quality | Raw Train Count | Raw Test Count | Total |
| :--- | :--- | :--- | :--- | :--- |
| `freshapples` | Fresh | 1,693 | 395 | 2,088 |
| `freshbanana` | Fresh | 1,581 | 381 | 1,962 |
| `freshoranges` | Fresh | 1,466 | 388 | 1,854 |
| `rottenapples` | Rotten | 2,342 | 601 | 2,943 |
| `rottenbanana` | Rotten | 2,224 | 530 | 2,754 |
| `rottenoranges` | Rotten | 1,595 | 403 | 1,998 |
| **Total** | | **10,901** | **2,698** | **13,599** |

### Experimental Split Used
To balance rigorous benchmarking with rapid reproducible training on local hardware:
- **Train Set**: 1,704 images (stratified across 6 classes)
- **Validation Set**: 294 images (stratified 15% split)
- **Test Set**: 996 images (stratified test evaluation)

## 4. Preprocessing & Augmentation
1. **Resolution Normalization**: Resized to 224 × 224 pixels.
2. **Data Augmentation (Train)**:
   - Random Horizontal Flip ($p = 0.5$).
   - Random Affine Rotation ($\pm 15^\circ$).
   - Color Jitter (Brightness $\pm 0.15$, Contrast $\pm 0.15$).
3. **Channel Normalization**:
   $$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$

## 5. Methodology & Model Architecture
### MobileNetV2 Backbone
- Employs inverted residual blocks with linear bottlenecks, providing high representational capacity with minimal parameter footprint (3.4M parameters).
- **Custom Classification Head**:
  - Global Average Pooling (1280 channels)
  - Dropout ($p = 0.3$)
  - Fully-Connected Linear layer (1280 $\to$ 256)
  - Batch Normalization & ReLU activation
  - Dropout ($p = 0.2$)
  - Final Linear projection (256 $\to$ 6 classes)

## 6. Tools and Libraries
- Python 3.12.3
- PyTorch 2.14.0+cu130 (GPU acceleration on NVIDIA RTX 3050 Laptop GPU)
- TorchVision 0.29.0
- OpenCV 5.0.0 (image overlay rendering)
- Matplotlib 3.11.2 & Seaborn 0.13.2 (confusion matrix, distributions, training curves)
- Scikit-Learn 1.9.1 (classification metrics)

## 7. Experimental Configuration
- **Epochs**: 4
- **Batch Size**: 32
- **Optimizer**: Adam ($\text{lr} = 3 \times 10^{-4}$, weight decay $= 10^{-4}$)
- **Learning Rate Scheduler**: CosineAnnealingLR ($T_{\max} = 4$)
- **Loss Function**: Cross-Entropy Loss

## 8. Actual Evaluation Metrics (996 Test Samples)

| Metric | Measured Score |
| :--- | :--- |
| **Accuracy** | **98.80%** |
| **Macro Precision** | **0.9880** |
| **Macro Recall** | **0.9880** |
| **Macro F1-Score** | **0.9880** |

### Class-Wise Test Performance
| Class Name | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| `freshapples` | 0.9939 | 0.9879 | 0.9909 | 166 |
| `freshbanana` | 0.9880 | 0.9940 | 0.9910 | 166 |
| `freshoranges` | 0.9699 | 0.9699 | 0.9699 | 166 |
| `rottenapples` | 0.9940 | 0.9940 | 0.9940 | 166 |
| `rottenbanana` | 0.9940 | 0.9940 | 0.9940 | 166 |
| `rottenoranges` | 0.9880 | 0.9880 | 0.9880 | 166 |

## 9. Sample Inputs and Outputs
- **Metrics File**: `02_fruit_quality_inspection/results/metrics.json`
- **Confusion Matrix**: `02_fruit_quality_inspection/results/confusion_matrix.png`
- **Training Curves**: `02_fruit_quality_inspection/results/training_curves.png`
- **Class Distribution Plot**: `02_fruit_quality_inspection/results/class_distribution.png`
- **Exploration Samples**: `02_fruit_quality_inspection/results/dataset_samples.png`
- **Correct Prediction Grid**: `02_fruit_quality_inspection/results/predictions/correct_samples.png`
- **Misclassified Prediction Grid**: `02_fruit_quality_inspection/results/predictions/misclassified_samples.png`

## 10. Challenges & Limitations
1. **Subtle Surface Blemishes**: Slight bruising on fresh apples can occasionally be misclassified as early rot.
2. **Background Illumination Variance**: Laboratory plain-white backgrounds vs. cluttered conveyor belts require robust domain generalization.
3. **Class Overlap**: Rotten oranges and rotten apples with fungal bloom share visual textures (green/white mold patterns) requiring high-resolution color analysis.

## 11. Conclusion
The fine-tuned MobileNetV2 architecture achieves exceptional classification accuracy (98.80%) across all 6 fruit categories. Its compact model footprint and sub-10ms inference latency make it suitable for deployment on industrial edge sorters and automated packaging inspection systems.
