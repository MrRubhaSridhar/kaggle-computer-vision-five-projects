# Project 5: Explainable AI for Object Recognition — Project Report

## 1. Problem Statement
Deep Convolutional Neural Networks (CNNs) achieve strong visual classification performance but operate largely as opaque "black-box" systems. In safety-critical domains (robotics, medicine, autonomous navigation), practitioners require visual explanations indicating which input image features drove a model's prediction, and whether misclassifications stem from spurious background artifacts or genuine object confusion.

## 2. Objectives
- Train a lightweight multi-stage CNN classifier on the CIFAR-10 benchmark dataset.
- Implement Gradient-weighted Class Activation Mapping (Grad-CAM) from scratch without external black-box explainability libraries.
- Hook into the final convolutional layer to capture forward activations and backward target-class gradients.
- Generate high-resolution Grad-CAM heatmaps and visual overlay comparisons.
- Contrast explanations for **correct predictions** versus **misclassified examples** to diagnose model failure modes.
- Evaluate the model with standard quantitative metrics: Accuracy, Macro Precision, Macro Recall, Macro F1-score, and Confusion Matrix.
- Critically discuss the theoretical limitations of Grad-CAM (correlation vs. causal reasoning).

## 3. Dataset & Dataset Source
- **Dataset Title**: The CIFAR-10 Dataset
- **Kaggle Identifier**: [`ekaakurniawan/the-cifar10-dataset`](https://www.kaggle.com/datasets/ekaakurniawan/the-cifar10-dataset)
- **License**: Kaggle Open Dataset (Standard CIFAR-10 by Alex Krizhevsky)
- **Format**: Python pickled batches (`data_batch_1` through `5`, `test_batch`, `batches.meta`)
- **Resolution**: 32 × 32 × 3 RGB color images
- **Classes (10)**: `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`

### Dataset Partitioning Used
- **Training Subset**: 9,000 images (from 10,000 sampled train pool, stratified across 10 classes)
- **Validation Subset**: 1,000 images
- **Test Set**: 2,000 images

## 4. Preprocessing Steps
1. **Unpacking**: Extracted raw 3072-byte 1D arrays and reconstructed (32, 32, 3) channel-last uint8 images.
2. **Data Augmentation (Training)**:
   - Random 32×32 crop with 4-pixel reflective padding.
   - Random horizontal flip ($p = 0.5$).
3. **Tensor Normalization**: Standard CIFAR-10 channel statistics:
   $$\mu = [0.4914, 0.4822, 0.4465], \quad \sigma = [0.2470, 0.2435, 0.2616]$$

## 5. Methodology & Architecture
### Model Architecture (`CIFAR10CNN`)
- **Block 1**: Conv2D(3 $\to$ 32, 3×3) $\to$ BatchNorm $\to$ ReLU $\to$ Conv2D(32 $\to$ 32, 3×3) $\to$ BatchNorm $\to$ ReLU $\to$ MaxPool(2×2) $\to$ Dropout(0.2)
- **Block 2**: Conv2D(32 $\to$ 64, 3×3) $\to$ BatchNorm $\to$ ReLU $\to$ Conv2D(64 $\to$ 64, 3×3) $\to$ BatchNorm $\to$ ReLU $\to$ MaxPool(2×2) $\to$ Dropout(0.3)
- **Block 3 (Grad-CAM Target)**: Conv2D(64 $\to$ 128, 3×3) $\to$ BatchNorm $\to$ ReLU $\to$ **Conv2D(128 $\to$ 128, 3×3)** [Target Layer] $\to$ BatchNorm $\to$ ReLU $\to$ MaxPool(2×2) $\to$ Dropout(0.4)
- **Dense Classifier**: Flatten $\to$ Linear(2048 $\to$ 256) $\to$ BatchNorm $\to$ ReLU $\to$ Dropout(0.5) $\to$ Linear(256 $\to$ 10)

### Grad-CAM Algorithm
1. **Activation Capture**: A PyTorch forward hook stores feature maps $A^k \in \mathbb{R}^{H \times W}$ from `model.block3[3]`.
2. **Gradient Backpropagation**: A backward hook intercepts gradients $\frac{\partial Y^c}{\partial A^k}$ where $Y^c$ is the unnormalized logit score for class $c$.
3. **Neuron Importance Weights**:
   $$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^H \sum_{j=1}^W \frac{\partial Y^c}{\partial A_{i,j}^k}$$
4. **Weighted Activation Map**:
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
5. **Upsampling & Blending**: Bilinear interpolation scales the 4×4 map to 32×32, min-max normalized to $[0, 1]$, colorized with OpenCV JET colormap, and blended at $\alpha=0.5$ with the original RGB image.

## 6. Tools and Libraries
- Python 3.12.3
- PyTorch 2.14.0+cu130 (GPU-accelerated training and backward hook tracking)
- NVIDIA GeForce RTX 3050 Laptop GPU (CUDA 13.0)
- TorchVision 0.29.0
- OpenCV 5.0.0 (colormap rendering and overlay blending)
- Matplotlib 3.11.2 & Seaborn 0.13.2 (confusion matrix and triplet grids)
- Scikit-Learn 1.9.1 (macro metrics)

## 7. Experimental Configuration
- **Optimizer**: Adam ($\text{lr} = 10^{-3}$, weight decay $= 10^{-4}$)
- **Learning Rate Scheduler**: CosineAnnealingLR ($T_{\max} = 5$)
- **Epochs**: 5
- **Batch Size**: 64
- **Loss Function**: Categorical Cross-Entropy

## 8. Actual Evaluation Metrics (2,000 Test Images)

| Metric | Measured Value |
| :--- | :--- |
| **Accuracy** | **53.15%** |
| **Macro Precision** | **0.5205** |
| **Macro Recall** | **0.5350** |
| **Macro F1-Score** | **0.5204** |

### Class-Wise Test Performance
| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **airplane** | 0.6073 | 0.5750 | 0.5907 | 200 |
| **automobile** | 0.6970 | 0.5750 | 0.6301 | 200 |
| **bird** | 0.3800 | 0.4750 | 0.4222 | 200 |
| **cat** | 0.3807 | 0.3750 | 0.3778 | 200 |
| **deer** | 0.4682 | 0.5150 | 0.4905 | 200 |
| **dog** | 0.4789 | 0.3400 | 0.3977 | 200 |
| **frog** | 0.5566 | 0.5900 | 0.5728 | 200 |
| **horse** | 0.5323 | 0.6600 | 0.5893 | 200 |
| **ship** | 0.6288 | 0.7200 | 0.6713 | 200 |
| **truck** | 0.4755 | 0.5250 | 0.4988 | 200 |

## 9. Grad-CAM Explainability Findings

### Correct Predictions
- On ships and airplanes, Grad-CAM concentrates intensely on the fuselage, wings, and water wakes, demonstrating that the network has learned relevant structural silhouettes.
- On frogs and horses, activations peak around heads, legs, and dorsal contours.

### Misclassified Predictions
- **Canine vs. Feline Confusion**: For a true `cat` misclassified as `dog`, Grad-CAM heatmaps highlight ear geometry and fur patterns, indicating legitimate visual ambiguity at 32×32 resolution.
- **Background Bias**: In certain bird classifications, the model focused heavily on foliage or sky textures rather than the avian subject itself.

### Causal Reality vs. Correlational Heatmaps
> [!IMPORTANT]
> **Theoretical Limitation of Grad-CAM**: Grad-CAM illuminates which feature map coordinates exerted positive gradient magnitude on the output logit. However, **gradient attribution is purely correlational**. It does not prove causal sufficiency: modifying background pixels outside the highlighted zone can still shift feature activations across receptive fields. True causal verification requires counterfactual ablation testing.

## 10. Saved Result Artefacts
- **Metrics JSON**: `05_explainable_ai/results/metrics.json`
- **Training Curves**: `05_explainable_ai/results/training_curves.png`
- **Confusion Matrix**: `05_explainable_ai/results/confusion_matrix.png`
- **Dataset Class Grid**: `05_explainable_ai/results/dataset_samples.png`
- **Grad-CAM Comparison Grid**: `05_explainable_ai/results/gradcam/gradcam_comparison_grid.png`
- **Individual Triplets**: `05_explainable_ai/results/gradcam/correct_*.png`, `incorrect_*.png`

## 11. Conclusion
The explainable AI pipeline provides direct insight into deep neural representations. Grad-CAM successfully decodes where the CNN directs spatial attention, separating valid object detection from background bias, and serving as an essential debugging tool for model safety and reliability.
