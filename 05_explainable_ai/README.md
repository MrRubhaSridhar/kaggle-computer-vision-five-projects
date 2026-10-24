# Project 5: Explainable AI for Object Recognition (Grad-CAM)

An Explainable AI (XAI) system featuring a custom Convolutional Neural Network trained on CIFAR-10, integrated with a ground-up Gradient-weighted Class Activation Mapping (Grad-CAM) engine. It visualizes receptive field influence for both correct classifications and failure modes.

---

## 📌 Dataset Information
- **Dataset Title**: The CIFAR-10 Dataset
- **Kaggle Identifier**: [ekaakurniawan/the-cifar10-dataset](https://www.kaggle.com/datasets/ekaakurniawan/the-cifar10-dataset)
- **Classes**: 10 categories (`airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`).
- **Structure**: 60,000 32×32 color images packed in standard Python batch files.

### Download Command
```bash
kaggle datasets download -d ekaakurniawan/the-cifar10-dataset -p 05_explainable_ai/data/raw --unzip
```

---

## 🔬 Explainable AI Architecture
1. **CNN Backbone**: 3 dual-conv residual-style blocks with Batch Normalization, MaxPool, and Dropout.
2. **Target Conv Layer**: The final 128-channel convolution block prior to spatial pooling (`model.block3[3]`).
3. **Forward & Backward Hooks**: Intercepts feature activations $A^k$ and backpropagated target-class gradients $\frac{\partial Y^c}{\partial A^k}$.
4. **Grad-CAM Calculation**:
   $$\alpha_k^c = \frac{1}{Z}\sum_i \sum_j \frac{\partial Y^c}{\partial A_{i,j}^k}, \quad L^c_{\text{Grad-CAM}} = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
5. **Overlay Generation**: Heatmap upsampled, normalized, colored with JET colormap, and blended onto the input image.

---

## 🚀 Reproduction Commands

### 1. Train CNN Classifier
```bash
python 05_explainable_ai/scripts/train.py --epochs 5 --train_limit 10000 --batch_size 64
```

### 2. Evaluate on Test Set
```bash
python 05_explainable_ai/scripts/evaluate.py --test_limit 2000
```

### 3. Generate Grad-CAM Heatmaps & Diagnostic Grids
```bash
python 05_explainable_ai/scripts/explain.py
```

---

## 📊 Measured Test Performance (2,000 Samples)

| Metric | Measured Value |
| :--- | :--- |
| **Accuracy** | **53.15%** |
| **Macro Precision** | **0.5205** |
| **Macro Recall** | **0.5350** |
| **Macro F1-Score** | **0.5204** |

### Output Visualisations
- **Master Comparison Grid**: `results/gradcam/gradcam_comparison_grid.png`
- **Confusion Matrix**: `results/confusion_matrix.png`
- **Training Curves**: `results/training_curves.png`
- **Exploration Samples**: `results/dataset_samples.png`

---

## ⚠️ Important Note on Grad-CAM Interpretation
Grad-CAM indicates regions correlated with high gradient sensitivity for the target logit. It highlights *where* the model is looking, but does not prove *causal sufficiency*. Misclassifications often reveal background co-occurrence biases or fine-grained feature confusion (e.g., cat vs. dog ears).
