# Breast Cancer Classification – BUSI Ultrasound Dataset (DenseNet201)

Breast cancer classification using the **BUSI (Breast Ultrasound Images)** dataset and a **pretrained DenseNet201** model. The project covers image preprocessing, data augmentation, transfer learning, model training, evaluation, confusion matrix analysis, multi-class ROC-AUC, and t-SNE feature visualization.

---

##  Dataset

- **Name:** BUSI (Breast Ultrasound Images)
- **Classes:** `benign`, `malignant`, `normal`
- **Expected structure:**

```
DATA_DIR/
    benign/
    malignant/
    normal/
```

---

##Pipeline

```
ImageFolder → Stratified 80/20 Split → DenseNet201 (Transfer Learning)
   → CrossEntropyLoss + Adam → Training / Validation
   → Confusion Matrix → Multi-class ROC / AUC
   → DenseNet201 Feature Extraction → t-SNE Visualization
```

**Key details:**
- Image size: `224 x 224`
- Batch size: `32`
- Epochs: `10`
- Learning rate: `0.0001`
- Optimizer: `Adam`
- Loss: `CrossEntropyLoss`
- Backbone: `DenseNet201` (ImageNet pretrained), final classifier replaced with a linear layer for 3 classes

> **Note:** Training accuracy and held-out validation accuracy are different quantities — validation accuracy is used for model evaluation.

---

## ⚙️ Requirements

```bash
pip install torch torchvision scikit-learn matplotlib seaborn numpy
```

---

##Usage

1. Update `DATA_DIR` in the script to point to your local BUSI dataset folder.
2. Run the script (e.g. in Google Colab or locally with a GPU):

```bash
python busi_densenet201.py
```

3. Results (model checkpoint, plots, reports, and `.npy` arrays) are saved to `RESULTS_DIR`.

---

## Results

### Multi-Class ROC Curve

![ROC Curve](results/busi_roc_curve.png)

The model achieves strong class-wise separability, with AUC scores of **0.97 (Benign)**, **0.96 (Malignant)**, and **1.00 (Normal)**.

### t-SNE Feature Embedding

![t-SNE Embedding](results/busi_tsne.png)

DenseNet201 deep features projected to 2D via t-SNE show clear separation between the *normal* class and the *benign/malignant* clusters, with benign and malignant partially overlapping — consistent with their closer visual similarity in ultrasound images.

### Other Generated Outputs
- `busi_confusion_matrix.png` — confusion matrix heatmap
- `busi_accuracy_curve.png` — training vs. validation accuracy per epoch
- `busi_loss_curve.png` — training vs. validation loss per epoch
- `classification_report.txt` — precision, recall, F1-score per class
- `busi_results.txt` — full experiment summary
- `best_busi_densenet201.pth` — best model checkpoint

---

## Repository Structure

```
breast-cancer-busi/
├── busi_densenet201.py        # Main training/evaluation script
├── results/
│   ├── busi_roc_curve.png
│   ├── busi_tsne.png
│   ├── busi_confusion_matrix.png
│   ├── busi_accuracy_curve.png
│   ├── busi_loss_curve.png
│   ├── classification_report.txt
│   └── busi_results.txt
└── README.md
```

---

## Evaluation Metrics

- **Confusion Matrix** — per-class prediction breakdown
- **Classification Report** — precision, recall, F1-score per class
- **Macro ROC-AUC** — averaged across all classes
- **t-SNE** — 2D visualization of learned DenseNet201 features

---

## Future Work

- Experiment with other backbones (EfficientNet, ResNet, ViT)
- Add k-fold cross-validation
- Apply Grad-CAM for model interpretability
- Address class imbalance with weighted loss or oversampling

---

## License

This project is open-source and available for research and educational purposes.
