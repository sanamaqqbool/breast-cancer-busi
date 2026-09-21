# Breast Cancer Classification Using BUSI Dataset

PyTorch DenseNet201 transfer-learning pipeline for the Breast Ultrasound Images (BUSI) dataset.

## Classes
- Benign
- Malignant
- Normal

## Method
- Resize: 224 x 224
- ImageNet normalization
- Training augmentation
- Stratified 80/20 train-validation split
- ImageNet-pretrained DenseNet201
- Adam, learning rate 0.0001
- Batch size 32
- 10 epochs
- Cross-entropy loss

## Evaluation
Accuracy, classification report, confusion matrix, multiclass one-vs-rest ROC-AUC, DenseNet201 feature extraction, and t-SNE.

## Dataset structure
```text
Untitled Folder/
├── benign/
├── malignant/
└── normal/
```

Change `DATA_DIR` in `busi_densenet201.py` to your local/Colab dataset path.

Do not upload the medical image dataset itself to GitHub.

## Run
```bash
pip install -r requirements.txt
python busi_densenet201.py
```

## Dataset reference
Al-Dhabyani, W., Gomaa, M., Khaled, H., & Fahmy, A. (2020). Dataset of breast ultrasound images. Data in Brief, 28, 104863.
https://doi.org/10.1016/j.dib.2019.104863
