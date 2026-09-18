# breast-cancer-busi
Breast cancer classification using the BUSI ultrasound image dataset and a pretrained DenseNet201 model. The project includes image preprocessing, data augmentation, transfer learning, model training, evaluation, confusion matrix, ROC-AUC analysis, and t-SNE feature visualization.
"""
BUSI Breast Ultrasound Classification using DenseNet201
=========================================================

Dataset:
    BUSI (Breast Ultrasound Images)

Classes:
    benign, malignant, normal

Pipeline:
    ImageFolder -> Stratified 80/20 split -> DenseNet201
    -> CrossEntropyLoss -> Adam -> Evaluation
    -> Confusion Matrix -> Multi-class ROC/AUC
    -> DenseNet201 feature extraction -> t-SNE

Expected dataset structure:
    DATA_DIR/
        benign/
        malignant/
        normal/

For Google Colab, update DATA_DIR below, for example:
    DATA_DIR = "/content/Untitled Folder"

Important:
    The reported training accuracy and held-out validation accuracy are
    different quantities. Use the validation accuracy for model evaluation.
"""

import os
import random
import shutil
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "/content/Untitled Folder"
RESULTS_DIR = "/content/BUSI_results"

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.0001
RANDOM_SEED = 42
NUM_WORKERS = 2

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Reproducibility settings.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(RANDOM_SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("BUSI DenseNet201 Classification")
print("=" * 70)
print("Device:", device)
print("Dataset:", DATA_DIR)
print("Results:", RESULTS_DIR)
print("=" * 70)


# ============================================================
# CHECK DATASET
# ============================================================

if not os.path.isdir(DATA_DIR):
    raise FileNotFoundError(
        f"Dataset directory was not found:\n{DATA_DIR}\n\n"
        "Update DATA_DIR at the top of this script."
    )


# ============================================================
# REMOVE JUPYTER CHECKPOINT FOLDERS
# ============================================================

for root, dirs, files in os.walk(DATA_DIR):
    for directory in list(dirs):
        if directory == ".ipynb_checkpoints":
            checkpoint_path = os.path.join(root, directory)
            try:
                shutil.rmtree(checkpoint_path)
                print("Removed:", checkpoint_path)
            except Exception as exc:
                print("Could not remove:", checkpoint_path, exc)


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ============================================================
# LOAD DATASET
# ============================================================

base_dataset = datasets.ImageFolder(DATA_DIR)

class_names = base_dataset.classes
num_classes = len(class_names)

if num_classes < 2:
    raise ValueError(
        f"Expected at least 2 classes, found: {class_names}"
    )

print("\nClasses:")
for index, class_name in enumerate(class_names):
    print(f"  {index}: {class_name}")

print("\nTotal images:", len(base_dataset))


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

class_counts = np.bincount(
    np.array(base_dataset.targets),
    minlength=num_classes
)

print("\nClass distribution:")
for class_name, count in zip(class_names, class_counts):
    print(f"  {class_name}: {count}")


# ============================================================
# STRATIFIED 80/20 SPLIT
# ============================================================

indices = np.arange(len(base_dataset))
labels = np.array(base_dataset.targets)

train_indices, val_indices = train_test_split(
    indices,
    test_size=0.20,
    random_state=RANDOM_SEED,
    stratify=labels,
)

print("\nTrain images:", len(train_indices))
print("Validation images:", len(val_indices))


# ============================================================
# DATASETS WITH DIFFERENT TRANSFORMS
# ============================================================

train_dataset_full = datasets.ImageFolder(
    DATA_DIR,
    transform=train_transform
)

val_dataset_full = datasets.ImageFolder(
    DATA_DIR,
    transform=val_transform
)

train_dataset = Subset(
    train_dataset_full,
    train_indices
)

val_dataset = Subset(
    val_dataset_full,
    val_indices
)


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)


# ============================================================
# DENSENET201
# ============================================================

print("\nLoading pretrained DenseNet201...")

try:
    weights = models.DenseNet201_Weights.DEFAULT
    model = models.densenet201(weights=weights)
except AttributeError:
    # Compatibility with older torchvision versions.
    model = models.densenet201(pretrained=True)

input_features = model.classifier.in_features

# Final classification layer.
model.classifier = nn.Linear(
    input_features,
    num_classes
)

model = model.to(device)

print("DenseNet201 feature size:", input_features)
print("Number of output classes:", num_classes)


# ============================================================
# LOSS AND OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING HISTORY
# ============================================================

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

best_val_accuracy = 0.0

best_model_path = os.path.join(
    RESULTS_DIR,
    "best_busi_densenet201.pth"
)


# ============================================================
# TRAINING LOOP
# ============================================================

print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)

for epoch in range(NUM_EPOCHS):

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels_batch in train_loader:

        images = images.to(device)
        labels_batch = labels_batch.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels_batch
        )

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            predictions == labels_batch
        ).sum().item()

        total += labels_batch.size(0)

    epoch_train_loss = running_loss / total
    epoch_train_accuracy = correct / total

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    model.eval()

    val_running_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels_batch in val_loader:

            images = images.to(device)
            labels_batch = labels_batch.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels_batch
            )

            val_running_loss += (
                loss.item() * images.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            val_correct += (
                predictions == labels_batch
            ).sum().item()

            val_total += labels_batch.size(0)

    epoch_val_loss = val_running_loss / val_total
    epoch_val_accuracy = val_correct / val_total

    train_losses.append(epoch_train_loss)
    train_accuracies.append(epoch_train_accuracy)

    val_losses.append(epoch_val_loss)
    val_accuracies.append(epoch_val_accuracy)

    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if epoch_val_accuracy > best_val_accuracy:

        best_val_accuracy = epoch_val_accuracy

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "image_size": IMAGE_SIZE,
                "epoch": epoch + 1,
                "validation_accuracy": epoch_val_accuracy,
            },
            best_model_path
        )

    # --------------------------------------------------------
    # Print epoch results
    # --------------------------------------------------------

    print(
        f"Epoch {epoch + 1:02d}/{NUM_EPOCHS} | "
        f"Loss: {epoch_train_loss:.4f} | "
        f"Accuracy: {epoch_train_accuracy:.4f} | "
        f"Val Loss: {epoch_val_loss:.4f} | "
        f"Val Accuracy: {epoch_val_accuracy:.4f}"
    )


# ============================================================
# LOAD BEST MODEL
# ============================================================

checkpoint = torch.load(
    best_model_path,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("\nBest validation accuracy:")
print(f"{best_val_accuracy * 100:.2f}%")


# ============================================================
# FINAL VALIDATION PREDICTIONS
# ============================================================

y_true = []
y_pred = []
y_prob = []

with torch.no_grad():

    for images, labels_batch in val_loader:

        images = images.to(device)

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = torch.argmax(
            probabilities,
            dim=1
        )

        y_true.extend(
            labels_batch.numpy()
        )

        y_pred.extend(
            predictions.cpu().numpy()
        )

        y_prob.extend(
            probabilities.cpu().numpy()
        )


y_true = np.asarray(y_true)
y_pred = np.asarray(y_pred)
y_prob = np.asarray(y_prob)

final_accuracy = np.mean(
    y_true == y_pred
)

print("\nFinal held-out validation accuracy:")
print(f"{final_accuracy * 100:.2f}%")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
)

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)
print(report)

with open(
    os.path.join(
        RESULTS_DIR,
        "classification_report.txt"
    ),
    "w"
) as file:

    file.write(
        "BUSI DenseNet201 Classification Report\n\n"
    )

    file.write(
        f"Final validation accuracy: "
        f"{final_accuracy * 100:.2f}%\n"
    )

    file.write(
        f"Best validation accuracy: "
        f"{best_val_accuracy * 100:.2f}%\n\n"
    )

    file.write(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - BUSI Dataset")
plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "busi_confusion_matrix.png"
    ),
    dpi=600,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# MULTI-CLASS ROC / AUC
# ============================================================

y_true_binary = label_binarize(
    y_true,
    classes=np.arange(num_classes)
)

fpr = {}
tpr = {}
roc_auc = {}

for i in range(num_classes):

    fpr[i], tpr[i], _ = roc_curve(
        y_true_binary[:, i],
        y_prob[:, i]
    )

    roc_auc[i] = auc(
        fpr[i],
        tpr[i]
    )


# Macro-average ROC
all_fpr = np.unique(
    np.concatenate(
        [
            fpr[i]
            for i in range(num_classes)
        ]
    )
)

mean_tpr = np.zeros_like(
    all_fpr
)

for i in range(num_classes):

    mean_tpr += np.interp(
        all_fpr,
        fpr[i],
        tpr[i]
    )

mean_tpr /= num_classes

macro_auc = auc(
    all_fpr,
    mean_tpr
)


# ============================================================
# ROC PLOT
# ============================================================

plt.figure(figsize=(9, 7))

for i, class_name in enumerate(class_names):

    plt.plot(
        fpr[i],
        tpr[i],
        linewidth=2,
        label=(
            f"{class_name} "
            f"(AUC = {roc_auc[i]:.3f})"
        )
    )

plt.plot(
    [0, 1],
    [0, 1],
    "k--",
    linewidth=1.5,
    label="Random classifier"
)

plt.plot(
    all_fpr,
    mean_tpr,
    "k:",
    linewidth=3,
    label=(
        f"Macro-average "
        f"(AUC = {macro_auc:.3f})"
    )
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multi-Class ROC Curve - BUSI Dataset")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "busi_roc_curve.png"
    ),
    dpi=600,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# DENSENET201 FEATURE EXTRACTION
# ============================================================

print("\nExtracting DenseNet201 deep features...")

feature_extractor = nn.Sequential(
    model.features,
    nn.ReLU(inplace=False),
    nn.AdaptiveAvgPool2d((1, 1))
).to(device)

feature_extractor.eval()

features = []
feature_labels = []

with torch.no_grad():

    for images, labels_batch in val_loader:

        images = images.to(device)

        feature_maps = feature_extractor(
            images
        )

        feature_vectors = feature_maps.view(
            feature_maps.size(0),
            -1
        )

        features.append(
            feature_vectors.cpu().numpy()
        )

        feature_labels.append(
            labels_batch.numpy()
        )


features = np.concatenate(
    features,
    axis=0
)

feature_labels = np.concatenate(
    feature_labels,
    axis=0
)

print(
    "Feature matrix shape:",
    features.shape
)


# ============================================================
# t-SNE
# ============================================================

if len(features) >= 4:

    print("\nRunning t-SNE...")

    # t-SNE requires perplexity < number of samples.
    perplexity = min(30, len(features) - 1)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=RANDOM_SEED,
        init="pca",
        learning_rate="auto"
    )

    features_tsne = tsne.fit_transform(
        features
    )

    # --------------------------------------------------------
    # t-SNE plot
    # --------------------------------------------------------

    plt.figure(figsize=(10, 8))

    for class_id, class_name in enumerate(class_names):

        mask = (
            feature_labels == class_id
        )

        plt.scatter(
            features_tsne[mask, 0],
            features_tsne[mask, 1],
            s=45,
            alpha=0.75,
            label=class_name
        )

    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.title(
        "t-SNE Visualization of DenseNet201 Features - BUSI"
    )
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "busi_tsne.png"
        ),
        dpi=600,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

else:

    print(
        "Not enough validation samples for t-SNE."
    )


# ============================================================
# TRAINING / VALIDATION ACCURACY
# ============================================================

epochs = range(
    1,
    NUM_EPOCHS + 1
)

plt.figure(figsize=(9, 6))

plt.plot(
    epochs,
    train_accuracies,
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    epochs,
    val_accuracies,
    marker="s",
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title(
    "Training and Validation Accuracy - BUSI"
)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "busi_accuracy_curve.png"
    ),
    dpi=600,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# TRAINING / VALIDATION LOSS
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(
    epochs,
    train_losses,
    marker="o",
    label="Training Loss"
)

plt.plot(
    epochs,
    val_losses,
    marker="s",
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(
    "Training and Validation Loss - BUSI"
)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "busi_loss_curve.png"
    ),
    dpi=600,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# SAVE NUMERICAL ARRAYS
# ============================================================

np.save(
    os.path.join(
        RESULTS_DIR,
        "busi_features.npy"
    ),
    features
)

np.save(
    os.path.join(
        RESULTS_DIR,
        "busi_labels.npy"
    ),
    feature_labels
)

np.save(
    os.path.join(
        RESULTS_DIR,
        "busi_probabilities.npy"
    ),
    y_prob
)

np.save(
    os.path.join(
        RESULTS_DIR,
        "busi_predictions.npy"
    ),
    y_pred
)

np.save(
    os.path.join(
        RESULTS_DIR,
        "busi_true_labels.npy"
    ),
    y_true
)


# ============================================================
# SAVE A COMPLETE RESULTS SUMMARY
# ============================================================

results_file = os.path.join(
    RESULTS_DIR,
    "busi_results.txt"
)

with open(results_file, "w") as file:

    file.write(
        "BUSI Breast Ultrasound Classification\n"
    )

    file.write(
        "DenseNet201 Transfer Learning\n"
    )

    file.write("=" * 60 + "\n\n")

    file.write(
        f"Classes: {class_names}\n"
    )

    file.write(
        f"Total images: {len(base_dataset)}\n"
    )

    file.write(
        f"Training images: {len(train_dataset)}\n"
    )

    file.write(
        f"Validation images: {len(val_dataset)}\n"
    )

    file.write(
        f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}\n"
    )

    file.write(
        f"Batch size: {BATCH_SIZE}\n"
    )

    file.write(
        f"Epochs: {NUM_EPOCHS}\n"
    )

    file.write(
        f"Learning rate: {LEARNING_RATE}\n"
    )

    file.write(
        "\n"
    )

    file.write(
        f"Final validation accuracy: "
        f"{final_accuracy * 100:.2f}%\n"
    )

    file.write(
        f"Best validation accuracy: "
        f"{best_val_accuracy * 100:.2f}%\n"
    )

    file.write(
        f"Macro ROC-AUC: {macro_auc:.4f}\n\n"
    )

    file.write("Class-wise ROC-AUC:\n")

    for i, class_name in enumerate(class_names):

        file.write(
            f"  {class_name}: "
            f"{roc_auc[i]:.4f}\n"
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("BUSI EXPERIMENT COMPLETED")
print("=" * 70)

print(
    f"Final validation accuracy: "
    f"{final_accuracy * 100:.2f}%"
)

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    f"Macro ROC-AUC: "
    f"{macro_auc:.4f}"
)

print("\nResults saved to:")
print(RESULTS_DIR)

print("\nGenerated files:")

for filename in sorted(os.listdir(RESULTS_DIR)):
    print("  -", filename)

print("\nDone.")

