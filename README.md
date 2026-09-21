# ============================================================
# Breast Cancer Classification using BUSI Dataset
# Model: DenseNet201 Transfer Learning
# ============================================================

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE


# ============================================================
# 1. CONFIGURATION
# ============================================================

# Change this path according to your dataset location
DATA_DIR = "/content/Untitled Folder"

# Folder where results will be saved
RESULTS_DIR = "/content/BUSI_results"

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.0001
RANDOM_SEED = 42

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(RANDOM_SEED)


# ============================================================
# 3. DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("==============================================")
print("Device:", device)
print("==============================================")


# ============================================================
# 4. IMAGE TRANSFORMS
# ============================================================

# Training augmentation
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
    )
])


# Validation transform
val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# 5. LOAD BUSI DATASET
# ============================================================

print("\nLoading BUSI dataset...")

# Temporary dataset used to obtain image paths and labels
base_dataset = datasets.ImageFolder(
    root=DATA_DIR
)

class_names = base_dataset.classes
num_classes = len(class_names)

print("\nClasses:")
for i, class_name in enumerate(class_names):
    print(f"{i}: {class_name}")

print("\nTotal images:", len(base_dataset))


# ============================================================
# 6. CLASS DISTRIBUTION
# ============================================================

labels = np.array(base_dataset.targets)

print("\nClass distribution:")

for class_index, class_name in enumerate(class_names):
    count = np.sum(labels == class_index)
    print(f"{class_name}: {count}")


# ============================================================
# 7. STRATIFIED TRAIN/VALIDATION SPLIT
# ============================================================

indices = np.arange(len(base_dataset))

train_indices, val_indices = train_test_split(
    indices,
    test_size=0.20,
    random_state=RANDOM_SEED,
    stratify=labels
)

print("\n==============================================")
print("Dataset Split")
print("==============================================")
print("Training images:", len(train_indices))
print("Validation images:", len(val_indices))


# ============================================================
# 8. CREATE DATASETS WITH DIFFERENT TRANSFORMS
# ============================================================

train_dataset_full = datasets.ImageFolder(
    root=DATA_DIR,
    transform=train_transform
)

val_dataset_full = datasets.ImageFolder(
    root=DATA_DIR,
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
# 9. DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)


# ============================================================
# 10. LOAD PRETRAINED DENSENET201
# ============================================================

print("\nLoading pretrained DenseNet201...")

try:
    weights = models.DenseNet201_Weights.DEFAULT
    model = models.densenet201(weights=weights)
except AttributeError:
    model = models.densenet201(pretrained=True)


# ============================================================
# 11. REPLACE FINAL CLASSIFICATION LAYER
# ============================================================

input_features = model.classifier.in_features

model.classifier = nn.Linear(
    input_features,
    num_classes
)

model = model.to(device)

print("\nDenseNet201 loaded successfully.")
print("Number of output classes:", num_classes)


# ============================================================
# 12. LOSS FUNCTION AND OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 13. TRAINING FUNCTION
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels_batch in loader:

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

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels_batch.size(0)

        correct += (
            predicted == labels_batch
        ).sum().item()

    epoch_loss = running_loss / total

    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# 14. VALIDATION FUNCTION
# ============================================================

def validate(model, loader, criterion):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():

        for images, labels_batch in loader:

            images = images.to(device)
            labels_batch = labels_batch.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels_batch
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            _, predicted = torch.max(
                outputs,
                1
            )

            running_loss += (
                loss.item() * images.size(0)
            )

            total += labels_batch.size(0)

            correct += (
                predicted == labels_batch
            ).sum().item()

            all_labels.extend(
                labels_batch.cpu().numpy()
            )

            all_predictions.extend(
                predicted.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

    epoch_loss = running_loss / total

    epoch_accuracy = correct / total

    return (
        epoch_loss,
        epoch_accuracy,
        np.array(all_labels),
        np.array(all_predictions),
        np.array(all_probabilities)
    )


# ============================================================
# 15. TRAIN MODEL
# ============================================================

train_losses = []
train_accuracies = []

val_losses = []
val_accuracies = []

best_val_accuracy = 0.0

best_model_path = os.path.join(
    RESULTS_DIR,
    "best_densenet201_busi.pth"
)


print("\n==============================================")
print("Starting Training")
print("==============================================")


for epoch in range(NUM_EPOCHS):

    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer
    )

    (
        val_loss,
        val_accuracy,
        _,
        _,
        _
    ) = validate(
        model,
        val_loader,
        criterion
    )

    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    print(
        f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_accuracy:.4f} "
        f"Val Loss: {val_loss:.4f} "
        f"Val Acc: {val_accuracy:.4f}"
    )

    # Save best model
    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            best_model_path
        )


print("\nTraining completed.")

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy:.4f}"
)


# ============================================================
# 16. LOAD BEST MODEL
# ============================================================

model.load_state_dict(
    torch.load(
        best_model_path,
        map_location=device
    )
)

model.eval()


# ============================================================
# 17. FINAL VALIDATION PREDICTIONS
# ============================================================

(
    val_loss,
    val_accuracy,
    true_labels,
    predictions,
    probabilities
) = validate(
    model,
    val_loader,
    criterion
)


print("\n==============================================")
print("Final Validation Results")
print("==============================================")

print(
    f"Validation Loss: {val_loss:.4f}"
)

print(
    f"Validation Accuracy: {val_accuracy:.4f}"
)


# ============================================================
# 18. CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    true_labels,
    predictions,
    target_names=class_names,
    digits=4
)

print("\nClassification Report:")
print(report)


with open(
    os.path.join(
        RESULTS_DIR,
        "classification_report.txt"
    ),
    "w"
) as file:

    file.write(report)


# ============================================================
# 19. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    true_labels,
    predictions
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
plt.title("BUSI - DenseNet201 Confusion Matrix")

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "confusion_matrix.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# 20. ROC CURVE
# ============================================================

binary_labels = label_binarize(
    true_labels,
    classes=np.arange(num_classes)
)

plt.figure(figsize=(8, 6))

roc_auc_values = {}

for i in range(num_classes):

    fpr, tpr, _ = roc_curve(
        binary_labels[:, i],
        probabilities[:, i]
    )

    class_auc = auc(
        fpr,
        tpr
    )

    roc_auc_values[class_names[i]] = class_auc

    plt.plot(
        fpr,
        tpr,
        label=f"{class_names[i]} (AUC = {class_auc:.4f})"
    )


# Macro-average ROC
all_fpr = np.unique(
    np.concatenate([
        roc_curve(
            binary_labels[:, i],
            probabilities[:, i]
        )[0]
        for i in range(num_classes)
    ])
)

mean_tpr = np.zeros_like(all_fpr)

for i in range(num_classes):

    fpr, tpr, _ = roc_curve(
        binary_labels[:, i],
        probabilities[:, i]
    )

    mean_tpr += np.interp(
        all_fpr,
        fpr,
        tpr
    )

mean_tpr /= num_classes

macro_auc = auc(
    all_fpr,
    mean_tpr
)

plt.plot(
    all_fpr,
    mean_tpr,
    linestyle="--",
    linewidth=2,
    label=f"Macro-average (AUC = {macro_auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle=":"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "BUSI - DenseNet201 ROC Curves"
)

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "roc_curve.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# 21. TRAINING CURVES
# ============================================================

epochs_range = range(
    1,
    NUM_EPOCHS + 1
)

plt.figure(figsize=(8, 6))

plt.plot(
    epochs_range,
    train_accuracies,
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    epochs_range,
    val_accuracies,
    marker="o",
    label="Validation Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title(
    "BUSI - Training and Validation Accuracy"
)

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "accuracy_curve.png"
    ),
    dpi=300
)

plt.show()


plt.figure(figsize=(8, 6))

plt.plot(
    epochs_range,
    train_losses,
    marker="o",
    label="Training Loss"
)

plt.plot(
    epochs_range,
    val_losses,
    marker="o",
    label="Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title(
    "BUSI - Training and Validation Loss"
)

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "loss_curve.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# 22. DENSENET201 FEATURE EXTRACTION
# ============================================================

print("\nExtracting DenseNet201 features...")


feature_extractor = nn.Sequential(
    model.features,
    nn.ReLU(inplace=False),
    nn.AdaptiveAvgPool2d((1, 1))
)

feature_extractor = feature_extractor.to(device)

feature_extractor.eval()


def extract_features(model, loader):

    features = []
    labels = []

    with torch.no_grad():

        for images, batch_labels in loader:

            images = images.to(device)

            output = model(images)

            output = output.view(
                output.size(0),
                -1
            )

            features.append(
                output.cpu().numpy()
            )

            labels.append(
                batch_labels.numpy()
            )

    features = np.concatenate(
        features,
        axis=0
    )

    labels = np.concatenate(
        labels,
        axis=0
    )

    return features, labels


features, feature_labels = extract_features(
    feature_extractor,
    val_loader
)


print(
    "Feature matrix shape:",
    features.shape
)


# ============================================================
# 23. SAVE FEATURES
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
    probabilities
)

np.save(
    os.path.join(
        RESULTS_DIR,
        "busi_predictions.npy"
    ),
    predictions
)

np.save(
    os.path.join(
        RESULTS_DIR,
        "busi_true_labels.npy"
    ),
    true_labels
)


# ============================================================
# 24. t-SNE VISUALIZATION
# ============================================================

print("\nRunning t-SNE...")

# Limit samples if the validation set is very large
MAX_TSNE_SAMPLES = 1000

if len(features) > MAX_TSNE_SAMPLES:

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    selected_indices = rng.choice(
        len(features),
        MAX_TSNE_SAMPLES,
        replace=False
    )

    tsne_features = features[
        selected_indices
    ]

    tsne_labels = feature_labels[
        selected_indices
    ]

else:

    tsne_features = features

    tsne_labels = feature_labels


perplexity = min(
    30,
    max(5, len(tsne_features) // 4)
)

tsne = TSNE(
    n_components=2,
    random_state=RANDOM_SEED,
    perplexity=perplexity,
    init="pca",
    learning_rate="auto"
)

features_2d = tsne.fit_transform(
    tsne_features
)


plt.figure(figsize=(9, 7))

for class_index, class_name in enumerate(class_names):

    mask = (
        tsne_labels == class_index
    )

    plt.scatter(
        features_2d[mask, 0],
        features_2d[mask, 1],
        label=class_name,
        alpha=0.7
    )

plt.xlabel("t-SNE Dimension 1")

plt.ylabel("t-SNE Dimension 2")

plt.title(
    "BUSI - DenseNet201 Feature Space"
)

plt.legend()

plt.grid(alpha=0.2)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "tsne.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# 25. SAVE EXPERIMENT SUMMARY
# ============================================================

summary_path = os.path.join(
    RESULTS_DIR,
    "experiment_summary.txt"
)

with open(
    summary_path,
    "w"
) as file:

    file.write(
        "BUSI Breast Cancer Classification\n"
    )

    file.write(
        "=================================\n\n"
    )

    file.write(
        f"Model: DenseNet201\n"
    )

    file.write(
        f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}\n"
    )

    file.write(
        f"Batch size: {BATCH_SIZE}\n"
    )

    file.write(
        f"Learning rate: {LEARNING_RATE}\n"
    )

    file.write(
        f"Epochs: {NUM_EPOCHS}\n"
    )

    file.write(
        f"Number of classes: {num_classes}\n"
    )

    file.write(
        f"Classes: {class_names}\n"
    )

    file.write(
        f"Training images: {len(train_dataset)}\n"
    )

    file.write(
        f"Validation images: {len(val_dataset)}\n"
    )

    file.write(
        f"Best validation accuracy: "
        f"{best_val_accuracy:.4f}\n"
    )

    file.write(
        f"Final validation accuracy: "
        f"{val_accuracy:.4f}\n"
    )

    file.write(
        f"Macro ROC-AUC: "
        f"{macro_auc:.4f}\n"
    )


# ============================================================
# 26. FINAL OUTPUT
# ============================================================

print("\n==============================================")
print("EXPERIMENT COMPLETED")
print("==============================================")

print(
    "Best model saved at:"
)

print(best_model_path)

print(
    "\nResults saved at:"
)

print(RESULTS_DIR)

print("\nGenerated files include:")

print("- best_densenet201_busi.pth")
print("- classification_report.txt")
print("- confusion_matrix.png")
print("- roc_curve.png")
print("- accuracy_curve.png")
print("- loss_curve.png")
print("- tsne.png")
print("- experiment_summary.txt")
print("- busi_features.npy")
print("- busi_labels.npy")
print("- busi_probabilities.npy")
print("- busi_predictions.npy")
print("- busi_true_labels.npy")
