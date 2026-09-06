"""
SPQ - Module 0
FP32 Baseline Training Module

Purpose:
    1. Stream the SPQ dataset from Hugging Face.
    2. Detect all class names.
    3. Create a deterministic 80/20 per-class train/test split.
    4. Handle class imbalance using predefined class weights.
    5. Train an FP32 MobileNetV2 disease-classification model.
    6. Evaluate the untouched test split.

Important:
    - No quantization is performed in Module 0.
    - Test data is never used for training.
    - Hugging Face token is supplied by the caller.
    - No secret/token is stored in this file.
"""

from collections import defaultdict

import torch
import torch.nn as nn

from torch.utils.data import DataLoader, IterableDataset

from torchvision import transforms
from torchvision.models import (
    mobilenet_v2,
    MobileNet_V2_Weights
)

from datasets import load_dataset

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)


# ================================================================
# STREAMING DATASET
# ================================================================

class SPQStreamingDataset(IterableDataset):

    def __init__(
        self,
        repo_id,
        split,
        token,
        transform
    ):

        self.repo_id = repo_id
        self.split = split
        self.token = token
        self.transform = transform

    def __iter__(self):

        dataset = load_dataset(
            "imagefolder",
            data_files={
                "train": f"hf://datasets/{self.repo_id}/**"
            },
            streaming=True,
            token=self.token,
        )["train"]

        class_counters = defaultdict(int)

        for sample in dataset:

            label = int(sample["label"])

            count = class_counters[label]

            class_counters[label] += 1

            # Deterministic 80/20 split
            current_split = (
                "test"
                if count % 5 == 0
                else "train"
            )

            if current_split != self.split:
                continue

            image = sample["image"]

            if image.mode != "RGB":
                image = image.convert("RGB")

            image = self.transform(image)

            yield image, label


# ================================================================
# DATASET INFORMATION
# ================================================================

def get_dataset_info(repo_id, token):

    dataset = load_dataset(
        "imagefolder",
        data_files={
            "train": f"hf://datasets/{repo_id}/**"
        },
        streaming=True,
        token=token,
    )["train"]

    class_names = dataset.features["label"].names

    return class_names, len(class_names)


# ================================================================
# TRANSFORMS
# ================================================================

def create_transforms():

    train_transform = transforms.Compose([

        transforms.Resize((224, 224)),

        transforms.RandomHorizontalFlip(),

        transforms.RandomRotation(10),

        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    test_transform = transforms.Compose([

        transforms.Resize((224, 224)),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return train_transform, test_transform


# ================================================================
# MODEL
# ================================================================

def create_model(num_classes, device):

    model = mobilenet_v2(
        weights=MobileNet_V2_Weights.DEFAULT
    )

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        num_classes,
    )

    # Keep FP32
    model = model.float().to(device)

    return model


# ================================================================
# CLASS WEIGHTS
# ================================================================

def get_class_weights():

    """
    Class counts from the cleaned local dataset.

    Class order:
        0 - Fall-Armyworm
        1 - Healthy
        2 - Maize-Streak-Virus
        3 - Maydis-Leaf-Blight
    """

    class_counts = torch.tensor(
        [
            1337,
            7928,
            7342,
            3953
        ],
        dtype=torch.float32
    )

    total = class_counts.sum()

    num_classes = len(class_counts)

    weights = total / (
        num_classes * class_counts
    )

    return weights


# ================================================================
# TRAIN ONE EPOCH
# ================================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    epoch,
    total_epochs
):

    model.train()

    running_loss = 0.0

    correct = 0
    total = 0
    batches = 0

    print()
    print(
        f"Epoch {epoch}/{total_epochs}"
    )

    print("-" * 60)

    for images, labels in loader:

        images = images.to(
            device,
            dtype=torch.float32
        )

        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        batches += 1

        # Show progress every 10 batches
        if batches % 10 == 0:

            accuracy = (
                correct / total
            ) * 100.0

            print(
                f"Batch {batches:4d} | "
                f"Loss: {loss.item():.4f} | "
                f"Accuracy: {accuracy:.2f}%",
                flush=True
            )

    if batches == 0:

        raise RuntimeError(
            "No training batches were produced."
        )

    loss_value = (
        running_loss / batches
    )

    accuracy = (
        correct / total
    ) * 100.0

    print("-" * 60)

    print(
        f"Epoch {epoch}/{total_epochs} completed | "
        f"Loss: {loss_value:.4f} | "
        f"Accuracy: {accuracy:.2f}%",
        flush=True
    )

    return loss_value, accuracy


# ================================================================
# EVALUATION
# ================================================================

def evaluate(
    model,
    loader,
    criterion,
    device,
    class_names
):

    model.eval()

    total_loss = 0.0

    total = 0
    correct = 0
    batches = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                device,
                dtype=torch.float32
            )

            labels_device = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels_device
            )

            predictions = outputs.argmax(
                dim=1
            )

            total_loss += loss.item()

            batches += 1

            correct += (
                predictions == labels_device
            ).sum().item()

            total += labels.size(0)

            all_labels.extend(
                labels.numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    if batches == 0:

        raise RuntimeError(
            "No test batches were produced."
        )

    accuracy = (
        correct / total
    ) * 100.0

    precision = precision_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0,
    )

    cm = confusion_matrix(
        all_labels,
        all_predictions,
    )

    report = classification_report(
        all_labels,
        all_predictions,
        target_names=class_names,
        zero_division=0,
    )

    return {
        "loss": total_loss / batches,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "classification_report": report,
        "test_images": total,
    }


# ================================================================
# MAIN MODULE 0
# ================================================================

def run_module0(
    repo_id,
    hf_token,
    batch_size=32,
    epochs=5,
    learning_rate=1e-4,
    step_size=3,
    gamma=0.1,
    device=None,
):

    # ------------------------------------------------------------
    # Device
    # ------------------------------------------------------------

    if device is None:

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    else:

        device = torch.device(device)

    print(
        "=============================================="
    )

    print(
        "SPQ MODULE 0 - FP32 BASELINE TRAINING"
    )

    print(
        "=============================================="
    )

    print(
        "Device:",
        device
    )

    print(
        "Precision: FP32"
    )

    print(
        "Batch size:",
        batch_size
    )

    print(
        "Epochs:",
        epochs
    )

    print(
        "Learning rate:",
        learning_rate
    )

    print()

    # ------------------------------------------------------------
    # Dataset information
    # ------------------------------------------------------------

    class_names, num_classes = (
        get_dataset_info(
            repo_id,
            hf_token,
        )
    )

    print("Detected classes:")

    for index, name in enumerate(
        class_names
    ):

        print(
            f"  {index}: {name}"
        )

    print(
        "Number of classes:",
        num_classes
    )

    print()

    # ------------------------------------------------------------
    # Class weights
    # ------------------------------------------------------------

    class_weights = get_class_weights()

    print("Class weights:")

    for index, weight in enumerate(
        class_weights
    ):

        print(
            f"  {class_names[index]}: "
            f"{weight.item():.4f}"
        )

    print()

    # ------------------------------------------------------------
    # Transforms
    # ------------------------------------------------------------

    train_transform, test_transform = (
        create_transforms()
    )

    # ------------------------------------------------------------
    # Streaming datasets
    # ------------------------------------------------------------

    train_dataset = SPQStreamingDataset(
        repo_id=repo_id,
        split="train",
        token=hf_token,
        transform=train_transform,
    )

    test_dataset = SPQStreamingDataset(
        repo_id=repo_id,
        split="test",
        token=hf_token,
        transform=test_transform,
    )

    # ------------------------------------------------------------
    # Data loaders
    # ------------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=0,
    )

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------

    model = create_model(
        num_classes,
        device,
    )

    # ------------------------------------------------------------
    # Weighted loss
    # ------------------------------------------------------------

    class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # ------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    # ------------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=step_size,
        gamma=gamma,
    )

    history = []

    # ============================================================
    # TRAINING
    # ============================================================

    for epoch in range(
        1,
        epochs + 1
    ):

        train_loss, train_accuracy = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                epoch,
                epochs,
            )
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        history.append({

            "epoch": epoch,

            "loss": train_loss,

            "accuracy": train_accuracy,

            "learning_rate": current_lr,
        })

        scheduler.step()

    # ============================================================
    # FINAL TEST
    # ============================================================

    print()

    print(
        "=============================================="
    )

    print(
        "MODULE 0 - FINAL FP32 TEST EVALUATION"
    )

    print(
        "=============================================="
    )

    results = evaluate(
        model,
        test_loader,
        criterion,
        device,
        class_names,
    )

    print(
        f"Test Loss     : "
        f"{results['loss']:.4f}"
    )

    print(
        f"Test Accuracy : "
        f"{results['accuracy']:.2f}%"
    )

    print(
        f"Precision     : "
        f"{results['precision']:.4f}"
    )

    print(
        f"Recall        : "
        f"{results['recall']:.4f}"
    )

    print(
        f"F1 Score      : "
        f"{results['f1']:.4f}"
    )

    print(
        f"Test Images   : "
        f"{results['test_images']}"
    )

    print()

    print(
        "Classification Report:"
    )

    print(
        results["classification_report"]
    )

    print(
        "Confusion Matrix:"
    )

    print(
        results["confusion_matrix"]
    )

    print()

    print(
        "=============================================="
    )

    print(
        "MODULE 0 COMPLETED"
    )

    print(
        "=============================================="
    )

    return (
        model,
        class_names,
        history,
        results,
    )
