import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

DATASET_DIR = Path("dataset/layout")
OUTPUT_DIR = Path("models")
MODEL_PATH = OUTPUT_DIR / "gold_layout_classifier.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-4
IMAGE_SIZE = 224
NUM_WORKERS = 2


def build_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.05),
        transforms.RandomRotation(2),
        transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def main():
    train_tf, val_tf = build_transforms()

    train_ds = datasets.ImageFolder(DATASET_DIR / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(DATASET_DIR / "val", transform=val_tf)

    if len(train_ds.classes) < 1:
        raise RuntimeError("No layout classes found")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.DEFAULT
    )
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(train_ds.classes))
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    history = []

    for epoch in range(EPOCHS):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_count = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * images.size(0)
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_count += labels.size(0)

        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_count = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                logits = model(images)
                loss = criterion(logits, labels)

                val_loss_sum += loss.item() * images.size(0)
                val_correct += (logits.argmax(dim=1) == labels).sum().item()
                val_count += labels.size(0)

        train_loss = train_loss_sum / max(train_count, 1)
        train_acc = train_correct / max(train_count, 1)
        val_loss = val_loss_sum / max(val_count, 1)
        val_acc = val_correct / max(val_count, 1)

        history.append({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
        })

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_name": "efficientnet_b0",
                    "task": "gold_layout_classifier",
                    "image_size": IMAGE_SIZE,
                    "class_to_idx": train_ds.class_to_idx,
                    "model_state_dict": model.state_dict(),
                    "best_val_acc": best_val_acc,
                },
                MODEL_PATH,
            )
            print(f"Saved best model to {MODEL_PATH}")

    metrics_path = OUTPUT_DIR / "gold_layout_classifier_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "best_val_acc": best_val_acc,
                "history": history,
                "class_to_idx": train_ds.class_to_idx,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
