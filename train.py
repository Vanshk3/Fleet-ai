"""
Train the tyre defect classifier using MobileNetV2 transfer learning.

Usage:
    python train.py
    python train.py --epochs 20 --batch-size 32
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def build_model(num_classes: int = 2, freeze_backbone: bool = True):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes)
    )
    return model.to(DEVICE)


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def val_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels, all_probs


def main(args):
    print(f"Using device: {DEVICE}")

    train_tf, val_tf = get_transforms()

    train_dataset = datasets.ImageFolder("data/train", transform=train_tf)
    val_dataset = datasets.ImageFolder("data/val", transform=val_tf)
    test_dataset = datasets.ImageFolder("data/test", transform=val_tf)

    class_names = train_dataset.classes
    print(f"Classes: {class_names}")
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = build_model(num_classes=len(class_names))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"\nTraining for {args.epochs} epochs...\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, _, _, _ = val_epoch(model, val_loader, criterion)
        scheduler.step(val_loss)

        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 4))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_DIR / "best_model.pth")
            print(f"Epoch {epoch:02d}/{args.epochs} | loss {train_loss:.4f} | acc {train_acc:.4f} | val_loss {val_loss:.4f} | val_acc {val_acc:.4f} | saved ✓  ({time.time()-t0:.1f}s)")
        else:
            print(f"Epoch {epoch:02d}/{args.epochs} | loss {train_loss:.4f} | acc {train_acc:.4f} | val_loss {val_loss:.4f} | val_acc {val_acc:.4f}  ({time.time()-t0:.1f}s)")

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")

    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(MODEL_DIR / "best_model.pth", map_location=DEVICE))
    _, test_acc, preds, labels, probs = val_epoch(model, test_loader, criterion)

    print(f"Test accuracy: {test_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=class_names))

    cm = confusion_matrix(labels, preds)
    print("Confusion Matrix:")
    print(cm)

    metadata = {
        "classes": class_names,
        "test_accuracy": round(test_acc, 4),
        "best_val_accuracy": round(best_val_acc, 4),
        "history": history,
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(labels, preds, target_names=class_names, output_dict=True),
        "device": str(DEVICE),
        "epochs": args.epochs,
        "architecture": "MobileNetV2 (transfer learning, ImageNet weights)"
    }

    with open(MODEL_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nModel saved to models/best_model.pth")
    print(f"Metadata saved to models/metadata.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    main(args)
