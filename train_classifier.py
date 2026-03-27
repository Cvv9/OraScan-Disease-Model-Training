"""
Train an EfficientNet-B0 disease classification model on the preprocessed dataset.

Training strategy:
  Phase 1: Train classifier head only (backbone frozen) - 5 epochs
  Phase 2: Fine-tune full model - 20 epochs with lower LR

Handles class imbalance via WeightedRandomSampler and class-weighted loss.
Exports best model to ONNX for DirectML inference on AMD GPU.

Usage:
  python train_classifier.py
  python train_classifier.py --epochs 30 --batch-size 32
"""

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report, confusion_matrix

# ── Configuration ──────────────────────────────────────────────────────
DATA_DIR = Path(r"E:\ECG and Dental Images\Dental data set\Training_Ready")
MODEL_DIR = Path(r"E:\ECG and Dental Images\Dental data set\Models")
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]  # ImageNet normalization
STD = [0.229, 0.224, 0.225]


def get_args():
    parser = argparse.ArgumentParser(description="Train dental disease classifier")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs-frozen", type=int, default=5,
                        help="Epochs with frozen backbone")
    parser.add_argument("--epochs-finetune", type=int, default=20,
                        help="Epochs for full fine-tuning")
    parser.add_argument("--lr-head", type=float, default=3e-3,
                        help="Learning rate for head training")
    parser.add_argument("--lr-finetune", type=float, default=3e-4,
                        help="Learning rate for fine-tuning")
    parser.add_argument("--patience", type=int, default=7,
                        help="Early stopping patience")
    parser.add_argument("--workers", type=int, default=4,
                        help="DataLoader workers")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    return parser.parse_args()


def get_transforms():
    """Data augmentation for training, standard transforms for val/test."""
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.1),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.14)),  # 256 for 224 input
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    return train_transform, eval_transform


def get_weighted_sampler(dataset):
    """Create WeightedRandomSampler for class-balanced batches."""
    targets = [s[1] for s in dataset.samples]
    class_counts = Counter(targets)
    num_samples = len(targets)

    # Inverse frequency weighting
    class_weights = {cls: num_samples / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[t] for t in targets]

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples,
        replacement=True
    )


def get_class_weights(dataset, device):
    """Compute class weights for loss function."""
    targets = [s[1] for s in dataset.samples]
    class_counts = Counter(targets)
    num_classes = len(class_counts)
    total = sum(class_counts.values())

    weights = torch.zeros(num_classes)
    for cls, count in class_counts.items():
        weights[cls] = total / (num_classes * count)

    # Clip extreme weights
    weights = torch.clamp(weights, min=0.5, max=5.0)
    return weights.to(device)


def build_model(num_classes):
    """Build EfficientNet-B0 with custom classifier head."""
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Replace classifier
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes),
    )

    return model


def freeze_backbone(model):
    """Freeze all layers except classifier."""
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False


def unfreeze_backbone(model):
    """Unfreeze all layers."""
    for param in model.parameters():
        param.requires_grad = True


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    start = time.time()
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(loader):
            elapsed = time.time() - start
            print(f"  Epoch [{epoch}/{total_epochs}] "
                  f"Batch [{batch_idx + 1}/{len(loader)}] "
                  f"Loss: {running_loss / total:.4f} "
                  f"Acc: {100. * correct / total:.1f}% "
                  f"({elapsed:.0f}s)")

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_targets)


def export_onnx(model, num_classes, class_names, device):
    """Export model to ONNX for DirectML inference."""
    model.eval()
    onnx_path = MODEL_DIR / "dental_classifier.onnx"

    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    print(f"ONNX model exported to: {onnx_path}")

    # Save class names alongside
    meta_path = MODEL_DIR / "model_metadata.json"
    metadata = {
        "model": "EfficientNet-B0",
        "num_classes": num_classes,
        "img_size": IMG_SIZE,
        "mean": MEAN,
        "std": STD,
        "class_names": class_names,
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Model metadata saved to: {meta_path}")

    return onnx_path


def main():
    args = get_args()
    device = torch.device("cpu")
    print(f"Using device: {device}")
    print(f"Batch size: {args.batch_size}, Workers: {args.workers}")

    # Load datasets
    train_transform, eval_transform = get_transforms()

    train_dir = DATA_DIR / "train"
    val_dir = DATA_DIR / "val"
    test_dir = DATA_DIR / "test"

    if not train_dir.exists():
        print(f"ERROR: Training data not found at {train_dir}")
        print("Run preprocess_for_training.py first.")
        return

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=eval_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=eval_transform)

    num_classes = len(train_dataset.classes)
    class_names = train_dataset.classes
    print(f"\nClasses ({num_classes}): {class_names}")

    # Print class distribution
    train_targets = [s[1] for s in train_dataset.samples]
    train_counts = Counter(train_targets)
    print("\nTraining set class distribution:")
    for cls_idx, name in enumerate(class_names):
        count = train_counts.get(cls_idx, 0)
        print(f"  {name}: {count}")

    # Data loaders with weighted sampling
    train_sampler = get_weighted_sampler(train_dataset)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers,
        pin_memory=False, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers
    )

    print(f"\nDataset sizes: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    print(f"Batches per epoch: {len(train_loader)}")

    # Build model
    model = build_model(num_classes).to(device)
    class_weights = get_class_weights(train_dataset, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    best_epoch = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # Live progress file for training_monitor.py GUI
    progress_path = MODEL_DIR / "training_progress.json"
    progress_data = {
        "model": "EfficientNet-B0",
        "dataset_size": len(train_dataset) + len(val_dataset) + len(test_dataset),
        "total_epochs": total_epochs,
        "epochs_frozen": args.epochs_frozen,
        "best_val_acc": 0.0,
        "best_epoch": 0,
        "status": "training",
        "epochs": [],
    }

    def save_progress():
        try:
            with open(progress_path, "w") as pf:
                json.dump(progress_data, pf, indent=2)
        except IOError:
            pass

    total_epochs = args.epochs_frozen + args.epochs_finetune

    # ── Phase 1: Train classifier head only ────────────────────────────
    print("\n" + "=" * 70)
    print(f"Phase 1: Training classifier head ({args.epochs_frozen} epochs)")
    print("=" * 70)

    freeze_backbone(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total_params:,}")

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr_head, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_frozen)

    for epoch in range(1, args.epochs_frozen + 1):
        print(f"\n--- Epoch {epoch}/{total_epochs} (Phase 1: Head) ---")
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, total_epochs
        )
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}%")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "class_names": class_names,
                "num_classes": num_classes,
            }, MODEL_DIR / "best_model.pth")
            print(f"  -> New best model saved! (Val Acc: {val_acc:.1f}%)")

        # Update live progress
        progress_data["epochs"].append({
            "epoch": epoch, "phase": "head",
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 1),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 1),
        })
        progress_data["best_val_acc"] = round(best_val_acc, 1)
        progress_data["best_epoch"] = best_epoch
        save_progress()

    # ── Phase 2: Fine-tune full model ──────────────────────────────────
    print("\n" + "=" * 70)
    print(f"Phase 2: Fine-tuning full model ({args.epochs_finetune} epochs)")
    print("=" * 70)

    unfreeze_backbone(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,} / {total_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr_finetune, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_finetune)

    patience_counter = 0
    for epoch in range(args.epochs_frozen + 1, total_epochs + 1):
        print(f"\n--- Epoch {epoch}/{total_epochs} (Phase 2: Fine-tune) ---")
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, total_epochs
        )
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}%")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "class_names": class_names,
                "num_classes": num_classes,
            }, MODEL_DIR / "best_model.pth")
            print(f"  -> New best model saved! (Val Acc: {val_acc:.1f}%)")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n  Early stopping triggered after {args.patience} epochs without improvement")
                progress_data["status"] = "early_stopped"
                save_progress()
                break

        # Update live progress
        progress_data["epochs"].append({
            "epoch": epoch, "phase": "finetune",
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 1),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 1),
        })
        progress_data["best_val_acc"] = round(best_val_acc, 1)
        progress_data["best_epoch"] = best_epoch
        save_progress()

    # ── Evaluation on test set ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Evaluating best model on test set")
    print("=" * 70)

    # Load best model
    checkpoint = torch.load(MODEL_DIR / "best_model.pth", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded best model from epoch {checkpoint['epoch']} (Val Acc: {checkpoint['val_acc']:.1f}%)")

    test_loss, test_acc, test_preds, test_targets = evaluate(
        model, test_loader, criterion, device
    )

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.1f}%")

    # Classification report
    report = classification_report(
        test_targets, test_preds, target_names=class_names, digits=3
    )
    print(f"\nClassification Report:\n{report}")

    # Confusion matrix
    cm = confusion_matrix(test_targets, test_preds)
    print("Confusion Matrix:")
    print(cm)

    # Save report
    report_path = MODEL_DIR / "evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Dental Disease Classification - Evaluation Report\n")
        f.write(f"{'=' * 60}\n\n")
        f.write(f"Model: EfficientNet-B0\n")
        f.write(f"Best Epoch: {checkpoint['epoch']}\n")
        f.write(f"Best Val Accuracy: {checkpoint['val_acc']:.1f}%\n")
        f.write(f"Test Accuracy: {test_acc:.1f}%\n\n")
        f.write(f"Classification Report:\n{report}\n\n")
        f.write(f"Confusion Matrix:\n{cm}\n\n")
        f.write(f"Training History:\n")
        for i, (tl, ta, vl, va) in enumerate(zip(
            history["train_loss"], history["train_acc"],
            history["val_loss"], history["val_acc"]
        )):
            f.write(f"  Epoch {i + 1}: train_loss={tl:.4f}, train_acc={ta:.1f}%, "
                    f"val_loss={vl:.4f}, val_acc={va:.1f}%\n")

    print(f"\nEvaluation report saved to: {report_path}")

    # Update progress as completed
    progress_data["status"] = "completed"
    progress_data["test_accuracy"] = round(test_acc, 1)
    save_progress()

    # Save training history
    history_path = MODEL_DIR / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # ── Export to ONNX ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Exporting model to ONNX for DirectML inference")
    print("=" * 70)

    onnx_path = export_onnx(model, num_classes, class_names, device)

    # Verify ONNX model
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"Available ONNX Runtime providers: {providers}")

        # Try DirectML first, fall back to CPU
        if "DmlExecutionProvider" in providers:
            session = ort.InferenceSession(str(onnx_path), providers=["DmlExecutionProvider"])
            print("ONNX model loaded with DirectML (AMD GPU) provider")
        else:
            session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            print("ONNX model loaded with CPU provider")

        # Test inference
        dummy = np.random.randn(1, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)
        result = session.run(None, {"input": dummy})
        print(f"ONNX inference test: output shape = {result[0].shape}")
        print("ONNX model verified successfully!")
    except Exception as e:
        print(f"ONNX verification warning: {e}")

    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"Best model: {MODEL_DIR / 'best_model.pth'}")
    print(f"ONNX model: {MODEL_DIR / 'dental_classifier.onnx'}")
    print(f"Metadata:   {MODEL_DIR / 'model_metadata.json'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
