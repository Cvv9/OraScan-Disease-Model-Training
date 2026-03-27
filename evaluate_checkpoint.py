"""Evaluate a saved model checkpoint on the test set and export ONNX."""

import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report, confusion_matrix

DATA_DIR = Path(r"E:\ECG and Dental Images\Dental data set\Training_Ready")
MODEL_DIR = Path(r"E:\ECG and Dental Images\Dental data set\Models")
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def build_model(num_classes):
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes),
    )
    return model


def main():
    device = torch.device("cpu")
    eval_transform = transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.14)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    test_dataset = datasets.ImageFolder(DATA_DIR / "test", transform=eval_transform)
    num_classes = len(test_dataset.classes)
    class_names = test_dataset.classes
    print(f"Classes ({num_classes}): {class_names}")
    print(f"Test samples: {len(test_dataset)}")

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    # Load checkpoint
    ckpt_path = MODEL_DIR / "best_model.pth"
    checkpoint = torch.load(ckpt_path, weights_only=False, map_location=device)
    print(f"\nCheckpoint from epoch {checkpoint['epoch']}, Val Acc: {checkpoint['val_acc']:.1f}%")

    model = build_model(num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Evaluate
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    test_acc = 100. * correct / total
    test_loss = running_loss / total
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.1f}%")

    report = classification_report(
        all_targets, all_preds, target_names=class_names, digits=3
    )
    print(f"\nClassification Report:\n{report}")

    cm = confusion_matrix(all_targets, all_preds)
    print(f"Confusion Matrix:\n{cm}")

    # Save evaluation report
    report_path = MODEL_DIR / "evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write("Dental Disease Classification - Evaluation Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: EfficientNet-B0\n")
        f.write(f"Best Epoch: {checkpoint['epoch']}\n")
        f.write(f"Best Val Accuracy: {checkpoint['val_acc']:.1f}%\n")
        f.write(f"Test Accuracy: {test_acc:.1f}%\n\n")
        f.write(f"Classification Report:\n{report}\n\n")
        f.write(f"Confusion Matrix:\n{cm}\n")
    print(f"\nReport saved to: {report_path}")

    # Export ONNX
    print("\nExporting ONNX model...")
    onnx_path = MODEL_DIR / "dental_classifier.onnx"
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
    torch.onnx.export(
        model, dummy_input, str(onnx_path),
        export_params=True, opset_version=17,
        do_constant_folding=True,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"ONNX exported to: {onnx_path}")

    # Save metadata
    meta_path = MODEL_DIR / "model_metadata.json"
    metadata = {
        "model": "EfficientNet-B0",
        "num_classes": num_classes,
        "img_size": IMG_SIZE,
        "mean": MEAN,
        "std": STD,
        "class_names": list(class_names),
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {meta_path}")

    # Verify ONNX
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        provider = "DmlExecutionProvider" if "DmlExecutionProvider" in providers else "CPUExecutionProvider"
        session = ort.InferenceSession(str(onnx_path), providers=[provider])
        dummy = np.random.randn(1, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)
        result = session.run(None, {"input": dummy})
        print(f"ONNX verified with {provider}: output shape = {result[0].shape}")
    except Exception as e:
        print(f"ONNX verification warning: {e}")


if __name__ == "__main__":
    main()
