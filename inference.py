"""
Dental disease inference using ONNX Runtime with DirectML (AMD GPU).

Takes an image path and returns disease predictions with confidence scores.
Uses the exported ONNX model from train_classifier.py.

Usage:
  python inference.py path/to/image.jpg
  python inference.py path/to/image.jpg --top-k 3
  python inference.py --folder path/to/folder/ --output results.csv
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

MODEL_DIR = Path(r"E:\ECG and Dental Images\Dental data set\Models")


def load_model():
    """Load ONNX model with DirectML (AMD GPU) or CPU fallback."""
    onnx_path = MODEL_DIR / "dental_classifier.onnx"
    meta_path = MODEL_DIR / "model_metadata.json"

    if not onnx_path.exists():
        raise FileNotFoundError(f"Model not found at {onnx_path}. Run train_classifier.py first.")

    with open(meta_path) as f:
        metadata = json.load(f)

    providers = ort.get_available_providers()
    if "DmlExecutionProvider" in providers:
        session = ort.InferenceSession(str(onnx_path), providers=["DmlExecutionProvider"])
        print("Using DirectML (AMD GPU) for inference")
    else:
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        print("Using CPU for inference")

    return session, metadata


def preprocess_image(image_path, img_size, mean, std):
    """Preprocess image for model input."""
    img = Image.open(image_path).convert("RGB")

    # Resize with aspect ratio preservation + center crop
    w, h = img.size
    scale = int(img_size * 1.14) / min(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)

    # Center crop
    left = (new_w - img_size) // 2
    top = (new_h - img_size) // 2
    img = img.crop((left, top, left + img_size, top + img_size))

    # Convert to numpy and normalize
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - np.array(mean)) / np.array(std)
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    arr = np.expand_dims(arr, 0)  # Add batch dimension

    return arr


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def predict(session, metadata, image_path, top_k=5):
    """Run inference on a single image."""
    img_size = metadata["img_size"]
    mean = metadata["mean"]
    std = metadata["std"]
    class_names = metadata["class_names"]

    input_data = preprocess_image(image_path, img_size, mean, std)

    start = time.time()
    outputs = session.run(None, {"input": input_data})
    elapsed = (time.time() - start) * 1000  # ms

    logits = outputs[0][0]
    probs = softmax(logits)

    top_indices = np.argsort(probs)[::-1][:top_k]
    results = []
    for idx in top_indices:
        results.append({
            "class": class_names[idx],
            "confidence": float(probs[idx]),
            "class_index": int(idx),
        })

    return results, elapsed


def main():
    parser = argparse.ArgumentParser(description="Dental disease inference")
    parser.add_argument("image", nargs="?", help="Path to image file")
    parser.add_argument("--folder", help="Process all images in folder")
    parser.add_argument("--output", help="Save results to CSV file")
    parser.add_argument("--top-k", type=int, default=3, help="Number of top predictions")
    args = parser.parse_args()

    session, metadata = load_model()
    print(f"Model: {metadata['model']}, Classes: {metadata['num_classes']}")
    print(f"Classes: {metadata['class_names']}\n")

    if args.image:
        # Single image inference
        results, elapsed = predict(session, metadata, args.image, args.top_k)
        print(f"Image: {args.image}")
        print(f"Inference time: {elapsed:.1f}ms")
        print(f"Predictions:")
        for r in results:
            bar = "#" * int(r["confidence"] * 40)
            print(f"  {r['class']:<25} {r['confidence']:>6.1%}  {bar}")

    elif args.folder:
        # Batch inference
        folder = Path(args.folder)
        image_files = [f for f in folder.rglob("*")
                       if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]

        all_results = []
        total_time = 0
        for img_path in sorted(image_files):
            results, elapsed = predict(session, metadata, img_path, args.top_k)
            total_time += elapsed
            top = results[0]
            print(f"  {img_path.name:<40} -> {top['class']:<20} ({top['confidence']:.1%}) [{elapsed:.0f}ms]")
            all_results.append({
                "file": str(img_path),
                "prediction": top["class"],
                "confidence": top["confidence"],
            })

        print(f"\nProcessed {len(image_files)} images in {total_time:.0f}ms "
              f"({total_time / len(image_files):.0f}ms/image avg)")

        if args.output:
            with open(args.output, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["file", "prediction", "confidence"])
                writer.writeheader()
                writer.writerows(all_results)
            print(f"Results saved to: {args.output}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
