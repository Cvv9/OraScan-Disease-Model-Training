"""
Preprocess disease-based dental dataset for classification training.

Collects classification-suitable images from By_Disease folders,
extracts crops from segmentation masks for Tooth_Crack,
deduplicates, resizes, and creates stratified train/val/test splits
in ImageFolder format.

Output: E:\ECG and Dental Images\Dental data set\Training_Ready\
  train/<disease_name>/*.jpg
  val/<disease_name>/*.jpg
  test/<disease_name>/*.jpg
"""

import os
import hashlib
import shutil
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# ── Paths ──────────────────────────────────────────────────────────────
SRC = Path(r"E:\ECG and Dental Images\Dental data set\By_Disease")
DST = Path(r"E:\ECG and Dental Images\Dental data set\Training_Ready")
IMG_SIZE = 224  # EfficientNet-B0 input size
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# ── Disease classes (ordered by folder name) ───────────────────────────
DISEASE_CLASSES = {
    "01_Caries": "Caries",
    "02_Gingivitis": "Gingivitis",
    "03_Mouth_Ulcer": "Mouth_Ulcer",
    "04_Calculus": "Calculus",
    "05_Tooth_Discoloration": "Tooth_Discoloration",
    "06_Oral_Cancer": "Oral_Cancer",
    "07_Hypodontia": "Hypodontia",
    "08_Mucocele": "Mucocele",
    "09_Periodontal_Disease": "Periodontal_Disease",
    "10_Tooth_Crack": "Tooth_Crack",
    "11_Healthy": "Healthy",
}

# Subfolders to collect classification images from
CLASSIFICATION_SOURCES = [
    "classification", "classification_csv",
    "classification_new",
    "classification_kaggle", "classification_gdrive",
    "classification_merged", "classification_dentalai",
]

# Segmentation class map for cropping (pixel value -> disease folder)
SEG_CRACK_VALUE = 3  # Crack class in segmentation masks


def file_hash(path, chunk_size=8192):
    """Quick hash for deduplication (first 64KB)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        data = f.read(65536)
        h.update(data)
    return h.hexdigest()


def is_image(path):
    return path.suffix.lower() in IMG_EXTENSIONS


def collect_classification_images(disease_dir):
    """Collect images from classification and smart_om subfolders."""
    images = []
    for src_name in CLASSIFICATION_SOURCES:
        src_dir = disease_dir / src_name
        if src_dir.exists():
            for f in src_dir.rglob("*"):
                if f.is_file() and is_image(f):
                    images.append(f)

    # Also collect from smart_om_* folders
    for sub in disease_dir.iterdir():
        if sub.is_dir() and sub.name.startswith("smart_om_"):
            for f in sub.rglob("*"):
                if f.is_file() and is_image(f) and "annotation" not in str(f).lower():
                    images.append(f)

    return images


def extract_crack_crops(disease_dir):
    """Extract crack region crops from segmentation masks."""
    crops = []
    seg_dir = disease_dir / "segmentation"
    if not seg_dir.exists():
        return crops

    for split in ["train", "test"]:
        img_dir = seg_dir / split / "images"
        mask_dir = seg_dir / split / "masks"
        if not img_dir or not img_dir.exists() or not mask_dir or not mask_dir.exists():
            continue

        for img_file in img_dir.iterdir():
            if not is_image(img_file):
                continue

            # Find matching mask
            mask_file = None
            for pattern in [f"{img_file.stem}_mask.png", img_file.name, f"{img_file.stem}.png"]:
                candidate = mask_dir / pattern
                if candidate.exists():
                    mask_file = candidate
                    break

            if not mask_file:
                continue

            try:
                mask = np.array(Image.open(mask_file))
                # Find pixels with crack value
                crack_pixels = (mask == SEG_CRACK_VALUE)
                if not crack_pixels.any():
                    continue

                # Get bounding box of crack region
                rows = np.any(crack_pixels, axis=1)
                cols = np.any(crack_pixels, axis=0)
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]

                # Add padding (20% of region size)
                h, w = rmax - rmin, cmax - cmin
                pad_h, pad_w = max(int(h * 0.2), 10), max(int(w * 0.2), 10)
                rmin = max(0, rmin - pad_h)
                rmax = min(mask.shape[0], rmax + pad_h)
                cmin = max(0, cmin - pad_w)
                cmax = min(mask.shape[1], cmax + pad_w)

                crops.append((img_file, (rmin, rmax, cmin, cmax)))
            except Exception as e:
                print(f"  [WARN] Failed to process mask {mask_file.name}: {e}")

    return crops


def deduplicate_images(image_paths):
    """Remove duplicate images based on content hash."""
    seen_hashes = {}
    unique = []
    for path in image_paths:
        try:
            h = file_hash(path)
            if h not in seen_hashes:
                seen_hashes[h] = path
                unique.append(path)
        except Exception:
            unique.append(path)  # Keep if can't hash
    return unique


def resize_and_save(src_path, dst_path, size=IMG_SIZE, crop_box=None):
    """Resize image to square and save as JPEG."""
    try:
        img = Image.open(src_path).convert("RGB")
        if crop_box:
            rmin, rmax, cmin, cmax = crop_box
            img = img.crop((cmin, rmin, cmax, rmax))
        # Resize maintaining aspect ratio with center crop
        w, h = img.size
        scale = size / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        # Center crop to exact size
        left = (new_w - size) // 2
        top = (new_h - size) // 2
        img = img.crop((left, top, left + size, top + size))
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst_path, "JPEG", quality=95)
        return True
    except Exception as e:
        print(f"  [WARN] Failed to process {src_path.name}: {e}")
        return False


def main():
    print("=" * 70)
    print("Preprocessing dental dataset for classification training")
    print(f"Source: {SRC}")
    print(f"Output: {DST}")
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
    print("=" * 70)

    # Collect all images per disease
    all_data = {}  # disease_name -> list of (path, crop_box_or_None)

    for folder_name, display_name in DISEASE_CLASSES.items():
        disease_dir = SRC / folder_name
        if not disease_dir.exists():
            print(f"  [SKIP] {folder_name}: directory not found")
            continue

        images = []

        # Standard classification images
        cls_images = collect_classification_images(disease_dir)
        for img in cls_images:
            images.append((img, None))

        # Special: extract crops from segmentation for Tooth_Crack
        if folder_name == "10_Tooth_Crack":
            crack_crops = extract_crack_crops(disease_dir)
            for img_path, box in crack_crops:
                images.append((img_path, box))
            print(f"  [INFO] Extracted {len(crack_crops)} crack crops from segmentation")

        # Deduplicate (only for non-cropped images)
        regular = [(p, b) for p, b in images if b is None]
        cropped = [(p, b) for p, b in images if b is not None]

        regular_paths = [p for p, _ in regular]
        unique_paths = set(deduplicate_images(regular_paths))
        regular = [(p, None) for p in regular_paths if p in unique_paths]

        all_images = regular + cropped
        all_data[display_name] = all_images
        print(f"  {display_name}: {len(all_images)} images ({len(regular)} unique + {len(cropped)} crops)")

    print()

    # Create stratified train/val/test splits
    print("Creating stratified splits (80/10/10)...")
    split_counts = defaultdict(lambda: defaultdict(int))

    for disease_name, images in all_data.items():
        if len(images) < 3:
            print(f"  [SKIP] {disease_name}: too few images ({len(images)})")
            continue

        paths = list(range(len(images)))

        # Handle very small classes differently
        if len(images) < 20:
            # For very small classes, use 60/20/20 to ensure at least 1 per split
            train_idx, temp_idx = train_test_split(paths, test_size=0.4, random_state=42)
            val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
        else:
            train_idx, temp_idx = train_test_split(paths, test_size=0.2, random_state=42)
            val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

        splits = {"train": train_idx, "val": val_idx, "test": test_idx}

        for split_name, indices in splits.items():
            for i, idx in enumerate(indices):
                img_path, crop_box = images[idx]
                # Create unique filename
                suffix = f"_crop{i}" if crop_box else ""
                dst_name = f"{img_path.stem}{suffix}.jpg"
                dst_path = DST / split_name / disease_name / dst_name

                if resize_and_save(img_path, dst_path, IMG_SIZE, crop_box):
                    split_counts[split_name][disease_name] += 1

    # Print summary
    print("\n" + "=" * 70)
    print("Dataset Summary")
    print("=" * 70)
    print(f"{'Disease':<25} {'Train':>7} {'Val':>7} {'Test':>7} {'Total':>7}")
    print("-" * 55)

    total_train = total_val = total_test = 0
    class_info = []
    for disease_name in sorted(all_data.keys()):
        tr = split_counts["train"].get(disease_name, 0)
        va = split_counts["val"].get(disease_name, 0)
        te = split_counts["test"].get(disease_name, 0)
        tot = tr + va + te
        if tot > 0:
            print(f"  {disease_name:<23} {tr:>7} {va:>7} {te:>7} {tot:>7}")
            class_info.append((disease_name, tr, va, te))
            total_train += tr
            total_val += va
            total_test += te

    print("-" * 55)
    print(f"  {'TOTAL':<23} {total_train:>7} {total_val:>7} {total_test:>7} {total_train + total_val + total_test:>7}")

    # Save class mapping
    class_map_path = DST / "class_mapping.txt"
    with open(class_map_path, "w") as f:
        f.write("# Class index -> Disease name mapping\n")
        f.write("# Used by train_classifier.py\n\n")
        for i, (name, tr, va, te) in enumerate(class_info):
            f.write(f"{i}\t{name}\t{tr}\t{va}\t{te}\n")
    print(f"\nClass mapping saved to: {class_map_path}")

    # Save dataset stats for training script
    stats_path = DST / "dataset_stats.txt"
    with open(stats_path, "w") as f:
        f.write(f"num_classes={len(class_info)}\n")
        f.write(f"img_size={IMG_SIZE}\n")
        f.write(f"train_total={total_train}\n")
        f.write(f"val_total={total_val}\n")
        f.write(f"test_total={total_test}\n")
        for i, (name, tr, va, te) in enumerate(class_info):
            f.write(f"class_{i}={name},{tr},{va},{te}\n")
    print(f"Dataset stats saved to: {stats_path}")

    print("\nDone! Dataset ready for training.")


if __name__ == "__main__":
    main()
