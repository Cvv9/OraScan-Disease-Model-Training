"""
Augment and expand the dental disease dataset for underrepresented classes.

Two strategies:
1. Crop YOLO detection bounding boxes into classification images
   (Gingivitis intraoral, Calculus)
2. Offline augmentation for tiny classes to reach a minimum sample count

This script adds images to By_Disease/<class>/classification_augmented/
Then re-run preprocess_for_training.py to rebuild Training_Ready splits.

Usage:
    python augment_and_expand.py
    python augment_and_expand.py --min-samples 500 --aug-factor 8
"""

import argparse
import hashlib
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

# ── Paths ──────────────────────────────────────────────────────────────
BY_DISEASE = Path(r"E:\ECG and Dental Images\Dental data set\By_Disease")
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Classes that need augmentation (train count < 500)
SMALL_CLASSES = {
    "08_Mucocele": 500,
    "09_Periodontal_Disease": 500,
    "10_Tooth_Crack": 500,
    "04_Calculus": 800,
}

# YOLO detection sources to crop into classification images
YOLO_CROP_SOURCES = {
    "02_Gingivitis": [
        "detection_yolo_Intraoral",
    ],
}

# MOD sources to selectively include for tiny classes only
MOD_INCLUDE_CLASSES = [
    "08_Mucocele",  # 48 -> 135 images (critical for this tiny class)
]


def get_args():
    parser = argparse.ArgumentParser(description="Augment dental dataset")
    parser.add_argument("--min-samples", type=int, default=500,
                        help="Minimum training samples target per class")
    parser.add_argument("--aug-factor", type=int, default=8,
                        help="Max augmentation multiplier per source image")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def is_image(path):
    return path.suffix.lower() in IMG_EXTENSIONS


def file_hash(path):
    """Quick hash for deduplication."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read(65536))
    return h.hexdigest()


# ── Augmentation transforms ───────────────────────────────────────────

def augment_image(img, rng):
    """Apply random augmentation transforms to a PIL image.

    Returns a new augmented image.
    """
    # Random horizontal flip
    if rng.random() < 0.5:
        img = ImageOps.mirror(img)

    # Random vertical flip
    if rng.random() < 0.2:
        img = ImageOps.flip(img)

    # Random rotation (-30 to +30 degrees)
    angle = rng.uniform(-30, 30)
    img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=(0, 0, 0))

    # Random brightness (0.7 - 1.3)
    factor = rng.uniform(0.7, 1.3)
    img = ImageEnhance.Brightness(img).enhance(factor)

    # Random contrast (0.7 - 1.3)
    factor = rng.uniform(0.7, 1.3)
    img = ImageEnhance.Contrast(img).enhance(factor)

    # Random saturation (0.7 - 1.4)
    factor = rng.uniform(0.7, 1.4)
    img = ImageEnhance.Color(img).enhance(factor)

    # Random sharpness
    if rng.random() < 0.3:
        factor = rng.uniform(0.5, 2.0)
        img = ImageEnhance.Sharpness(img).enhance(factor)

    # Random Gaussian blur
    if rng.random() < 0.2:
        radius = rng.uniform(0.5, 1.5)
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))

    # Random crop and resize (zoom effect)
    if rng.random() < 0.5:
        w, h = img.size
        crop_ratio = rng.uniform(0.75, 0.95)
        new_w, new_h = int(w * crop_ratio), int(h * crop_ratio)
        left = rng.randint(0, w - new_w)
        top = rng.randint(0, h - new_h)
        img = img.crop((left, top, left + new_w, top + new_h))
        img = img.resize((w, h), Image.LANCZOS)

    return img


# ── YOLO crop extraction ──────────────────────────────────────────────

def parse_yolo_label(label_path, img_w, img_h):
    """Parse YOLO format label file into bounding boxes.

    Returns list of (class_id, x1, y1, x2, y2) in pixel coordinates.
    """
    boxes = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            # Convert normalized center format to pixel coordinates
            x1 = int((cx - bw / 2) * img_w)
            y1 = int((cy - bh / 2) * img_h)
            x2 = int((cx + bw / 2) * img_w)
            y2 = int((cy + bh / 2) * img_h)
            # Clamp to image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_w, x2)
            y2 = min(img_h, y2)
            boxes.append((cls_id, x1, y1, x2, y2))
    return boxes


def crop_yolo_detections(disease_folder, yolo_subfolder, output_dir):
    """Crop bounding box regions from YOLO detection images.

    Returns number of crops created.
    """
    yolo_dir = BY_DISEASE / disease_folder / yolo_subfolder
    if not yolo_dir.exists():
        print(f"  [SKIP] {yolo_dir} not found")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    min_crop_size = 32  # Minimum crop dimension in pixels

    for split in ["train", "val", "test"]:
        img_dir = yolo_dir / split / "images"
        lbl_dir = yolo_dir / split / "labels"
        if not img_dir.exists() or not lbl_dir.exists():
            continue

        for img_file in img_dir.iterdir():
            if not is_image(img_file):
                continue

            # Find matching label
            label_file = lbl_dir / f"{img_file.stem}.txt"
            if not label_file.exists():
                continue

            try:
                img = Image.open(img_file).convert("RGB")
                w, h = img.size
                boxes = parse_yolo_label(label_file, w, h)

                for i, (cls_id, x1, y1, x2, y2) in enumerate(boxes):
                    crop_w = x2 - x1
                    crop_h = y2 - y1
                    if crop_w < min_crop_size or crop_h < min_crop_size:
                        continue

                    # Add 15% padding
                    pad_w = int(crop_w * 0.15)
                    pad_h = int(crop_h * 0.15)
                    x1p = max(0, x1 - pad_w)
                    y1p = max(0, y1 - pad_h)
                    x2p = min(w, x2 + pad_w)
                    y2p = min(h, y2 + pad_h)

                    crop = img.crop((x1p, y1p, x2p, y2p))
                    crop_name = f"yolo_{img_file.stem}_box{i}.jpg"
                    crop.save(output_dir / crop_name, "JPEG", quality=95)
                    count += 1
            except Exception as e:
                print(f"  [WARN] Failed to process {img_file.name}: {e}")

    return count


# ── MOD selective inclusion ────────────────────────────────────────────

def copy_mod_images(disease_folder, output_dir):
    """Copy classification_mod images to classification_augmented."""
    mod_dir = BY_DISEASE / disease_folder / "classification_mod"
    if not mod_dir.exists():
        print(f"  [SKIP] {mod_dir} not found")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for f in mod_dir.iterdir():
        if is_image(f):
            dst = output_dir / f"mod_{f.name}"
            if not dst.exists():
                img = Image.open(f).convert("RGB")
                img.save(dst, "JPEG", quality=95)
                count += 1
    return count


# ── Offline augmentation ──────────────────────────────────────────────

def collect_source_images(disease_folder):
    """Collect all classification images for a disease (including new crops)."""
    disease_dir = BY_DISEASE / disease_folder
    images = []
    classification_dirs = [
        "classification", "classification_csv", "classification_new",
        "classification_kaggle", "classification_gdrive",
        "classification_merged", "classification_dentalai",
        "classification_augmented",  # includes MOD copies and YOLO crops
    ]
    for src_name in classification_dirs:
        src_dir = disease_dir / src_name
        if src_dir.exists():
            for f in src_dir.rglob("*"):
                if f.is_file() and is_image(f):
                    images.append(f)

    # Also smart_om_ folders
    for sub in disease_dir.iterdir():
        if sub.is_dir() and sub.name.startswith("smart_om_"):
            for f in sub.rglob("*"):
                if f.is_file() and is_image(f) and "annotation" not in str(f).lower():
                    images.append(f)

    return images


def deduplicate(image_paths):
    """Remove duplicates by content hash."""
    seen = set()
    unique = []
    for p in image_paths:
        try:
            h = file_hash(p)
            if h not in seen:
                seen.add(h)
                unique.append(p)
        except Exception:
            unique.append(p)
    return unique


def generate_augmented_images(disease_folder, target_count, aug_factor, rng):
    """Generate augmented images to reach target_count.

    Saves to classification_augmented/aug_*.jpg
    """
    source_images = collect_source_images(disease_folder)
    source_images = deduplicate(source_images)

    current_count = len(source_images)
    if current_count >= target_count:
        print(f"  Already have {current_count} images (target: {target_count}), skipping augmentation")
        return 0

    needed = target_count - current_count
    # Cap augmentations per source image
    per_image = min(aug_factor, max(1, needed // current_count + 1))
    actual_needed = needed

    output_dir = BY_DISEASE / disease_folder / "classification_augmented"
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    source_cycle = source_images * per_image  # Repeat source list
    rng.shuffle(source_cycle)

    for src_path in source_cycle:
        if count >= actual_needed:
            break

        try:
            img = Image.open(src_path).convert("RGB")
            aug_img = augment_image(img, rng)
            aug_name = f"aug_{src_path.stem}_{count:04d}.jpg"
            aug_img.save(output_dir / aug_name, "JPEG", quality=90)
            count += 1
        except Exception as e:
            print(f"  [WARN] Augmentation failed for {src_path.name}: {e}")

    return count


# ── Main ───────────────────────────────────────────────────────────────

def main():
    args = get_args()
    rng = random.Random(args.seed)

    print("=" * 70)
    print("Dataset Augmentation and Expansion")
    print("=" * 70)

    # Step 1: Crop YOLO detections
    print("\n--- Step 1: Cropping YOLO detection bounding boxes ---")
    for disease_folder, yolo_sources in YOLO_CROP_SOURCES.items():
        disease_name = disease_folder.split("_", 1)[1]
        output_dir = BY_DISEASE / disease_folder / "classification_augmented"
        for yolo_sub in yolo_sources:
            print(f"\n  {disease_name} <- {yolo_sub}")
            n = crop_yolo_detections(disease_folder, yolo_sub, output_dir)
            print(f"    Cropped {n} bounding box regions")

    # Step 2: Copy MOD images for tiny classes
    print("\n--- Step 2: Selective MOD inclusion for tiny classes ---")
    for disease_folder in MOD_INCLUDE_CLASSES:
        disease_name = disease_folder.split("_", 1)[1]
        output_dir = BY_DISEASE / disease_folder / "classification_augmented"
        print(f"\n  {disease_name} <- classification_mod")
        n = copy_mod_images(disease_folder, output_dir)
        print(f"    Copied {n} MOD images")

    # Step 3: Offline augmentation for small classes
    print("\n--- Step 3: Offline augmentation for small classes ---")
    for disease_folder, target in SMALL_CLASSES.items():
        disease_name = disease_folder.split("_", 1)[1]
        effective_target = max(target, args.min_samples)
        print(f"\n  {disease_name} (target: {effective_target})")

        # Count current images
        sources = collect_source_images(disease_folder)
        sources = deduplicate(sources)
        print(f"    Source images (deduplicated): {len(sources)}")

        n = generate_augmented_images(disease_folder, effective_target, args.aug_factor, rng)
        print(f"    Generated {n} augmented images")

    # Summary
    print("\n" + "=" * 70)
    print("Summary - classification_augmented folders:")
    print("=" * 70)
    for disease_dir in sorted(BY_DISEASE.iterdir()):
        if not disease_dir.is_dir() or disease_dir.name.startswith("_"):
            continue
        aug_dir = disease_dir / "classification_augmented"
        if aug_dir.exists():
            count = len([f for f in aug_dir.iterdir() if is_image(f)])
            print(f"  {disease_dir.name}: {count} augmented images")

    print("\nDone! Now run preprocess_for_training.py to rebuild Training_Ready splits.")
    print("Then run train_classifier.py --gui to train with the expanded dataset.")


if __name__ == "__main__":
    main()
