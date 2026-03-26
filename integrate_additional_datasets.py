"""
Integrate downloaded additional datasets into the By_Disease folder structure.

Handles:
  1. Dentalai dataset: Extract crack crops from segmentation masks (pixel=3)
  2. Oral Diseases (Kaggle): Map Calculus, Gingivitis (original only, skip augmented)
  3. MOD dataset: Map MC->Mucocele, CaS->Calculus, Gum->Gingivitis, OLP->Periodontal_Disease
  4. Google Drive calculus: Map calculus folder images
"""

import hashlib
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

DOWNLOADS = Path(r"E:\ECG and Dental Images\Dental data set\Downloads\additional")
BY_DISEASE = Path(r"E:\ECG and Dental Images\Dental data set\By_Disease")

# Our standard folder names (with numeric prefixes)
DISEASE_FOLDERS = {
    "Calculus": "04_Calculus",
    "Gingivitis": "02_Gingivitis",
    "Mucocele": "08_Mucocele",
    "Tooth_Crack": "10_Tooth_Crack",
    "Periodontal_Disease": "09_Periodontal_Disease",
    "Caries": "01_Caries",
    "Healthy": "11_Healthy",
    "Mouth_Ulcer": "03_Mouth_Ulcer",
    "Hypodontia": "07_Hypodontia",
    "Oral_Cancer": "06_Oral_Cancer",
    "Tooth_Discoloration": "05_Tooth_Discoloration",
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def file_hash(path, chunk_size=65536):
    """Quick hash for deduplication."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read(chunk_size))
    return h.hexdigest()[:12]


def copy_images(src_dir, disease_name, subfolder_name, max_images=None):
    """Copy images from src_dir to By_Disease/{disease_folder}/{subfolder_name}/."""
    if not src_dir.exists():
        return 0

    folder_name = DISEASE_FOLDERS.get(disease_name, disease_name)
    dest_dir = BY_DISEASE / folder_name / subfolder_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Get existing hashes to avoid duplicates
    existing_hashes = set()
    disease_root = BY_DISEASE / folder_name
    for f in disease_root.rglob("*"):
        if f.suffix.lower() in IMG_EXTS:
            try:
                existing_hashes.add(file_hash(f))
            except Exception:
                pass

    count = 0
    for img_file in sorted(src_dir.rglob("*")):
        if img_file.suffix.lower() not in IMG_EXTS:
            continue
        if max_images and count >= max_images:
            break

        h = file_hash(img_file)
        if h in existing_hashes:
            continue

        dest_name = f"{subfolder_name}_{h}{img_file.suffix.lower()}"
        dest_path = dest_dir / dest_name
        if not dest_path.exists():
            shutil.copy2(img_file, dest_path)
            existing_hashes.add(h)
            count += 1

    return count


def extract_crack_crops_from_dentalai():
    """Extract tooth crack crops from Dentalai segmentation masks."""
    print("\n--- Extracting Tooth Crack crops from Dentalai ---")

    base = DOWNLOADS / "dentalai" / "dentalai-2"
    if not base.exists():
        print("  Dentalai not found, skipping")
        return 0

    folder_name = DISEASE_FOLDERS["Tooth_Crack"]
    dest_dir = BY_DISEASE / folder_name / "classification_dentalai"
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Get existing hashes
    existing_hashes = set()
    disease_root = BY_DISEASE / folder_name
    for f in disease_root.rglob("*"):
        if f.suffix.lower() in IMG_EXTS:
            try:
                existing_hashes.add(file_hash(f))
            except Exception:
                pass

    total_crops = 0
    CRACK_PIXEL = 3
    MIN_CRACK_PIXELS = 50

    for split in ["train", "valid", "test"]:
        split_dir = base / split
        if not split_dir.exists():
            continue

        # Find all mask files
        masks = sorted(split_dir.glob("*_mask.png"))
        for mask_path in masks:
            # Find corresponding image
            img_name = mask_path.name.replace("_mask.png", ".jpg")
            img_path = split_dir / img_name
            if not img_path.exists():
                continue

            try:
                mask = np.array(Image.open(mask_path))
                crack_pixels = np.where(mask == CRACK_PIXEL)

                if len(crack_pixels[0]) < MIN_CRACK_PIXELS:
                    continue

                # Get bounding box with padding
                y_min, y_max = crack_pixels[0].min(), crack_pixels[0].max()
                x_min, x_max = crack_pixels[1].min(), crack_pixels[1].max()

                h = y_max - y_min
                w = x_max - x_min
                pad_y = max(int(h * 0.2), 10)
                pad_x = max(int(w * 0.2), 10)

                img = Image.open(img_path)
                img_w, img_h = img.size

                crop_box = (
                    max(0, x_min - pad_x),
                    max(0, y_min - pad_y),
                    min(img_w, x_max + pad_x),
                    min(img_h, y_max + pad_y),
                )

                crop = img.crop(crop_box)

                # Check minimum size
                if crop.size[0] < 32 or crop.size[1] < 32:
                    continue

                # Save crop
                crop_path = dest_dir / f"dentalai_crack_{split}_{mask_path.stem}.jpg"
                if not crop_path.exists():
                    crop.save(crop_path, quality=95)
                    total_crops += 1

            except Exception as e:
                continue

    print(f"  Extracted {total_crops} crack crops")
    return total_crops


def integrate_mod_dataset():
    """Integrate MOD (Mouth and Oral Diseases) dataset."""
    print("\n--- Integrating MOD Dataset ---")

    base = DOWNLOADS / "mod_dataset"
    if not base.exists():
        print("  MOD dataset not found, skipping")
        return {}

    # MOD class mapping
    mod_mapping = {
        "MC": "Mucocele",        # Mucocele - THIS IS KEY
        "CaS": "Calculus",       # Calculus
        "Gum": "Gingivitis",     # Gum disease = Gingivitis
        "OLP": "Periodontal_Disease",  # Oral Lichen Planus -> Periodontal
        "OC": "Oral_Cancer",     # Oral Cancer
        "CoS": "Mouth_Ulcer",    # Cold Sore -> Mouth Ulcer (closest match)
        "OT": None,              # Other - skip
    }

    stats = {}
    for split in ["Training", "Testing", "Validation"]:
        split_dir = base / split
        if not split_dir.exists():
            continue
        for cls_dir in sorted(split_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            disease = mod_mapping.get(cls_dir.name)
            if not disease:
                continue

            count = copy_images(cls_dir, disease, "classification_mod")
            stats[disease] = stats.get(disease, 0) + count

    for disease, count in sorted(stats.items()):
        print(f"  {disease}: +{count}")
    return stats


def integrate_oral_diseases_kaggle():
    """Integrate Kaggle Oral Diseases - ORIGINAL images only (skip augmented)."""
    print("\n--- Integrating Kaggle Oral Diseases (originals only) ---")

    base = DOWNLOADS / "oral_diseases"
    if not base.exists():
        print("  Oral Diseases not found, skipping")
        return {}

    # Map folder names to our diseases
    folder_mapping = {
        "Calculus": ("Calculus", "Calculus"),
        "Gingivitis": ("Gingivitis", "Gingivitis"),
        "hypodontia": ("Hypodontia", "hypodontia"),
        "Mouth Ulcer": ("Mouth_Ulcer", "Mouth Ulcer"),
        "Tooth Discoloration": ("Tooth_Discoloration", "Tooth Discoloration"),
        "Data caries": ("Caries", "Data caries"),
    }

    stats = {}
    for folder_name, (disease, _) in folder_mapping.items():
        # Find the actual image directory (handle double nesting)
        src = base / folder_name
        if not src.exists():
            continue

        # Look for original dataset subfolder (skip augmented)
        original_dirs = []
        for d in src.rglob("*"):
            if d.is_dir() and "original" in d.name.lower():
                original_dirs.append(d)

        if original_dirs:
            for orig_dir in original_dirs:
                count = copy_images(orig_dir, disease, "classification_kaggle")
                stats[disease] = stats.get(disease, 0) + count
        else:
            # If no 'original' subfolder, take all images but from deepest folder
            deepest = src
            while True:
                subdirs = [d for d in deepest.iterdir() if d.is_dir()]
                if len(subdirs) == 1:
                    deepest = subdirs[0]
                else:
                    break
            count = copy_images(deepest, disease, "classification_kaggle")
            stats[disease] = stats.get(disease, 0) + count

    for disease, count in sorted(stats.items()):
        print(f"  {disease}: +{count}")
    return stats


def integrate_gdrive_calculus():
    """Integrate Google Drive calculus dataset."""
    print("\n--- Integrating Google Drive Calculus ---")

    base = DOWNLOADS / "gdrive_calculus"
    if not base.exists():
        print("  Google Drive calculus not found, skipping")
        return {}

    stats = {}
    # Look for calculus folders
    for root, dirs, files in os.walk(base):
        folder_name = Path(root).name.lower()
        if "calculus" in folder_name:
            count = copy_images(Path(root), "Calculus", "classification_gdrive")
            stats["Calculus"] = stats.get("Calculus", 0) + count

    if stats:
        for disease, count in sorted(stats.items()):
            print(f"  {disease}: +{count}")
    else:
        print("  No calculus images found")
    return stats


def print_final_counts():
    """Print updated By_Disease counts."""
    print("\n\n" + "=" * 60)
    print("FINAL By_Disease Image Counts")
    print("=" * 60)

    priority = {"Mucocele", "Tooth_Crack", "Periodontal_Disease", "Calculus", "Gingivitis"}

    for disease_dir in sorted(BY_DISEASE.iterdir()):
        if not disease_dir.is_dir() or disease_dir.name.startswith("."):
            continue
        if disease_dir.name.startswith("_"):
            continue

        # Count classification-suitable images only
        class_count = 0
        for sub in disease_dir.iterdir():
            if sub.is_dir() and "classification" in sub.name.lower():
                class_count += sum(1 for f in sub.rglob("*") if f.suffix.lower() in IMG_EXTS)

        total = sum(1 for f in disease_dir.rglob("*") if f.suffix.lower() in IMG_EXTS)

        # Check if this is a priority class
        is_priority = any(p.lower() in disease_dir.name.lower() for p in priority)
        marker = " <<<" if is_priority else ""
        print(f"  {disease_dir.name:<30} {total:>6} total  ({class_count:>5} classification){marker}")


def main():
    print("=" * 60)
    print("Integrating Additional Datasets into By_Disease")
    print("=" * 60)

    # Print current counts for priority classes
    print("\nCurrent classification-ready counts:")
    for disease, folder in sorted(DISEASE_FOLDERS.items()):
        disease_dir = BY_DISEASE / folder
        if disease_dir.exists():
            class_count = 0
            for sub in disease_dir.iterdir():
                if sub.is_dir() and "classification" in sub.name.lower():
                    class_count += sum(1 for f in sub.rglob("*") if f.suffix.lower() in IMG_EXTS)
            if disease in {"Mucocele", "Tooth_Crack", "Periodontal_Disease", "Calculus", "Gingivitis"}:
                print(f"  {disease:<25} {class_count:>5} images")

    # Integrate each source
    extract_crack_crops_from_dentalai()
    integrate_mod_dataset()
    integrate_oral_diseases_kaggle()
    integrate_gdrive_calculus()

    # Final counts
    print_final_counts()


if __name__ == "__main__":
    main()
