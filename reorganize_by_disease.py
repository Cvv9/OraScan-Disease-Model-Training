"""
Reorganize dental datasets by DISEASE TYPE instead of annotation format.

This script takes the previously organized dataset (format-based structure)
and re-organizes everything by disease, keeping all relevant data (images,
bounding boxes, segmentation masks, captions, etc.) together per disease.

Disease folders created:
  01_Caries, 02_Gingivitis, 03_Mouth_Ulcer, 04_Calculus,
  05_Tooth_Discoloration, 06_Oral_Cancer, 07_Hypodontia,
  08_Mucocele, 09_Periodontal_Disease, 10_Tooth_Crack,
  11_Healthy, _Panoramic_Xray, _Dental_Anatomy, _Raw_Unlabeled
"""

import os
import sys
import json
import csv
import shutil
import time
import numpy as np
from pathlib import Path
from collections import defaultdict

# ── Paths ──────────────────────────────────────────────────────────────
SRC = Path(r"E:\ECG and Dental Images\Dental data set\Organized")
DST = Path(r"E:\ECG and Dental Images\Dental data set\By_Disease")
LOG_FILE = None

# ── Disease mapping ────────────────────────────────────────────────────
# Intraoral_4Class YOLO class IDs -> disease
INTRAORAL_CLASS_MAP = {
    0: "01_Caries",
    1: "03_Mouth_Ulcer",
    2: "05_Tooth_Discoloration",
    3: "02_Gingivitis",
}

# Labelled_Ulcer_Calculus_2Class YOLO class IDs -> disease
ULCER_CALCULUS_CLASS_MAP = {
    0: "03_Mouth_Ulcer",
    1: "04_Calculus",
}

# Segmentation pixel classes -> disease
SEGMENTATION_CLASS_MAP = {
    1: "01_Caries",      # Caries
    2: "01_Caries",      # Cavity (same disease family)
    3: "10_Tooth_Crack",  # Crack
    # 0: background, 4: Tooth (not disease)
}

# OPG COCO 31-class category IDs -> disease
OPG_DISEASE_CATEGORIES = {
    1:  "01_Caries",             # Caries
    7:  "07_Hypodontia",         # Missing teeth
    8:  "01_Caries",             # Periapical lesion (often from caries)
    14: "09_Periodontal_Disease", # Bone Loss
    15: "10_Tooth_Crack",         # Fracture teeth
    21: "09_Periodontal_Disease", # bone defect
    29: "01_Caries",             # Cyst (often periapical, caries-related)
    30: "09_Periodontal_Disease", # Root resorption
}

# CSV label -> disease
CSV_LABEL_MAP = {
    "c": "01_Caries",
    "g": "02_Gingivitis",
    "h": "11_Healthy",
}

stats = defaultdict(int)


def log(msg):
    print(msg)
    if LOG_FILE:
        LOG_FILE.write(msg + "\n")


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_copy(src, dst):
    """Copy file, handle name collisions with suffix."""
    if not src.exists():
        return False
    dst_path = Path(dst)
    if dst_path.exists():
        stem = dst_path.stem
        suffix = dst_path.suffix
        parent = dst_path.parent
        counter = 1
        while dst_path.exists():
            dst_path = parent / f"{stem}_{counter}{suffix}"
            counter += 1
    ensure_dir(dst_path.parent)
    shutil.copy2(src, dst_path)
    return True


def copy_tree_contents(src_dir, dst_dir):
    """Copy all files from src_dir to dst_dir preserving subfolder structure."""
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    count = 0
    if not src_dir.exists():
        return 0
    for f in src_dir.rglob("*"):
        if f.is_file():
            rel = f.relative_to(src_dir)
            dst_file = dst_dir / rel
            ensure_dir(dst_file.parent)
            shutil.copy2(f, dst_file)
            count += 1
    return count


# ═══════════════════════════════════════════════════════════════════════
# PHASE 1: Simple Classification datasets (folder-based)
# ═══════════════════════════════════════════════════════════════════════
def phase1_classification():
    log("\n" + "=" * 70)
    log("PHASE 1: Classification datasets (folder-based)")
    log("=" * 70)

    mapping = {
        "Caries":              ("01_Caries",              True),   # has original/augmented
        "Gingivitis":          ("02_Gingivitis",           False),
        "Mouth_Ulcer":         ("03_Mouth_Ulcer",          True),   # has original/augmented
        "Calculus":            ("04_Calculus",              False),
        "Tooth_Discoloration": ("05_Tooth_Discoloration",  True),   # has original/augmented
        "Oral_Cancer":         ("06_Oral_Cancer",           False),  # has cancer/non_cancer
        "Oral_Lesions_Malignancy": ("06_Oral_Cancer",       False),  # same disease family
        "Hypodontia":          ("07_Hypodontia",            False),
        "Mucocele":            ("08_Mucocele",              False),
        "Periodontal":         ("09_Periodontal_Disease",   False),
        "Healthy":             ("11_Healthy",               False),
    }

    cls_base = SRC / "Classification"

    for folder_name, (disease, has_aug) in mapping.items():
        src_folder = cls_base / folder_name
        if not src_folder.exists():
            log(f"  [SKIP] {folder_name} not found")
            continue

        dst_base = DST / disease / "classification"

        if folder_name == "Oral_Lesions_Malignancy":
            # Goes into Oral_Cancer as a subfolder
            dst_base = DST / disease / "classification_lesions_malignancy"

        n = copy_tree_contents(src_folder, dst_base)
        stats[disease] += n
        log(f"  [OK] {folder_name} -> {disease}/classification : {n} files")


# ═══════════════════════════════════════════════════════════════════════
# PHASE 2: CSV-labeled classification (split by disease label)
# ═══════════════════════════════════════════════════════════════════════
def phase2_csv_classification():
    log("\n" + "=" * 70)
    log("PHASE 2: CSV-labeled classification (split by disease)")
    log("=" * 70)

    csv_base = SRC / "Classification_CSV" / "Mouth_Disease_3Class"

    for split in ["train", "test"]:
        csv_file = csv_base / f"{split}.csv"
        img_dir = csv_base / split

        if not csv_file.exists():
            log(f"  [SKIP] {csv_file} not found")
            continue

        # Read CSV and split images by disease label
        disease_images = defaultdict(list)
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_name = row["ImgName"]
                label = row["Label"]
                disease = CSV_LABEL_MAP.get(label)
                if disease:
                    disease_images[disease].append(img_name)

        for disease, images in disease_images.items():
            dst_dir = ensure_dir(DST / disease / "classification_csv" / split)
            copied = 0
            csv_rows = []
            for img_name in images:
                src_img = img_dir / img_name
                if src_img.exists():
                    shutil.copy2(src_img, dst_dir / img_name)
                    copied += 1
                    csv_rows.append(img_name)

            # Write a disease-specific CSV
            with open(dst_dir.parent / f"{split}_labels.csv", "w", newline="") as cf:
                writer = csv.writer(cf)
                writer.writerow(["ImgName", "Label"])
                for name in csv_rows:
                    writer.writerow([name, disease.split("_", 1)[1]])

            stats[disease] += copied
            log(f"  [OK] CSV {split} -> {disease} : {copied} images")


# ═══════════════════════════════════════════════════════════════════════
# PHASE 3: Single-disease YOLO detection datasets
# ═══════════════════════════════════════════════════════════════════════
def phase3_single_disease_yolo():
    log("\n" + "=" * 70)
    log("PHASE 3: Single-disease YOLO detection datasets")
    log("=" * 70)

    # Datasets that map entirely to one disease
    single_disease = {
        "Dental_Cavity_2Class":      "01_Caries",
        "Gingivitis_Severity_7Class": "02_Gingivitis",
    }

    yolo_base = SRC / "Detection_YOLO"

    for dataset_name, disease in single_disease.items():
        src = yolo_base / dataset_name
        if not src.exists():
            log(f"  [SKIP] {dataset_name} not found")
            continue

        dst = DST / disease / f"detection_yolo_{dataset_name}"
        n = copy_tree_contents(src, dst)
        stats[disease] += n
        log(f"  [OK] {dataset_name} -> {disease} : {n} files")


# ═══════════════════════════════════════════════════════════════════════
# PHASE 4: Multi-disease YOLO detection (split by class)
# ═══════════════════════════════════════════════════════════════════════
def split_yolo_by_disease(dataset_dir, class_map, dataset_name):
    """Parse YOLO labels and split images+labels by disease class."""
    dataset_dir = Path(dataset_dir)
    disease_counts = defaultdict(int)

    # Find all image/label pairs
    for split_dir in dataset_dir.iterdir():
        if not split_dir.is_dir():
            continue

        split_name = split_dir.name  # train, test, val, etc.

        # Detect structure: either {split}/images + {split}/labels
        # or {split} directly containing images and labels
        img_dir = split_dir / "images" if (split_dir / "images").exists() else split_dir
        lbl_dir = split_dir / "labels" if (split_dir / "labels").exists() else split_dir

        if not img_dir.exists():
            continue

        for img_file in img_dir.iterdir():
            if img_file.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue

            # Find corresponding label
            label_file = lbl_dir / (img_file.stem + ".txt")
            if not label_file.exists():
                continue

            # Parse label to find which diseases are present
            diseases_in_image = set()
            lines_by_disease = defaultdict(list)

            with open(label_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    class_id = int(parts[0])
                    disease = class_map.get(class_id)
                    if disease:
                        diseases_in_image.add(disease)
                        # Remap class to 0 for single-disease label
                        new_line = "0 " + " ".join(parts[1:])
                        lines_by_disease[disease].append(new_line)

            # Copy image and filtered labels to each disease folder
            for disease in diseases_in_image:
                dst_img_dir = ensure_dir(
                    DST / disease / f"detection_yolo_{dataset_name}" / split_name / "images"
                )
                dst_lbl_dir = ensure_dir(
                    DST / disease / f"detection_yolo_{dataset_name}" / split_name / "labels"
                )

                shutil.copy2(img_file, dst_img_dir / img_file.name)
                with open(dst_lbl_dir / label_file.name, "w") as f:
                    f.write("\n".join(lines_by_disease[disease]) + "\n")

                disease_counts[disease] += 1

    return disease_counts


def phase4_multi_disease_yolo():
    log("\n" + "=" * 70)
    log("PHASE 4: Multi-disease YOLO detection (split by class)")
    log("=" * 70)

    yolo_base = SRC / "Detection_YOLO"

    # Intraoral 4-Class
    src = yolo_base / "Intraoral_4Class"
    if src.exists():
        counts = split_yolo_by_disease(src, INTRAORAL_CLASS_MAP, "Intraoral")
        for disease, count in counts.items():
            stats[disease] += count * 2  # image + label
            log(f"  [OK] Intraoral_4Class -> {disease} : {count} images")

    # Labelled Ulcer Calculus 2-Class
    src = yolo_base / "Labelled_Ulcer_Calculus_2Class"
    if src.exists():
        counts = split_yolo_by_disease(src, ULCER_CALCULUS_CLASS_MAP, "Labelled")
        for disease, count in counts.items():
            stats[disease] += count * 2
            log(f"  [OK] Labelled_Ulcer_Calculus -> {disease} : {count} images")


# ═══════════════════════════════════════════════════════════════════════
# PHASE 5: VOC detection datasets (all cavity/caries)
# ═══════════════════════════════════════════════════════════════════════
def phase5_voc_detection():
    log("\n" + "=" * 70)
    log("PHASE 5: VOC detection datasets (cavity -> Caries)")
    log("=" * 70)

    voc_base = SRC / "Detection_VOC"
    disease = "01_Caries"

    for dataset_name in ["Dental_Cavity_Colored", "Dental_Cavity_Xray"]:
        src = voc_base / dataset_name
        if not src.exists():
            continue
        dst = DST / disease / f"detection_voc_{dataset_name}"
        n = copy_tree_contents(src, dst)
        stats[disease] += n
        log(f"  [OK] {dataset_name} -> {disease} : {n} files")


# ═══════════════════════════════════════════════════════════════════════
# PHASE 6: Segmentation (split by disease class in masks)
# ═══════════════════════════════════════════════════════════════════════
def phase6_segmentation():
    log("\n" + "=" * 70)
    log("PHASE 6: Segmentation (analyze masks, split by disease)")
    log("=" * 70)

    seg_base = SRC / "Segmentation" / "Dental_Caries_Cavity_Crack_4Class"

    try:
        from PIL import Image
    except ImportError:
        log("  [ERROR] PIL not available, skipping segmentation analysis")
        log("  Copying entire segmentation dataset to Caries (primary disease)")
        dst = DST / "01_Caries" / "segmentation"
        n = copy_tree_contents(seg_base, dst)
        stats["01_Caries"] += n
        return

    for split in ["train", "test"]:
        split_dir = seg_base / split
        if not split_dir.exists():
            continue

        img_dir = split_dir / "Image"
        mask_dir = split_dir / "masks"

        if not img_dir.exists() or not mask_dir.exists():
            continue

        for img_file in img_dir.iterdir():
            if img_file.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue

            # Find corresponding mask
            mask_file = mask_dir / img_file.name
            if not mask_file.exists():
                # Try with .png extension
                mask_file = mask_dir / (img_file.stem + ".png")
            if not mask_file.exists():
                continue

            # Analyze mask to find which disease classes are present
            try:
                mask = np.array(Image.open(mask_file))
                unique_values = set(np.unique(mask))
            except Exception:
                continue

            # Map pixel values to diseases
            diseases_in_image = set()
            for pv in unique_values:
                disease = SEGMENTATION_CLASS_MAP.get(int(pv))
                if disease:
                    diseases_in_image.add(disease)

            # If no disease found (only background/tooth), skip
            if not diseases_in_image:
                continue

            # Copy image and mask to each relevant disease folder
            for disease in diseases_in_image:
                dst_img = ensure_dir(
                    DST / disease / "segmentation" / split / "images"
                )
                dst_mask = ensure_dir(
                    DST / disease / "segmentation" / split / "masks"
                )
                shutil.copy2(img_file, dst_img / img_file.name)
                shutil.copy2(mask_file, dst_mask / mask_file.name)
                stats[disease] += 2

    # Also copy the _classes.csv to each disease that got segmentation data
    classes_csv = seg_base / "test" / "_classes.csv"
    if not classes_csv.exists():
        classes_csv = seg_base / "train" / "_classes.csv"

    for disease in ["01_Caries", "10_Tooth_Crack"]:
        dst_seg = DST / disease / "segmentation"
        if dst_seg.exists() and classes_csv.exists():
            shutil.copy2(classes_csv, dst_seg / "_classes.csv")

    log(f"  [OK] Segmentation masks analyzed and split by disease")


# ═══════════════════════════════════════════════════════════════════════
# PHASE 7: Captioning (all gingivitis)
# ═══════════════════════════════════════════════════════════════════════
def phase7_captioning():
    log("\n" + "=" * 70)
    log("PHASE 7: Captioning dataset (all gingivitis)")
    log("=" * 70)

    src = SRC / "Captioning" / "Gingivitis_Severity"
    if not src.exists():
        log("  [SKIP] Captioning dataset not found")
        return

    disease = "02_Gingivitis"
    dst = DST / disease / "captioning"
    n = copy_tree_contents(src, dst)
    stats[disease] += n
    log(f"  [OK] Gingivitis Captioning -> {disease} : {n} files")


# ═══════════════════════════════════════════════════════════════════════
# PHASE 8: COCO OPG Panoramic (extract disease-specific annotations)
# ═══════════════════════════════════════════════════════════════════════
def phase8_coco_panoramic():
    log("\n" + "=" * 70)
    log("PHASE 8: COCO OPG Panoramic 31-class (extract by disease)")
    log("=" * 70)

    coco_base = SRC / "Detection_COCO" / "OPG_Panoramic_31Class"
    if not coco_base.exists():
        log("  [SKIP] COCO OPG dataset not found")
        return

    # Also copy full dataset to _Panoramic_Xray for reference
    pano_dst = DST / "_Panoramic_Xray" / "coco_31class"
    log("  Copying full OPG COCO dataset to _Panoramic_Xray...")
    n_full = copy_tree_contents(coco_base, pano_dst)
    stats["_Panoramic_Xray"] += n_full
    log(f"  [OK] Full OPG COCO -> _Panoramic_Xray : {n_full} files")

    # Now extract disease-specific subsets
    for split in ["train", "valid", "test"]:
        ann_file = coco_base / "annotations" / f"{split}_coco.json"
        img_dir = coco_base / split

        if not ann_file.exists() or not img_dir.exists():
            continue

        with open(ann_file, "r") as f:
            coco_data = json.load(f)

        # Index annotations by image_id
        img_annotations = defaultdict(list)
        for ann in coco_data.get("annotations", []):
            img_annotations[ann["image_id"]].append(ann)

        # Index images by id
        img_by_id = {img["id"]: img for img in coco_data.get("images", [])}

        # For each disease category, find relevant images
        disease_images = defaultdict(set)  # disease -> set of image_ids
        disease_annotations = defaultdict(list)  # disease -> list of annotations

        for ann in coco_data.get("annotations", []):
            cat_id = ann["category_id"]
            disease = OPG_DISEASE_CATEGORIES.get(cat_id)
            if disease:
                disease_images[disease].add(ann["image_id"])
                disease_annotations[disease].append(ann)

        # Copy images and create filtered COCO JSON per disease
        for disease, image_ids in disease_images.items():
            dst_img_dir = ensure_dir(
                DST / disease / "detection_coco_opg" / split
            )
            dst_ann_dir = ensure_dir(
                DST / disease / "detection_coco_opg" / "annotations"
            )

            # Copy images
            copied = 0
            filtered_images = []
            for img_id in image_ids:
                img_info = img_by_id.get(img_id)
                if not img_info:
                    continue
                src_img = img_dir / img_info["file_name"]
                if src_img.exists():
                    shutil.copy2(src_img, dst_img_dir / img_info["file_name"])
                    filtered_images.append(img_info)
                    copied += 1

            # Create filtered COCO JSON
            # Only include categories relevant to this disease
            relevant_cat_ids = {
                cat_id for cat_id, d in OPG_DISEASE_CATEGORIES.items()
                if d == disease
            }
            filtered_categories = [
                c for c in coco_data.get("categories", [])
                if c["id"] in relevant_cat_ids
            ]
            filtered_anns = [
                a for a in disease_annotations[disease]
                if a["image_id"] in image_ids
            ]

            coco_out = {
                "images": filtered_images,
                "annotations": filtered_anns,
                "categories": filtered_categories,
            }

            with open(dst_ann_dir / f"{split}_coco.json", "w") as f:
                json.dump(coco_out, f)

            stats[disease] += copied
            log(f"  [OK] OPG COCO {split} -> {disease} : {copied} images, {len(filtered_anns)} annotations")


# ═══════════════════════════════════════════════════════════════════════
# PHASE 9: YOLO OPG datasets (panoramic reference)
# ═══════════════════════════════════════════════════════════════════════
def phase9_panoramic_yolo():
    log("\n" + "=" * 70)
    log("PHASE 9: YOLO OPG panoramic datasets -> _Panoramic_Xray")
    log("=" * 70)

    yolo_base = SRC / "Detection_YOLO"

    for dataset_name in ["OPG_Panoramic_31Class_YOLO", "OPG_Kennedy_10Class"]:
        src = yolo_base / dataset_name
        if not src.exists():
            continue
        dst = DST / "_Panoramic_Xray" / dataset_name
        n = copy_tree_contents(src, dst)
        stats["_Panoramic_Xray"] += n
        log(f"  [OK] {dataset_name} -> _Panoramic_Xray : {n} files")


# ═══════════════════════════════════════════════════════════════════════
# PHASE 10: Dental Anatomy dataset
# ═══════════════════════════════════════════════════════════════════════
def phase10_dental_anatomy():
    log("\n" + "=" * 70)
    log("PHASE 10: Dental Anatomy 7-class -> _Dental_Anatomy")
    log("=" * 70)

    src = SRC / "Detection_YOLO" / "Dental_Anatomy_7Class"
    if not src.exists():
        log("  [SKIP] Dental Anatomy not found")
        return

    dst = DST / "_Dental_Anatomy" / "detection_yolo_7class"
    n = copy_tree_contents(src, dst)
    stats["_Dental_Anatomy"] += n
    log(f"  [OK] Dental Anatomy -> _Dental_Anatomy : {n} files")


# ═══════════════════════════════════════════════════════════════════════
# PHASE 11: Raw unlabeled patient data
# ═══════════════════════════════════════════════════════════════════════
def phase11_raw_unlabeled():
    log("\n" + "=" * 70)
    log("PHASE 11: Raw unlabeled patient data")
    log("=" * 70)

    src = SRC / "Raw_Unlabeled"
    if not src.exists():
        log("  [SKIP] Raw_Unlabeled not found")
        return

    dst = DST / "_Raw_Unlabeled"
    n = copy_tree_contents(src, dst)
    stats["_Raw_Unlabeled"] += n
    log(f"  [OK] Raw_Unlabeled -> _Raw_Unlabeled : {n} files")


# ═══════════════════════════════════════════════════════════════════════
# PHASE 12: Create data.yaml files per disease detection folder
# ═══════════════════════════════════════════════════════════════════════
def phase12_create_yaml_configs():
    log("\n" + "=" * 70)
    log("PHASE 12: Generate data.yaml configs per disease")
    log("=" * 70)

    disease_names = {
        "01_Caries": "Caries (Tooth Decay / Dental Cavity)",
        "02_Gingivitis": "Gingivitis (Gum Disease)",
        "03_Mouth_Ulcer": "Mouth Ulcer (Oral Ulcer)",
        "04_Calculus": "Calculus (Tartar)",
        "05_Tooth_Discoloration": "Tooth Discoloration",
        "06_Oral_Cancer": "Oral Cancer / Oral Lesions",
        "07_Hypodontia": "Hypodontia (Missing Teeth)",
        "08_Mucocele": "Mucocele",
        "09_Periodontal_Disease": "Periodontal Disease",
        "10_Tooth_Crack": "Tooth Crack / Fracture",
        "11_Healthy": "Healthy (No Disease)",
    }

    for disease_folder in sorted(DST.iterdir()):
        if not disease_folder.is_dir():
            continue
        if disease_folder.name.startswith("_"):
            continue

        disease = disease_folder.name
        display_name = disease_names.get(disease, disease)

        # Find YOLO detection folders and create data.yaml
        for yolo_dir in disease_folder.glob("detection_yolo_*"):
            if not yolo_dir.is_dir():
                continue

            # Check if train/test/val splits exist
            splits = {}
            for split_name in ["train", "test", "val", "valid"]:
                split_path = yolo_dir / split_name
                if split_path.exists():
                    img_sub = split_path / "images"
                    if img_sub.exists():
                        splits[split_name] = f"./{split_name}/images"
                    else:
                        splits[split_name] = f"./{split_name}"

            if not splits:
                continue

            yaml_content = f"# {display_name} - Detection Dataset\n"
            yaml_content += f"# Source: {yolo_dir.name}\n\n"
            if "train" in splits:
                yaml_content += f"train: {splits['train']}\n"
            if "val" in splits:
                yaml_content += f"val: {splits['val']}\n"
            elif "valid" in splits:
                yaml_content += f"val: {splits['valid']}\n"
            if "test" in splits:
                yaml_content += f"test: {splits['test']}\n"
            yaml_content += f"\nnc: 1\n"
            yaml_content += f"names: ['{display_name}']\n"

            yaml_path = yolo_dir / "data.yaml"
            with open(yaml_path, "w") as f:
                f.write(yaml_content)
            log(f"  [OK] Created {yaml_path.relative_to(DST)}")


# ═══════════════════════════════════════════════════════════════════════
# PHASE 13: Generate comprehensive README
# ═══════════════════════════════════════════════════════════════════════
def phase13_readme():
    log("\n" + "=" * 70)
    log("PHASE 13: Generating README")
    log("=" * 70)

    readme = "# Dental Disease Datasets - Organized by Disease\n\n"
    readme += "All datasets reorganized by disease type for targeted model training.\n\n"
    readme += "## Disease Folders\n\n"

    for disease_dir in sorted(DST.iterdir()):
        if not disease_dir.is_dir():
            continue

        name = disease_dir.name
        # Count files
        file_count = sum(1 for _ in disease_dir.rglob("*") if _.is_file())
        img_count = sum(
            1 for f in disease_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".heic")
        )

        readme += f"### {name}\n"
        readme += f"- **Total files**: {file_count:,}\n"
        readme += f"- **Images**: {img_count:,}\n"

        # List subfolders
        subdirs = [d for d in disease_dir.iterdir() if d.is_dir()]
        if subdirs:
            readme += "- **Data types**:\n"
            for sd in sorted(subdirs):
                sd_files = sum(1 for _ in sd.rglob("*") if _.is_file())
                readme += f"  - `{sd.name}/` ({sd_files:,} files)\n"
        readme += "\n"

    readme += "## Data Type Legend\n\n"
    readme += "| Folder prefix | Description |\n"
    readme += "|---|---|\n"
    readme += "| `classification/` | Folder-based image classification (image -> disease label) |\n"
    readme += "| `classification_csv/` | CSV-labeled classification images |\n"
    readme += "| `classification_lesions_malignancy/` | Benign/malignant lesion classification |\n"
    readme += "| `detection_yolo_*/` | YOLO format bounding box annotations |\n"
    readme += "| `detection_voc_*/` | Pascal VOC XML bounding box annotations |\n"
    readme += "| `detection_coco_opg/` | COCO format annotations from OPG panoramic X-rays |\n"
    readme += "| `segmentation/` | Pixel-level segmentation masks |\n"
    readme += "| `captioning/` | Image captions + YOLO labels |\n\n"

    readme += "## Notes\n\n"
    readme += "- Multi-class detection datasets were split per disease with filtered labels\n"
    readme += "- YOLO labels in split datasets are remapped to class 0 for single-disease training\n"
    readme += "- Segmentation masks were analyzed per-pixel to determine disease presence\n"
    readme += "- OPG COCO annotations were filtered to disease-relevant categories\n"
    readme += "- Augmented images are in separate subfolders to prevent data leakage\n"
    readme += "- `_Panoramic_Xray/` contains full multi-class OPG datasets (not disease-specific)\n"
    readme += "- `_Dental_Anatomy/` contains tooth type identification (not disease-specific)\n"
    readme += "- `_Raw_Unlabeled/` contains patient photos that need annotation\n"

    with open(DST / "README.md", "w") as f:
        f.write(readme)
    log("  [OK] README.md generated")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    global LOG_FILE

    print(f"Source: {SRC}")
    print(f"Destination: {DST}")
    print(f"Starting disease-based reorganization...\n")

    ensure_dir(DST)
    LOG_FILE = open(DST / "_reorganization_log.txt", "w")

    start = time.time()

    phase1_classification()
    phase2_csv_classification()
    phase3_single_disease_yolo()
    phase4_multi_disease_yolo()
    phase5_voc_detection()
    phase6_segmentation()
    phase7_captioning()
    phase8_coco_panoramic()
    phase9_panoramic_yolo()
    phase10_dental_anatomy()
    phase11_raw_unlabeled()
    phase12_create_yaml_configs()
    phase13_readme()

    elapsed = time.time() - start

    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)
    total = 0
    for disease in sorted(stats.keys()):
        log(f"  {disease}: {stats[disease]:,} files")
        total += stats[disease]
    log(f"\n  TOTAL: {total:,} files organized")
    log(f"  Time: {elapsed:.1f}s")

    LOG_FILE.close()
    print(f"\nDone! Results in: {DST}")


if __name__ == "__main__":
    main()
