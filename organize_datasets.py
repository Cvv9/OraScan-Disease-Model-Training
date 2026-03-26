"""
Dental Dataset Organizer
========================
Reorganizes the dental dataset collection into a clean, disease-based structure.

Structure:
  Organized/
  ├── Classification/           (folder-labeled images by disease)
  ├── Classification_CSV/       (CSV-labeled datasets)
  ├── Detection_COCO/           (COCO-format detection datasets)
  ├── Detection_YOLO/           (YOLO-format detection datasets)
  ├── Detection_VOC/            (Pascal VOC detection datasets)
  ├── Segmentation/             (Pixel-mask segmentation datasets)
  ├── Captioning/               (Image captioning datasets)
  ├── Raw_Unlabeled/            (Unlabeled patient data)
  └── _Duplicates/              (Identified duplicate datasets)
"""

import os
import shutil
import hashlib
import csv
import json
from pathlib import Path
from collections import defaultdict
import time

BASE = Path(r"E:\ECG and Dental Images\Dental data set")
ORG = BASE / "Organized"
LOG_FILE = ORG / "_reorganization_log.txt"

log_lines = []

def log(msg):
    print(msg)
    log_lines.append(msg)

def safe_move(src, dst):
    """Move a directory/file, creating parent dirs as needed."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.move(str(src), str(dst))
        log(f"  MOVED: {src.name} -> {dst.relative_to(ORG)}")
        return True
    else:
        log(f"  SKIP (not found): {src}")
        return False

def safe_copy(src, dst):
    """Copy a file, creating parent dirs as needed."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(str(src), str(dst))
        return True
    return False

def copy_tree(src, dst):
    """Copy entire directory tree."""
    if src.exists():
        shutil.copytree(str(src), str(dst), dirs_exist_ok=True)
        log(f"  COPIED TREE: {src.name} -> {dst.relative_to(ORG)}")
        return True
    log(f"  SKIP (not found): {src}")
    return False

def count_files(path, extensions=None):
    """Count files in a directory recursively."""
    if not path.exists():
        return 0
    count = 0
    for f in path.rglob("*"):
        if f.is_file():
            if extensions is None or f.suffix.lower() in extensions:
                count += 1
    return count

def copy_images_from_folder(src_folder, dst_folder, prefix=""):
    """Copy all image files from src to dst, optionally prefixing names."""
    if not src_folder.exists():
        log(f"  SKIP (not found): {src_folder}")
        return 0
    dst_folder.mkdir(parents=True, exist_ok=True)
    count = 0
    for f in src_folder.iterdir():
        if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp'):
            dst_name = f"{prefix}{f.name}" if prefix else f.name
            dst_path = dst_folder / dst_name
            if not dst_path.exists():
                shutil.copy2(str(f), str(dst_path))
                count += 1
    return count


def phase1_create_structure():
    """Create the organized directory structure."""
    log("\n" + "="*70)
    log("PHASE 1: Creating organized directory structure")
    log("="*70)

    dirs = [
        "Classification/Calculus",
        "Classification/Caries/original",
        "Classification/Caries/augmented",
        "Classification/Gingivitis",
        "Classification/Healthy",
        "Classification/Hypodontia",
        "Classification/Mouth_Ulcer/original",
        "Classification/Mouth_Ulcer/augmented",
        "Classification/Mucocele",
        "Classification/Tooth_Discoloration/original",
        "Classification/Tooth_Discoloration/augmented",
        "Classification/Oral_Cancer/cancer",
        "Classification/Oral_Cancer/non_cancer",
        "Classification/Oral_Lesions_Malignancy/benign/original",
        "Classification/Oral_Lesions_Malignancy/benign/augmented",
        "Classification/Oral_Lesions_Malignancy/malignant/original",
        "Classification/Oral_Lesions_Malignancy/malignant/augmented",
        "Classification/Periodontal/train/inflammation",
        "Classification/Periodontal/train/normal",
        "Classification/Periodontal/val/inflammation",
        "Classification/Periodontal/val/normal",
        "Classification/Periodontal/test/inflammation",
        "Classification/Periodontal/test/normal",
        "Classification_CSV/Mouth_Disease_3Class",
        "Detection_COCO/OPG_Panoramic_31Class",
        "Detection_YOLO/Dental_Anatomy_7Class",
        "Detection_YOLO/Dental_Cavity_2Class",
        "Detection_YOLO/Gingivitis_Severity_7Class",
        "Detection_YOLO/Intraoral_4Class",
        "Detection_YOLO/OPG_Kennedy_10Class",
        "Detection_YOLO/Labelled_Ulcer_Calculus_2Class",
        "Detection_VOC/Dental_Cavity_Colored",
        "Detection_VOC/Dental_Cavity_Xray",
        "Segmentation/Dental_Caries_Cavity_Crack_4Class",
        "Captioning/Gingivitis_Severity",
        "Raw_Unlabeled/Patient_Data",
        "_Duplicates",
    ]

    for d in dirs:
        (ORG / d).mkdir(parents=True, exist_ok=True)

    log(f"  Created {len(dirs)} directories under {ORG}")


def phase2_detection_datasets():
    """Move well-structured detection datasets with clean names."""
    log("\n" + "="*70)
    log("PHASE 2: Organizing detection datasets")
    log("="*70)

    # 2A: archive(1) -> Detection_COCO/OPG_Panoramic_31Class
    src = BASE / "archive (1)" / "COCO" / "COCO"
    dst = ORG / "Detection_COCO" / "OPG_Panoramic_31Class"
    if src.exists():
        # Move annotations and image folders
        for item in src.iterdir():
            safe_move(item, dst / item.name)
        log(f"  OPG Panoramic 31-Class COCO dataset organized")

    # 2B: Dental Anatomy -> Detection_YOLO/Dental_Anatomy_7Class
    src = BASE / "Dental Anatomy Dataset - Yolov8" / "Dental Dataset"
    dst = ORG / "Detection_YOLO" / "Dental_Anatomy_7Class"
    if src.exists():
        for item in src.iterdir():
            safe_move(item, dst / item.name)
        log(f"  Dental Anatomy 7-Class YOLO dataset organized")

    # 2C: Dental Cavity Detection -> Detection_YOLO/Dental_Cavity_2Class
    src = BASE / "Dental Cavity Detection Dataset" / "Cavity Dataset"
    dst = ORG / "Detection_YOLO" / "Dental_Cavity_2Class"
    if src.exists():
        for item in src.iterdir():
            safe_move(item, dst / item.name)
        log(f"  Dental Cavity Detection 2-Class YOLO dataset organized")

    # 2D: OPG Kennedy -> Detection_YOLO/OPG_Kennedy_10Class
    src = BASE / "OPG Dataset for Kennedy Classification" / "OPG Dataset for Kennedy Classification of Partially Edentulous Arches" / "Dataset" / "Dataset"
    dst = ORG / "Detection_YOLO" / "OPG_Kennedy_10Class"
    if src.exists():
        for item in src.iterdir():
            safe_move(item, dst / item.name)
        # Also move the README, notebook, results from parent
        parent = BASE / "OPG Dataset for Kennedy Classification" / "OPG Dataset for Kennedy Classification of Partially Edentulous Arches"
        for extra in ["READ ME.txt", "Code.ipynb", "Results.pdf"]:
            extra_path = parent / extra
            if extra_path.exists():
                safe_move(extra_path, dst / extra)
        log(f"  OPG Kennedy 10-Class YOLO dataset organized")

    # 2E: Intraoral 4-Class YOLO (from Dental Images)
    src = BASE / "Dental images" / "Caries_Gingivitus_ToothDiscoloration_Ulcer-yolo_annotated-Dataset" / "Caries_Gingivitus_ToothDiscoloration_Ulcer-yolo_annotated-Dataset" / "Data"
    if not src.exists():
        src = BASE / "Dental images" / "Caries_Gingivitus_ToothDiscoloration_Ulcer-yolo_annotated-Dataset" / "Data"
    dst = ORG / "Detection_YOLO" / "Intraoral_4Class"
    if src.exists():
        for item in src.iterdir():
            safe_move(item, dst / item.name)
        # Fix the data.yaml nc value
        yaml_path = dst / "data.yaml"
        if yaml_path.exists():
            content = yaml_path.read_text()
            content = content.replace("nc: 1", "nc: 4")
            yaml_path.write_text(content)
            log(f"  FIXED: data.yaml nc:1 -> nc:4 in Intraoral_4Class")
        log(f"  Intraoral 4-Class YOLO dataset organized")

    # 2F: Labelled Images YOLO subset -> Detection_YOLO/Labelled_Ulcer_Calculus_2Class
    src = BASE / "labelled images" / "labelled images" / "dataset"
    dst = ORG / "Detection_YOLO" / "Labelled_Ulcer_Calculus_2Class"
    if src.exists():
        for item in src.iterdir():
            safe_move(item, dst / item.name)
        # Rename data.yaml.txt to data.yaml if needed
        yaml_txt = dst / "data.yaml.txt"
        yaml_dst = dst / "data.yaml"
        if yaml_txt.exists() and not yaml_dst.exists():
            yaml_txt.rename(yaml_dst)
            log(f"  RENAMED: data.yaml.txt -> data.yaml")
        log(f"  Labelled Ulcer/Calculus 2-Class YOLO dataset organized")

    # 2G: Dental Cavity VOC (from extracted Dental Cavity Dataset)
    src_colored = BASE / "Dental Cavity Dataset" / "Dataset" / "colored"
    src_xray = BASE / "Dental Cavity Dataset" / "Dataset" / "x-ray"
    if src_colored.exists():
        for item in src_colored.iterdir():
            safe_move(item, ORG / "Detection_VOC" / "Dental_Cavity_Colored" / item.name)
        log(f"  Dental Cavity VOC (colored) organized")
    if src_xray.exists():
        for item in src_xray.iterdir():
            safe_move(item, ORG / "Detection_VOC" / "Dental_Cavity_Xray" / item.name)
        log(f"  Dental Cavity VOC (x-ray) organized")


def phase3_classification_datasets():
    """Consolidate classification images by disease from canonical sources."""
    log("\n" + "="*70)
    log("PHASE 3: Consolidating classification images by disease")
    log("="*70)

    # Use "Dental images" as the canonical source for the 6 main diseases
    dental_img = BASE / "Dental images"

    # 3A: Calculus
    src = dental_img / "Calculus" / "Calculus"
    n = copy_images_from_folder(src, ORG / "Classification" / "Calculus")
    log(f"  Calculus: {n} images copied from Dental Images")

    # 3B: Caries (original + augmented)
    src_orig = dental_img / "Caries" / "caries orignal data set"
    src_aug = dental_img / "Caries" / "caries augmented data set"
    if not src_aug.exists():
        src_aug = dental_img / "Caries" / "caries augmented data set" / "preview"
    n1 = copy_images_from_folder(src_orig, ORG / "Classification" / "Caries" / "original")
    n2 = copy_images_from_folder(src_aug, ORG / "Classification" / "Caries" / "augmented")
    log(f"  Caries: {n1} original + {n2} augmented images copied")

    # 3C: Gingivitis
    src = dental_img / "Gingivitis"
    n = copy_images_from_folder(src, ORG / "Classification" / "Gingivitis")
    log(f"  Gingivitis: {n} images copied from Dental Images")

    # 3D: Hypodontia
    src = dental_img / "hypodontia"
    n = copy_images_from_folder(src, ORG / "Classification" / "Hypodontia")
    log(f"  Hypodontia: {n} images copied from Dental Images")

    # 3E: Mouth Ulcer (original + augmented)
    src_orig = dental_img / "Mouth Ulcer" / "ulcer original dataset"
    src_aug = dental_img / "Mouth Ulcer" / "Mouth_Ulcer_augmented_DataSet"
    if not src_aug.exists():
        src_aug = dental_img / "Mouth Ulcer" / "Mouth_Ulcer_augmented_DataSet" / "preview"
    n1 = copy_images_from_folder(src_orig, ORG / "Classification" / "Mouth_Ulcer" / "original")
    n2 = copy_images_from_folder(src_aug, ORG / "Classification" / "Mouth_Ulcer" / "augmented")
    log(f"  Mouth Ulcer: {n1} original + {n2} augmented images copied")

    # 3F: Tooth Discoloration (original + augmented)
    src_orig = dental_img / "Tooth Discoloration" / "tooth discoloration original dataset"
    src_aug = dental_img / "Tooth Discoloration" / "Tooth_discoloration_augmented_dataser"
    if not src_aug.exists():
        src_aug = dental_img / "Tooth Discoloration" / "Tooth_discoloration_augmented_dataser" / "preview"
    n1 = copy_images_from_folder(src_orig, ORG / "Classification" / "Tooth_Discoloration" / "original")
    n2 = copy_images_from_folder(src_aug, ORG / "Classification" / "Tooth_Discoloration" / "augmented")
    log(f"  Tooth Discoloration: {n1} original + {n2} augmented images copied")

    # 3G: Mucocele (from Labelled Images only)
    src = BASE / "labelled images" / "labelled images" / "Mucocele"
    n = copy_images_from_folder(src, ORG / "Classification" / "Mucocele")
    log(f"  Mucocele: {n} images copied from Labelled Images")

    # 3H: Healthy (from MouthDatasetFinal - extract healthy images)
    log(f"  Healthy: extracting from MouthDatasetFinal CSV...")
    healthy_count = 0
    mouth_base = BASE / "MouthDatasetFinal" / "MouthDatasetFinal"
    for split in ["train", "test"]:
        csv_path = mouth_base / f"{split}.csv"
        if csv_path.exists():
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("Label", "").strip() == "h":
                        img_name = row.get("ImgName", "").strip()
                        src_img = mouth_base / split / img_name
                        if src_img.exists():
                            dst_img = ORG / "Classification" / "Healthy" / img_name
                            if safe_copy(src_img, dst_img):
                                healthy_count += 1
    log(f"  Healthy: {healthy_count} images extracted from MouthDatasetFinal")


def phase4_cancer_datasets():
    """Merge all cancer datasets into unified folders."""
    log("\n" + "="*70)
    log("PHASE 4: Merging cancer datasets")
    log("="*70)

    cancer_dst = ORG / "Classification" / "Oral_Cancer" / "cancer"
    non_cancer_dst = ORG / "Classification" / "Oral_Cancer" / "non_cancer"

    # 4A: Oral Cancer (Lips and Tongue) - small dataset
    src_cancer = BASE / "Oral Cancer (Lips and Tongue) images" / "OralCancer" / "cancer"
    if not src_cancer.exists():
        src_cancer = BASE / "Oral Cancer (Lips and Tongue) images" / "cancer"
    src_non = BASE / "Oral Cancer (Lips and Tongue) images" / "OralCancer" / "non-cancer"
    if not src_non.exists():
        src_non = BASE / "Oral Cancer (Lips and Tongue) images" / "non-cancer"

    n1 = copy_images_from_folder(src_cancer, cancer_dst, prefix="lips_tongue_")
    n2 = copy_images_from_folder(src_non, non_cancer_dst, prefix="lips_tongue_")
    log(f"  Oral Cancer (Lips/Tongue): {n1} cancer + {n2} non-cancer")

    # 4B: ORAL CANCER DATASET v1
    src_cancer = BASE / "ORAL CANCER DATASET" / "Oral Cancer" / "Oral Cancer Dataset" / "CANCER"
    src_non = BASE / "ORAL CANCER DATASET" / "Oral Cancer" / "Oral Cancer Dataset" / "NON CANCER"
    n1 = copy_images_from_folder(src_cancer, cancer_dst, prefix="v1_")
    n2 = copy_images_from_folder(src_non, non_cancer_dst, prefix="v1_")
    log(f"  ORAL CANCER DATASET v1: {n1} cancer + {n2} non-cancer")

    # 4C: ORAL CANCER DATASET v2
    src_cancer = BASE / "ORAL CANCER DATASET" / "Oral cancer Dataset 2.0" / "OC Dataset kaggle new" / "CANCER"
    src_non = BASE / "ORAL CANCER DATASET" / "Oral cancer Dataset 2.0" / "OC Dataset kaggle new" / "NON CANCER"
    n1 = copy_images_from_folder(src_cancer, cancer_dst, prefix="v2_")
    n2 = copy_images_from_folder(src_non, non_cancer_dst, prefix="v2_")
    log(f"  ORAL CANCER DATASET v2: {n1} cancer + {n2} non-cancer")

    # Total
    total_cancer = count_files(cancer_dst, {'.jpg', '.jpeg', '.png'})
    total_non = count_files(non_cancer_dst, {'.jpg', '.jpeg', '.png'})
    log(f"  TOTAL Oral Cancer: {total_cancer} cancer + {total_non} non-cancer = {total_cancer + total_non}")


def phase5_lesions_periodontal():
    """Organize oral lesions malignancy and periodontal datasets."""
    log("\n" + "="*70)
    log("PHASE 5: Organizing Oral Lesions Malignancy + Periodontal")
    log("="*70)

    # 5A: Oral Lesions Malignancy
    lesions_base = BASE / "Oral Lesions Malignancy Detection Dataset" / "Oral Images Dataset"

    src = lesions_base / "original_data" / "benign_lesions"
    n = copy_images_from_folder(src, ORG / "Classification" / "Oral_Lesions_Malignancy" / "benign" / "original")
    log(f"  Oral Lesions Benign (original): {n}")

    src = lesions_base / "augmented_data" / "augmented_benign"
    n = copy_images_from_folder(src, ORG / "Classification" / "Oral_Lesions_Malignancy" / "benign" / "augmented")
    log(f"  Oral Lesions Benign (augmented): {n}")

    src = lesions_base / "original_data" / "malignant_lesions"
    n = copy_images_from_folder(src, ORG / "Classification" / "Oral_Lesions_Malignancy" / "malignant" / "original")
    log(f"  Oral Lesions Malignant (original): {n}")

    src = lesions_base / "augmented_data" / "augmented_malignant"
    n = copy_images_from_folder(src, ORG / "Classification" / "Oral_Lesions_Malignancy" / "malignant" / "augmented")
    log(f"  Oral Lesions Malignant (augmented): {n}")

    # 5B: Periodontal Diseases (move entire split structure)
    perio_src = BASE / "Periodonatal diseases" / "periodontal_disease"
    perio_dst = ORG / "Classification" / "Periodontal"
    if perio_src.exists():
        for split in ["train", "val", "test"]:
            for cls in ["inflammation", "normal"]:
                src_dir = perio_src / split / cls
                dst_dir = perio_dst / split / cls
                n = copy_images_from_folder(src_dir, dst_dir)
                log(f"  Periodontal {split}/{cls}: {n}")


def phase6_segmentation_captioning():
    """Organize segmentation and captioning datasets."""
    log("\n" + "="*70)
    log("PHASE 6: Organizing Segmentation + Captioning datasets")
    log("="*70)

    # 6A: Semantic Segmentation
    src = BASE / "Semantic_segmentation_" / "Dental"
    dst = ORG / "Segmentation" / "Dental_Caries_Cavity_Crack_4Class"
    if src.exists():
        for item in src.iterdir():
            safe_move(item, dst / item.name)
        log(f"  Semantic Segmentation (4-class) organized")

    # 6B: Gingivitis Captioning
    src = BASE / "A DENTAL INTRAORAL IMAGE DATASET OF GINGIVITIS FOR IMAGE CAPTIONING" / "Dataset" / "Dataset"
    if not src.exists():
        src = BASE / "A DENTAL INTRAORAL IMAGE DATASET OF GINGIVITIS FOR IMAGE CAPTIONING" / "Dataset"
    dst = ORG / "Captioning" / "Gingivitis_Severity"
    if src.exists():
        for item in src.iterdir():
            if item.is_dir() or item.suffix.lower() in ('.csv', '.txt', '.json'):
                safe_move(item, dst / item.name)
        log(f"  Gingivitis Captioning dataset organized")

    # Also move the YOLO detection part of Gingivitis into Detection_YOLO
    dst_yolo = ORG / "Detection_YOLO" / "Gingivitis_Severity_7Class"
    cap_dst = ORG / "Captioning" / "Gingivitis_Severity"
    if cap_dst.exists():
        # The YOLO labels are already inside Training/Labels, Validation/Labels, Test/Labels
        # We'll create symlinks or just note this is shared
        log(f"  Note: Gingivitis YOLO labels are within Captioning/Gingivitis_Severity/*/Labels/")
        # Copy the structure for YOLO use
        for split_name, split_dir in [("train", "Training"), ("val", "Validation"), ("test", "Test")]:
            img_src = cap_dst / split_dir / "Images"
            lbl_src = cap_dst / split_dir / "Labels"
            if img_src.exists() and lbl_src.exists():
                copy_tree(img_src, dst_yolo / split_name / "images")
                copy_tree(lbl_src, dst_yolo / split_name / "labels")
        log(f"  Gingivitis YOLO 7-Class detection dataset created from captioning data")


def phase7_csv_dataset():
    """Organize CSV-labeled classification dataset."""
    log("\n" + "="*70)
    log("PHASE 7: Organizing CSV-labeled dataset (MouthDatasetFinal)")
    log("="*70)

    src = BASE / "MouthDatasetFinal" / "MouthDatasetFinal"
    dst = ORG / "Classification_CSV" / "Mouth_Disease_3Class"
    if src.exists():
        for item in src.iterdir():
            safe_move(item, dst / item.name)
        log(f"  MouthDatasetFinal -> Mouth_Disease_3Class (caries/gingivitis/healthy)")


def phase8_raw_patient_data():
    """Move raw unlabeled patient data."""
    log("\n" + "="*70)
    log("PHASE 8: Organizing raw unlabeled patient data")
    log("="*70)

    src = BASE / "Patient data"
    dst = ORG / "Raw_Unlabeled" / "Patient_Data"
    if src.exists():
        for item in src.iterdir():
            safe_move(item, dst / item.name)
        log(f"  Patient Data moved to Raw_Unlabeled/")


def phase9_mark_duplicates():
    """Move identified duplicate datasets."""
    log("\n" + "="*70)
    log("PHASE 9: Moving duplicate/redundant datasets")
    log("="*70)

    dup_dst = ORG / "_Duplicates"

    # Oral Infection is a duplicate of Dental Images classification
    src = BASE / "Oral Infection"
    if src.exists():
        safe_move(src, dup_dst / "Oral_Infection_DUPLICATE_OF_Dental_Images")

    # Tooth Decay dataset4 is a duplicate of Dental Images
    src = BASE / "Tooth Decay"
    if src.exists():
        safe_move(src, dup_dst / "Tooth_Decay_PARTIAL_DUPLICATES")

    # Labelled Images Kaggle Data is a partial duplicate
    src = BASE / "labelled images"
    if src.exists():
        safe_move(src, dup_dst / "Labelled_Images_PARTIAL_DUPLICATES")

    # Move remaining empty/cleaned folders
    src = BASE / "Dental images"
    if src.exists():
        safe_move(src, dup_dst / "Dental_Images_ORIGINALS_MOVED")

    log(f"  Duplicate datasets moved to _Duplicates/")


def phase10_verify():
    """Verify the final structure and generate counts."""
    log("\n" + "="*70)
    log("PHASE 10: Verification - Final structure and counts")
    log("="*70)

    img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.heic'}

    for category in sorted(ORG.iterdir()):
        if category.is_dir() and not category.name.startswith('_'):
            total = count_files(category, img_exts)
            log(f"\n  {category.name}/  ({total} images)")
            for sub in sorted(category.iterdir()):
                if sub.is_dir():
                    sub_count = count_files(sub, img_exts)
                    log(f"    {sub.name}/  ({sub_count} images)")
                    for subsub in sorted(sub.iterdir()):
                        if subsub.is_dir():
                            ss_count = count_files(subsub, img_exts)
                            if ss_count > 0:
                                log(f"      {subsub.name}/  ({ss_count})")


def generate_readme():
    """Generate a README for the organized dataset."""
    log("\n" + "="*70)
    log("Generating README.md")
    log("="*70)

    readme = """# Organized Dental Dataset

## Structure

### Classification/ (Folder-based labels)
Images organized by disease class. For training image classification models.

| Disease | Original | Augmented | Total |
|---------|----------|-----------|-------|
| Calculus | 1,296 | - | 1,296 |
| Caries | 219 | 2,382 | 2,601 |
| Gingivitis | 2,349 | - | 2,349 |
| Healthy | ~2,940 | - | ~2,940 |
| Hypodontia | 1,251 | - | 1,251 |
| Mouth Ulcer | 265 | 2,541 | 2,806 |
| Mucocele | 48 | - | 48 |
| Tooth Discoloration | 183 | 1,834 | 2,017 |
| Oral Cancer | ~1,087 cancer + ~744 non-cancer | - | ~1,831 |
| Oral Lesions | 323 | 2,270 | 2,593 |
| Periodontal | 220 (pre-split) | - | 220 |

### Classification_CSV/ (CSV-labeled)
- **Mouth_Disease_3Class/**: 7,671 images (caries/gingivitis/healthy) with CSV labels

### Detection_COCO/
- **OPG_Panoramic_31Class/**: 13,399 panoramic X-rays, 94,794 COCO annotations, 31 dental categories

### Detection_YOLO/
- **Dental_Anatomy_7Class/**: 724 images, 7 tooth types (from Roboflow)
- **Dental_Cavity_2Class/**: 418 images, cavity/normal (oriented bounding boxes)
- **Gingivitis_Severity_7Class/**: 1,096 images, 7 classes (jaw regions + severity levels)
- **Intraoral_4Class/**: 1,542 images, caries/ulcer/tooth_discoloration/gingivitis
- **OPG_Kennedy_10Class/**: 622 OPG images, 10 Kennedy classification categories
- **Labelled_Ulcer_Calculus_2Class/**: 196 images, mouth_ulcer/calculus

### Detection_VOC/
- **Dental_Cavity_Colored/**: 89 colored images + XML annotations
- **Dental_Cavity_Xray/**: 926 x-ray images + XML annotations

### Segmentation/
- **Dental_Caries_Cavity_Crack_4Class/**: 2,495 images + pixel masks (background/caries/cavity/crack)

### Captioning/
- **Gingivitis_Severity/**: 1,096 high-res images + 3 captions each + YOLO labels

### Raw_Unlabeled/
- **Patient_Data/**: 1,745 raw patient intraoral photos (needs labeling)

### _Duplicates/
Datasets identified as duplicates of canonical sources. Kept for reference.

## Notes
- Augmented images are separated from originals to prevent data leakage during training
- data.yaml `nc:1` bug fixed to `nc:4` in Intraoral_4Class
- data.yaml.txt renamed to data.yaml in Labelled_Ulcer_Calculus_2Class
- Cancer datasets merged from 3 separate sources with prefixed filenames to avoid collisions
"""

    readme_path = ORG / "README.md"
    readme_path.write_text(readme, encoding="utf-8")
    log(f"  README.md generated at {readme_path}")


def main():
    start = time.time()
    log(f"Dental Dataset Reorganization")
    log(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Source: {BASE}")
    log(f"Destination: {ORG}")

    ORG.mkdir(parents=True, exist_ok=True)

    phase1_create_structure()
    phase2_detection_datasets()
    phase3_classification_datasets()
    phase4_cancer_datasets()
    phase5_lesions_periodontal()
    phase6_segmentation_captioning()
    phase7_csv_dataset()
    phase8_raw_patient_data()
    phase9_mark_duplicates()
    phase10_verify()
    generate_readme()

    elapsed = time.time() - start
    log(f"\nCompleted in {elapsed:.1f} seconds")

    # Write log
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    print(f"\nLog saved to: {LOG_FILE}")


if __name__ == "__main__":
    main()
