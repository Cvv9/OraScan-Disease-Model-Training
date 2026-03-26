"""Fix Phase 4 (multi-disease YOLO) and Phase 6 (segmentation)
from the disease reorganization."""

import os
import shutil
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from PIL import Image

DST = Path(r"E:\ECG and Dental Images\Dental data set\By_Disease")
SRC = Path(r"E:\ECG and Dental Images\Dental data set\Organized")

INTRAORAL_CLASS_MAP = {
    0: "01_Caries",
    1: "03_Mouth_Ulcer",
    2: "05_Tooth_Discoloration",
    3: "02_Gingivitis",
}

ULCER_CALCULUS_CLASS_MAP = {
    0: "03_Mouth_Ulcer",
    1: "04_Calculus",
}

SEGMENTATION_CLASS_MAP = {
    1: "01_Caries",       # Caries
    2: "01_Caries",       # Cavity (same disease family)
    3: "10_Tooth_Crack",  # Crack
}


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def split_yolo_alt_structure(dataset_dir, class_map, dataset_name):
    """Handle YOLO datasets with images/{split}/ and labels/{split}/ structure."""
    dataset_dir = Path(dataset_dir)
    img_root = dataset_dir / "images"
    lbl_root = dataset_dir / "labels"

    if not img_root.exists() or not lbl_root.exists():
        print(f"  [SKIP] {dataset_name}: images/ or labels/ not found")
        return

    disease_counts = defaultdict(int)

    for split_dir in img_root.iterdir():
        if not split_dir.is_dir():
            continue
        split_name = split_dir.name
        lbl_split = lbl_root / split_name

        if not lbl_split.exists():
            continue

        for img_file in split_dir.iterdir():
            if img_file.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue

            label_file = lbl_split / (img_file.stem + ".txt")
            if not label_file.exists():
                continue

            # Parse label to find which diseases are present
            lines_by_disease = defaultdict(list)

            with open(label_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    class_id = int(parts[0])
                    disease = class_map.get(class_id)
                    if disease:
                        new_line = "0 " + " ".join(parts[1:])
                        lines_by_disease[disease].append(new_line)

            for disease, lines in lines_by_disease.items():
                dst_img_dir = ensure_dir(
                    DST / disease / f"detection_yolo_{dataset_name}" / split_name / "images"
                )
                dst_lbl_dir = ensure_dir(
                    DST / disease / f"detection_yolo_{dataset_name}" / split_name / "labels"
                )

                shutil.copy2(img_file, dst_img_dir / img_file.name)
                with open(dst_lbl_dir / label_file.name, "w") as f:
                    f.write("\n".join(lines) + "\n")

                disease_counts[disease] += 1

    for disease, count in sorted(disease_counts.items()):
        print(f"  [OK] {dataset_name} -> {disease} : {count} images + labels")


def fix_segmentation():
    """Fix segmentation by handling _mask suffix in mask filenames."""
    print("\n=== Fixing Segmentation (mask naming: {stem}_mask.png) ===")

    seg_base = SRC / "Segmentation" / "Dental_Caries_Cavity_Crack_4Class"
    disease_counts = defaultdict(int)

    for split in ["train", "test"]:
        split_dir = seg_base / split
        if not split_dir.exists():
            continue

        img_dir = split_dir / "Image"
        mask_dir = split_dir / "masks"

        if not img_dir.exists() or not mask_dir.exists():
            print(f"  [SKIP] {split}: Image/ or masks/ not found")
            continue

        for img_file in img_dir.iterdir():
            if img_file.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue

            # Mask naming: {stem}_mask.png
            mask_file = mask_dir / f"{img_file.stem}_mask.png"
            if not mask_file.exists():
                # Try exact name
                mask_file = mask_dir / img_file.name
                if not mask_file.exists():
                    mask_file = mask_dir / f"{img_file.stem}.png"
                    if not mask_file.exists():
                        continue

            try:
                mask = np.array(Image.open(mask_file))
                unique_values = set(np.unique(mask))
            except Exception as e:
                print(f"  [WARN] Failed to read mask {mask_file.name}: {e}")
                continue

            diseases_in_image = set()
            for pv in unique_values:
                disease = SEGMENTATION_CLASS_MAP.get(int(pv))
                if disease:
                    diseases_in_image.add(disease)

            if not diseases_in_image:
                continue

            for disease in diseases_in_image:
                dst_img = ensure_dir(DST / disease / "segmentation" / split / "images")
                dst_mask = ensure_dir(DST / disease / "segmentation" / split / "masks")
                shutil.copy2(img_file, dst_img / img_file.name)
                shutil.copy2(mask_file, dst_mask / mask_file.name)
                disease_counts[disease] += 1

    # Copy classes CSV
    classes_csv = seg_base / "test" / "_classes.csv"
    for disease in disease_counts:
        dst_seg = DST / disease / "segmentation"
        if dst_seg.exists() and classes_csv.exists():
            shutil.copy2(classes_csv, dst_seg / "_classes.csv")

    for disease, count in sorted(disease_counts.items()):
        print(f"  [OK] Segmentation -> {disease} : {count} images + masks")


def create_yaml_for_new_folders():
    """Create data.yaml for the newly created detection_yolo folders."""
    print("\n=== Creating data.yaml configs ===")

    disease_names = {
        "01_Caries": "Caries",
        "02_Gingivitis": "Gingivitis",
        "03_Mouth_Ulcer": "Mouth Ulcer",
        "04_Calculus": "Calculus",
        "05_Tooth_Discoloration": "Tooth Discoloration",
    }

    for disease_dir in sorted(DST.iterdir()):
        if not disease_dir.is_dir() or disease_dir.name.startswith("_"):
            continue

        for yolo_dir in disease_dir.glob("detection_yolo_*"):
            if not yolo_dir.is_dir():
                continue

            # Skip if data.yaml already exists
            yaml_path = yolo_dir / "data.yaml"
            if yaml_path.exists():
                continue

            display = disease_names.get(disease_dir.name, disease_dir.name)
            splits = {}
            for s in ["train", "test", "val", "valid"]:
                sp = yolo_dir / s
                if sp.exists():
                    img_sub = sp / "images"
                    splits[s] = f"./{s}/images" if img_sub.exists() else f"./{s}"

            if not splits:
                continue

            content = f"# {display} Detection Dataset\n# Source: {yolo_dir.name}\n\n"
            if "train" in splits:
                content += f"train: {splits['train']}\n"
            if "val" in splits:
                content += f"val: {splits['val']}\n"
            elif "valid" in splits:
                content += f"val: {splits['valid']}\n"
            if "test" in splits:
                content += f"test: {splits['test']}\n"
            content += f"\nnc: 1\nnames: ['{display}']\n"

            with open(yaml_path, "w") as f:
                f.write(content)
            print(f"  [OK] Created {yaml_path.relative_to(DST)}")


if __name__ == "__main__":
    print("=== Fixing Phase 4: Multi-disease YOLO split ===")

    # Intraoral 4-Class (images/{split}/ structure)
    src = SRC / "Detection_YOLO" / "Intraoral_4Class"
    if src.exists():
        split_yolo_alt_structure(src, INTRAORAL_CLASS_MAP, "Intraoral")

    # Labelled Ulcer Calculus 2-Class
    src = SRC / "Detection_YOLO" / "Labelled_Ulcer_Calculus_2Class"
    if src.exists():
        split_yolo_alt_structure(src, ULCER_CALCULUS_CLASS_MAP, "Labelled")

    fix_segmentation()
    create_yaml_for_new_folders()

    print("\nDone!")
