"""
Integrate newly downloaded datasets (DENTEX, SMART-OM) into disease-based folders.

DENTEX: Panoramic X-rays with disease annotations (Caries, Deep Caries, Periapical Lesion, Impacted)
SMART-OM: Smartphone oral mucosa images (Normal, Variation from Normal, OPMD, Oral Cancer)
"""

import json
import os
import shutil
from pathlib import Path
from collections import defaultdict

DST = Path(r"E:\ECG and Dental Images\Dental data set\By_Disease")
DENTEX_BASE = Path(r"E:\ECG and Dental Images\Dental data set\Downloads\DENTEX\extracted")
DENTEX_ANN = Path(r"E:\ECG and Dental Images\Dental data set\Downloads\DENTEX\DENTEX")
SMART_OM_BASE = Path(r"E:\ECG and Dental Images\Dental data set\Downloads\SMART_OM\extracted\SMART-OM")


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


# ═══════════════════════════════════════════════════════════════════════
# DENTEX Integration
# ═══════════════════════════════════════════════════════════════════════
def integrate_dentex():
    """Integrate DENTEX panoramic X-rays split by disease."""
    print("=" * 70)
    print("Integrating DENTEX dataset")
    print("=" * 70)

    # Disease mapping from category_id_3
    disease_map = {
        0: "07_Hypodontia",         # Impacted teeth -> dental structural issue
        1: "01_Caries",             # Caries
        2: "01_Caries",             # Periapical Lesion (caries-related)
        3: "01_Caries",             # Deep Caries
    }

    disease_names_3 = {0: "Impacted", 1: "Caries", 2: "Periapical_Lesion", 3: "Deep_Caries"}

    # Process the fully labeled dataset (quadrant-enumeration-disease)
    qed_dir = DENTEX_BASE / "training_data" / "quadrant-enumeration-disease"
    qed_ann = None
    for f in qed_dir.glob("*.json"):
        qed_ann = f
        break

    if qed_ann and qed_ann.exists():
        data = json.load(open(qed_ann))
        img_dir = qed_dir / "xrays"

        # Group annotations by image_id
        img_by_id = {img["id"]: img for img in data["images"]}
        img_diseases = defaultdict(set)
        img_annotations = defaultdict(list)

        for ann in data["annotations"]:
            cat3 = ann.get("category_id_3")
            if cat3 is not None and cat3 in disease_map:
                disease = disease_map[cat3]
                img_diseases[ann["image_id"]].add(disease)
                img_annotations[(ann["image_id"], disease)].append(ann)

        # Copy images to disease folders
        disease_counts = defaultdict(int)
        for img_id, diseases in img_diseases.items():
            img_info = img_by_id.get(img_id)
            if not img_info:
                continue

            src_img = img_dir / img_info["file_name"]
            if not src_img.exists():
                continue

            for disease in diseases:
                dst_img_dir = ensure_dir(DST / disease / "dentex_panoramic" / "train" / "images")
                dst_ann_dir = ensure_dir(DST / disease / "dentex_panoramic" / "train" / "annotations")

                shutil.copy2(src_img, dst_img_dir / src_img.name)

                # Save filtered annotations as COCO JSON per image
                filtered_anns = img_annotations[(img_id, disease)]
                ann_data = {
                    "image": img_info,
                    "annotations": filtered_anns,
                    "disease_categories": [
                        {"id": k, "name": v}
                        for k, v in disease_names_3.items()
                        if disease_map.get(k) == disease
                    ]
                }
                with open(dst_ann_dir / f"{src_img.stem}.json", "w") as f:
                    json.dump(ann_data, f)

                disease_counts[disease] += 1

        for disease, count in sorted(disease_counts.items()):
            print(f"  [OK] DENTEX train -> {disease} : {count} images")

    # Process test/disease split
    disease_dir = DENTEX_BASE / "disease"
    if disease_dir.exists():
        input_dir = disease_dir / "input"
        label_dir = disease_dir / "label"

        disease_counts = defaultdict(int)

        if input_dir.exists() and label_dir.exists():
            for img_file in input_dir.iterdir():
                if img_file.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue

                # Find label
                label_file = label_dir / (img_file.stem + ".json")
                if not label_file.exists():
                    continue

                label_data = json.load(open(label_file))
                diseases_found = set()

                for shape in label_data.get("shapes", []):
                    label = shape.get("label", "")
                    # Parse: first number is disease type
                    parts = label.split("-")
                    if len(parts) >= 2:
                        try:
                            dtype = int(parts[0])
                            # Map Turkish disease codes
                            if dtype == 1:  # curuk = caries
                                diseases_found.add("01_Caries")
                            elif dtype == 2:  # kuretaj = curettage/periodontal
                                diseases_found.add("09_Periodontal_Disease")
                            elif dtype == 3:  # kanal = root canal (implies caries history)
                                diseases_found.add("01_Caries")
                            elif dtype == 6:  # gomulu = impacted
                                diseases_found.add("07_Hypodontia")
                            elif dtype == 7:  # lezyon = periapical lesion
                                diseases_found.add("01_Caries")
                        except ValueError:
                            pass

                for disease in diseases_found:
                    dst_img_dir = ensure_dir(DST / disease / "dentex_panoramic" / "test" / "images")
                    dst_lbl_dir = ensure_dir(DST / disease / "dentex_panoramic" / "test" / "labels")

                    shutil.copy2(img_file, dst_img_dir / img_file.name)
                    shutil.copy2(label_file, dst_lbl_dir / label_file.name)
                    disease_counts[disease] += 1

        for disease, count in sorted(disease_counts.items()):
            print(f"  [OK] DENTEX test -> {disease} : {count} images")

    # Process validation data
    val_ann = DENTEX_ANN / "validation_triple.json"
    val_img_dir = DENTEX_BASE / "validation_data" / "quadrant_enumeration_disease" / "xrays"

    if val_ann.exists() and val_img_dir.exists():
        data = json.load(open(val_ann))
        img_by_id = {img["id"]: img for img in data["images"]}
        img_diseases = defaultdict(set)

        for ann in data["annotations"]:
            cat3 = ann.get("category_id_3")
            if cat3 is not None and cat3 in disease_map:
                img_diseases[ann["image_id"]].add(disease_map[cat3])

        disease_counts = defaultdict(int)
        for img_id, diseases in img_diseases.items():
            img_info = img_by_id.get(img_id)
            if not img_info:
                continue

            src_img = val_img_dir / img_info["file_name"]
            if not src_img.exists():
                continue

            for disease in diseases:
                dst_img_dir = ensure_dir(DST / disease / "dentex_panoramic" / "val" / "images")
                shutil.copy2(src_img, dst_img_dir / src_img.name)
                disease_counts[disease] += 1

        for disease, count in sorted(disease_counts.items()):
            print(f"  [OK] DENTEX val -> {disease} : {count} images")


# ═══════════════════════════════════════════════════════════════════════
# SMART-OM Integration
# ═══════════════════════════════════════════════════════════════════════
def integrate_smart_om():
    """Integrate SMART-OM oral mucosa images."""
    print("\n" + "=" * 70)
    print("Integrating SMART-OM dataset")
    print("=" * 70)

    # Map SMART-OM categories to our diseases
    # 01. Normal -> Healthy
    # 02. Variation from normal -> could contain various conditions
    # 03. OPMD (Oral Potentially Malignant Disorders) -> Oral Cancer (pre-cancerous)
    # 04. Oral Cancer -> Oral Cancer

    category_map = {
        "01. Normal": "11_Healthy",
        "02. Variation from normal": "11_Healthy",  # variations are still essentially healthy tissue
        "03. OPMD": "06_Oral_Cancer",
        "04. Oral Cancer": "06_Oral_Cancer",
    }

    for category_dir in sorted(SMART_OM_BASE.iterdir()):
        if not category_dir.is_dir():
            continue

        cat_name = category_dir.name
        disease = category_map.get(cat_name)
        if not disease:
            continue

        # Use unannotated images (raw photos)
        unannotated_dir = category_dir / "01. Unannotated"
        if not unannotated_dir.exists():
            continue

        subfolder = cat_name.replace(" ", "_").replace(".", "")
        dst_base = DST / disease / f"smart_om_{subfolder}"

        count = 0
        for region_dir in sorted(unannotated_dir.iterdir()):
            if not region_dir.is_dir():
                continue

            region_name = region_dir.name.replace(" ", "_")
            dst_dir = ensure_dir(dst_base / region_name)

            for img_file in region_dir.iterdir():
                if img_file.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    shutil.copy2(img_file, dst_dir / img_file.name)
                    count += 1

        # Also copy JSON annotations if available
        for ann_type in ["02. Region annotation", "04. Lesion annotation"]:
            ann_dir = category_dir / ann_type
            if not ann_dir.exists():
                continue

            for sub in ann_dir.iterdir():
                if sub.is_dir() and "json" in sub.name.lower():
                    dst_json = ensure_dir(dst_base / "annotations")
                    for jf in sub.iterdir():
                        if jf.suffix.lower() == ".json":
                            shutil.copy2(jf, dst_json / jf.name)

        print(f"  [OK] SMART-OM {cat_name} -> {disease} : {count} images")

    # Copy metadata
    metadata_dir = SMART_OM_BASE / "Metadata"
    if metadata_dir.exists():
        for f in metadata_dir.iterdir():
            if f.is_file():
                dst_meta = ensure_dir(DST / "_SMART_OM_Metadata")
                shutil.copy2(f, dst_meta / f.name)
                print(f"  [OK] Copied metadata: {f.name}")


def main():
    print("Integrating new datasets into disease-based structure\n")
    integrate_dentex()
    integrate_smart_om()
    print("\nDone!")


if __name__ == "__main__":
    main()
