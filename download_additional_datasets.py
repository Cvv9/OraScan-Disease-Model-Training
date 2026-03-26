"""
Download additional dental image datasets to boost underrepresented classes.

Target classes (current count -> goal):
  Mucocele:            48  -> 500+
  Tooth_Crack:        128  -> 500+
  Periodontal_Disease: 218 -> 500+
  Calculus:            829 -> 1500+
  Gingivitis:        1,616 -> 2500+

Sources:
  1. Google Drive - Calculus detection dataset (megargayu/dentalclassification)
  2. Mendeley Data - Gingivitis intraoral images
  3. Kaggle dental segmentation (Dentalai) - crack images
  4. Web scraping from open medical image repositories
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve, urlopen, Request
from urllib.error import URLError

DOWNLOAD_DIR = Path(r"E:\ECG and Dental Images\Dental data set\Downloads\additional")
BY_DISEASE = Path(r"E:\ECG and Dental Images\Dental data set\By_Disease")

DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)


def download_file(url, dest, desc=""):
    """Download a file with progress."""
    if dest.exists():
        print(f"  Already exists: {dest.name}")
        return True
    print(f"  Downloading {desc or dest.name}...")
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req) as response, open(dest, "wb") as f:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  {downloaded / 1024 / 1024:.1f} MB / {total / 1024 / 1024:.1f} MB ({pct:.0f}%)", end="", flush=True)
            print()
        return True
    except Exception as e:
        print(f"  Failed: {e}")
        if dest.exists():
            dest.unlink()
        return False


def download_gdrive_folder():
    """Download calculus dataset from Google Drive using gdown."""
    print("\n=== Source 1: Google Drive Calculus Dataset ===")
    dest_dir = DOWNLOAD_DIR / "gdrive_calculus"
    if dest_dir.exists() and any(dest_dir.rglob("*.jpg")) or (dest_dir.exists() and any(dest_dir.rglob("*.png"))):
        print("  Already downloaded")
        return dest_dir

    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run([
            sys.executable, "-m", "gdown",
            "--folder",
            "https://drive.google.com/drive/folders/15VJGucwYyA-duD8c3ZJdYYENSzAD3qbW",
            "-O", str(dest_dir),
        ], check=True, timeout=600)
        return dest_dir
    except Exception as e:
        print(f"  Failed: {e}")
        return None


def download_mendeley_gingivitis():
    """Download gingivitis dataset from Mendeley Data."""
    print("\n=== Source 2: Mendeley Gingivitis Dataset ===")
    dest_dir = DOWNLOAD_DIR / "mendeley_gingivitis"
    if dest_dir.exists() and any(dest_dir.rglob("*.jpg")) or (dest_dir.exists() and any(dest_dir.rglob("*.png"))):
        print("  Already downloaded")
        return dest_dir

    # Mendeley datasets can be downloaded via their API
    # Dataset: 3253gj88rr version 1
    zip_path = DOWNLOAD_DIR / "mendeley_gingivitis.zip"
    url = "https://data.mendeley.com/public-files/datasets/3253gj88rr/files/fe3e29ae-5c3e-4fa8-b495-e99e4b8cc94d/file_downloaded"

    if download_file(url, zip_path, "Mendeley gingivitis dataset"):
        dest_dir.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(dest_dir)
            print(f"  Extracted to {dest_dir}")
            return dest_dir
        except zipfile.BadZipFile:
            print("  Not a valid zip - trying as direct folder download")
            # Mendeley may return individual files, handle accordingly
            return dest_dir if any(dest_dir.rglob("*.*")) else None
    return None


def download_kaggle_dental_segmentation():
    """Try to download Dentalai (dental segmentation) dataset - has crack images."""
    print("\n=== Source 3: Dentalai Crack Dataset ===")
    dest_dir = DOWNLOAD_DIR / "dentalai"
    if dest_dir.exists() and any(dest_dir.rglob("*.jpg")) or (dest_dir.exists() and any(dest_dir.rglob("*.png"))):
        print("  Already downloaded")
        return dest_dir

    # Try Kaggle first
    try:
        result = subprocess.run([
            "kaggle", "datasets", "download",
            "-d", "pawanvalluri/dental-segmentation",
            "-p", str(DOWNLOAD_DIR),
        ], capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            zip_path = DOWNLOAD_DIR / "dental-segmentation.zip"
            if zip_path.exists():
                dest_dir.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(zip_path) as zf:
                    zf.extractall(dest_dir)
                print(f"  Extracted to {dest_dir}")
                return dest_dir
    except Exception as e:
        print(f"  Kaggle download failed: {e}")

    # Fallback: try Dataset Ninja direct download
    print("  Trying Dataset Ninja...")
    zip_url = "https://github.com/dataset-ninja/dentalai/releases/download/v1.0.0/dentalai-DatasetNinja.tar"
    tar_path = DOWNLOAD_DIR / "dentalai.tar"
    if download_file(zip_url, tar_path, "Dentalai dataset"):
        dest_dir.mkdir(parents=True, exist_ok=True)
        try:
            import tarfile
            with tarfile.open(tar_path) as tf:
                tf.extractall(dest_dir)
            return dest_dir
        except Exception as e:
            print(f"  Extract failed: {e}")
    return None


def download_oral_diseases_kaggle():
    """Try to download the Oral Diseases dataset from Kaggle."""
    print("\n=== Source 4: Kaggle Oral Diseases ===")
    dest_dir = DOWNLOAD_DIR / "oral_diseases"
    if dest_dir.exists() and any(dest_dir.rglob("*.jpg")) or (dest_dir.exists() and any(dest_dir.rglob("*.png"))):
        print("  Already downloaded")
        return dest_dir

    try:
        result = subprocess.run([
            "kaggle", "datasets", "download",
            "-d", "salmansajid05/oral-diseases",
            "-p", str(DOWNLOAD_DIR),
        ], capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            zip_path = DOWNLOAD_DIR / "oral-diseases.zip"
            if zip_path.exists():
                dest_dir.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(zip_path) as zf:
                    zf.extractall(dest_dir)
                print(f"  Extracted to {dest_dir}")
                return dest_dir
        else:
            print(f"  Kaggle auth required: {result.stderr[:200]}")
    except Exception as e:
        print(f"  Failed: {e}")
    return None


def download_mod_kaggle():
    """Try Mouth and Oral Diseases (MOD) dataset."""
    print("\n=== Source 5: Kaggle MOD Dataset ===")
    dest_dir = DOWNLOAD_DIR / "mod_dataset"
    if dest_dir.exists() and any(dest_dir.rglob("*.jpg")) or (dest_dir.exists() and any(dest_dir.rglob("*.png"))):
        print("  Already downloaded")
        return dest_dir

    try:
        result = subprocess.run([
            "kaggle", "datasets", "download",
            "-d", "javedrashid/mouth-and-oral-diseases-mod",
            "-p", str(DOWNLOAD_DIR),
        ], capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            zip_path = DOWNLOAD_DIR / "mouth-and-oral-diseases-mod.zip"
            if zip_path.exists():
                dest_dir.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(zip_path) as zf:
                    zf.extractall(dest_dir)
                return dest_dir
        else:
            print(f"  Kaggle auth required: {result.stderr[:200]}")
    except Exception as e:
        print(f"  Failed: {e}")
    return None


def download_figshare_smart_om():
    """Download SMART-OM dataset from Figshare (has oral mucosa conditions)."""
    print("\n=== Source 6: Figshare SMART-OM Dataset ===")
    dest_dir = DOWNLOAD_DIR / "smart_om_v2"
    if dest_dir.exists() and any(dest_dir.rglob("*.jpg")) or (dest_dir.exists() and any(dest_dir.rglob("*.png"))):
        print("  Already downloaded")
        return dest_dir

    # Figshare dataset
    zip_path = DOWNLOAD_DIR / "smart_om_v2.zip"
    url = "https://figshare.com/ndownloader/articles/31341790/versions/1"
    if download_file(url, zip_path, "SMART-OM v2 dataset"):
        dest_dir.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(dest_dir)
            return dest_dir
        except Exception as e:
            print(f"  Extract failed: {e}")
    return None


def count_images(path):
    """Count image files recursively."""
    if not path or not path.exists():
        return 0
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sum(1 for f in path.rglob("*") if f.suffix.lower() in exts)


def organize_downloaded_images(source_dirs):
    """Scan downloaded datasets and copy relevant images to By_Disease folders."""
    print("\n\n=== Organizing Downloaded Images ===")

    # Class name mapping: various folder/label names -> our standard names
    class_mapping = {
        # Calculus
        "calculus": "Calculus", "tartar": "Calculus", "dental_calculus": "Calculus",
        "dental calculus": "Calculus", "Calculus": "Calculus", "Tartar": "Calculus",
        # Gingivitis
        "gingivitis": "Gingivitis", "Gingivitis": "Gingivitis",
        "gum_disease": "Gingivitis", "gum disease": "Gingivitis",
        # Periodontal
        "periodontal": "Periodontal_Disease", "periodontitis": "Periodontal_Disease",
        "Periodontal_Disease": "Periodontal_Disease", "periodontal_disease": "Periodontal_Disease",
        "Periodontal Disease": "Periodontal_Disease", "severe periodontitis": "Periodontal_Disease",
        # Mucocele
        "mucocele": "Mucocele", "Mucocele": "Mucocele", "muccocele": "Mucocele",
        # Tooth Crack
        "crack": "Tooth_Crack", "Crack": "Tooth_Crack", "tooth_crack": "Tooth_Crack",
        "cracked": "Tooth_Crack", "fractured": "Tooth_Crack", "fracture": "Tooth_Crack",
        "Tooth_Crack": "Tooth_Crack",
        # Others we already have plenty of but won't reject
        "caries": "Caries", "Caries": "Caries", "cavity": "Caries",
        "healthy": "Healthy", "Healthy": "Healthy", "normal": "Healthy",
        "hypodontia": "Hypodontia", "Hypodontia": "Hypodontia",
        "mouth_ulcer": "Mouth_Ulcer", "Mouth_Ulcer": "Mouth_Ulcer",
        "ulcer": "Mouth_Ulcer", "Ulcer": "Mouth_Ulcer",
        "oral_cancer": "Oral_Cancer", "Oral_Cancer": "Oral_Cancer",
        "tooth_discoloration": "Tooth_Discoloration", "Tooth_Discoloration": "Tooth_Discoloration",
        "discoloration": "Tooth_Discoloration", "Discoloration": "Tooth_Discoloration",
    }

    # Priority classes we want more of
    priority_classes = {"Mucocele", "Tooth_Crack", "Periodontal_Disease", "Calculus", "Gingivitis"}
    stats = {cls: 0 for cls in priority_classes}

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    for src_dir in source_dirs:
        if not src_dir or not src_dir.exists():
            continue
        print(f"\n  Scanning: {src_dir.name}")

        # Walk through all subdirectories
        for root, dirs, files in os.walk(src_dir):
            folder_name = Path(root).name

            # Try to match folder name to a disease class
            mapped_class = class_mapping.get(folder_name)

            if not mapped_class:
                # Try case-insensitive match
                for key, val in class_mapping.items():
                    if key.lower() == folder_name.lower():
                        mapped_class = val
                        break

            if not mapped_class:
                continue

            # Create target classification folder
            target_dir = BY_DISEASE / mapped_class / "classification_new"
            target_dir.mkdir(parents=True, exist_ok=True)

            for fname in files:
                fpath = Path(root) / fname
                if fpath.suffix.lower() not in exts:
                    continue

                # Hash-based dedup name
                with open(fpath, "rb") as f:
                    h = hashlib.md5(f.read(65536)).hexdigest()[:12]
                dest_name = f"new_{h}{fpath.suffix.lower()}"
                dest_path = target_dir / dest_name

                if not dest_path.exists():
                    shutil.copy2(fpath, dest_path)
                    if mapped_class in stats:
                        stats[mapped_class] += 1

    print(f"\n  New images added to By_Disease:")
    for cls, count in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"    {cls:<25} +{count}")

    return stats


def main():
    print("=" * 60)
    print("Downloading Additional Dental Datasets")
    print("Target: Boost Mucocele, Tooth_Crack, Periodontal_Disease,")
    print("        Calculus, and Gingivitis image counts")
    print("=" * 60)

    source_dirs = []

    # Download from all available sources
    source_dirs.append(download_gdrive_folder())
    source_dirs.append(download_mendeley_gingivitis())
    source_dirs.append(download_kaggle_dental_segmentation())
    source_dirs.append(download_oral_diseases_kaggle())
    source_dirs.append(download_mod_kaggle())
    source_dirs.append(download_figshare_smart_om())

    # Filter out None results
    valid_sources = [d for d in source_dirs if d and d.exists()]
    print(f"\n\nSuccessfully downloaded from {len(valid_sources)}/{len(source_dirs)} sources")

    for src in valid_sources:
        count = count_images(src)
        print(f"  {src.name}: {count} images")

    # Organize into By_Disease structure
    if valid_sources:
        organize_downloaded_images(valid_sources)

    # Print final counts per disease
    print("\n\n=== Updated By_Disease Counts ===")
    for disease_dir in sorted(BY_DISEASE.iterdir()):
        if disease_dir.is_dir() and not disease_dir.name.startswith("."):
            count = sum(1 for f in disease_dir.rglob("*") if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
            marker = " <<<" if disease_dir.name in {"Mucocele", "Tooth_Crack", "Periodontal_Disease", "Calculus", "Gingivitis"} else ""
            print(f"  {disease_dir.name:<25} {count:>6} images{marker}")


if __name__ == "__main__":
    main()
