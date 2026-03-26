# OraScan Disease Classification Model Training

End-to-end pipeline for training a dental disease classification model for the [OraScan](https://github.com/Cvv9) oral scanning kiosk. Classifies intraoral photographs into **11 disease categories** using transfer learning with EfficientNet-B0.

## Model Performance

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **94.7%** |
| **Best Validation Accuracy** | **95.2%** |
| **Model Architecture** | EfficientNet-B0 (4.67M parameters) |
| **Input Size** | 224 x 224 RGB |
| **Inference Backend** | ONNX Runtime + DirectML (AMD GPU) |

### Per-Class Performance (Test Set)

| Disease | Precision | Recall | F1-Score | Test Samples |
|---------|-----------|--------|----------|-------------|
| Caries | 97.7% | 98.5% | 98.1% | 259 |
| Tooth Discoloration | 100% | 98.5% | 99.2% | 201 |
| Hypodontia | 97.6% | 99.2% | 98.4% | 125 |
| Mouth Ulcer | 96.8% | 98.9% | 97.9% | 279 |
| Healthy | 99.2% | 96.4% | 97.8% | 526 |
| Tooth Crack | 100% | 92.3% | 96.0% | 13 |
| Periodontal Disease | 91.3% | 95.5% | 93.3% | 22 |
| Oral Cancer | 85.4% | 89.5% | 87.4% | 124 |
| Gingivitis | 91.6% | 74.1% | 81.9% | 162 |
| Calculus | 64.4% | 91.6% | 75.6% | 83 |
| Mucocele | 100% | 60.0% | 75.0% | 5 |

---

## Table of Contents

1. [Step-by-Step Guide: What Was Done](#step-by-step-guide-what-was-done)
2. [Project Structure](#project-structure)
3. [Scripts Overview](#scripts-overview)
4. [How to Use the Model](#how-to-use-the-model)
5. [Integrating with OraScan App](#integrating-with-orascan-app)
6. [Retraining the Model](#retraining-the-model)
7. [Future Improvements](#future-improvements)
8. [Requirements](#requirements)

---

## Step-by-Step Guide: What Was Done

### Phase 1: Dataset Organization (Format-Based)

**Script:** `organize_datasets.py`

1. Started with 17 raw ZIP files at `E:\ECG and Dental Images\Dental data set\` totaling ~15GB
2. Extracted all ZIPs and audited every dataset for format, class labels, and annotation types
3. Organized **68,933 images** into a format-based structure:
   - `Classification/` - Folder-based labeled images
   - `Classification_CSV/` - CSV-labeled images
   - `Detection_YOLO/` - YOLO bounding box annotations
   - `Detection_COCO/` - COCO JSON annotations (31-class panoramic X-rays)
   - `Detection_VOC/` - Pascal VOC XML annotations
   - `Segmentation/` - Pixel-level mask annotations
   - `Captioning/` - Images with text captions
   - `Raw_Unlabeled/` - Patient photos needing annotation

### Phase 2: Disease-Based Reorganization

**Script:** `reorganize_by_disease.py` (13 phases)

Reorganized everything by disease type instead of annotation format:

| Phase | Description |
|-------|-------------|
| 1 | Classification folder images -> disease folders |
| 2 | CSV-labeled images split by disease label |
| 3 | Single-disease YOLO datasets (Dental Cavity, Gingivitis Severity) |
| 4 | Multi-disease YOLO datasets split per-disease with remapped class IDs |
| 5 | Pascal VOC detection datasets by disease |
| 6 | Segmentation masks analyzed for per-pixel disease classes |
| 7 | Captioning data linked to disease folders |
| 8 | COCO 31-class panoramic X-rays filtered by disease categories |
| 9 | Panoramic YOLO datasets (Kennedy classification) |
| 10 | Dental anatomy reference data |
| 11 | Raw unlabeled patient data |
| 12 | Auto-generated `data.yaml` configs for detection folders |
| 13 | README documentation |

**Key challenges solved:**
- **Multi-class splitting:** Parsed YOLO label files per-image to extract disease-specific annotations and remap class IDs to 0 for single-disease training
- **COCO extraction:** Filtered 94,794 annotations across 31 categories into disease-relevant subsets
- **Segmentation analysis:** Used PIL/NumPy to read mask pixel values and route images to correct disease folders
- **Labelled_Ulcer_Calculus bug:** `data.yaml` claimed class IDs 0,1 but actual labels used IDs 7,9 - discovered and fixed
- **Turkish DENTEX labels:** Decoded Turkish dental codes (curuk=caries, kuretaj=periodontal, gomulu=impacted)

### Phase 3: Bug Fixes

**Script:** `fix_phases.py`

- Fixed Phase 4: Intraoral/Labelled datasets use `images/{split}/` structure (not `{split}/images/`)
- Fixed Phase 6: Segmentation masks use `{stem}_mask.png` naming convention
- Created `data.yaml` configs for newly generated detection folders

### Phase 4: External Dataset Integration

**Script:** `integrate_new_datasets.py`

Downloaded and integrated two external datasets to supplement weak categories:

| Dataset | Source | Size | Categories Added |
|---------|--------|------|-----------------|
| DENTEX | HuggingFace | 10.4GB | Caries (+941), Hypodontia (+361), Periodontal (+81) |
| SMART-OM | Figshare | 959MB | Healthy (+2,324), Oral Cancer (+145) |

**Final disease-based structure: 78,058 images across 11 diseases**

### Phase 5: Preprocessing for Training

**Script:** `preprocess_for_training.py`

1. Collected classification-suitable images from each disease folder
2. Extracted crack region crops from segmentation masks (for Tooth_Crack which had 0 classification images)
3. Deduplicated by content hash (MD5 of first 64KB)
4. Resized all images to 224x224 with aspect-ratio-preserving center crop
5. Created stratified train/val/test splits (80/10/10)
6. Output: **17,958 images** in ImageFolder format

### Phase 6: Model Training

**Script:** `train_classifier.py`

1. **Architecture:** EfficientNet-B0 pretrained on ImageNet
2. **Custom head:** Dropout(0.3) -> Linear(1280, 512) -> ReLU -> Dropout(0.2) -> Linear(512, 11)
3. **Class balancing:** WeightedRandomSampler (inverse frequency) + class-weighted CrossEntropyLoss
4. **Data augmentation:** RandomResizedCrop, HorizontalFlip, VerticalFlip, Rotation, ColorJitter, Affine, Grayscale, RandomErasing
5. **Phase 1 (epochs 1-5):** Frozen backbone, train head only at LR=3e-3
6. **Phase 2 (epochs 6-25):** Full fine-tuning at LR=3e-4 with cosine annealing
7. **Early stopping:** Patience=7 (not triggered - model kept improving)
8. **Best model saved** at epoch 25 with 95.2% validation accuracy

### Phase 7: Export & Verification

1. Exported best model to ONNX format (opset 17)
2. Verified ONNX model loads with DirectML provider (AMD GPU)
3. Tested inference produces correct output shape
4. Saved model metadata (class names, normalization params, input size)

---

## Project Structure

```
OraScan_Disease_Model_Training/
  organize_datasets.py          # Phase 1: Format-based organization
  reorganize_by_disease.py      # Phase 2: Disease-based reorganization
  fix_phases.py                 # Phase 3: Bug fixes for YOLO/segmentation
  integrate_new_datasets.py     # Phase 4: DENTEX + SMART-OM integration
  preprocess_for_training.py    # Phase 5: Create training splits
  train_classifier.py           # Phase 6-7: Train + export model
  inference.py                  # Production inference script
  requirements.txt              # Python dependencies
  README.md                     # This file
```

**Output locations on disk:**
```
E:\ECG and Dental Images\Dental data set\
  Organized/                    # Format-based structure (Phase 1)
  By_Disease/                   # Disease-based structure (Phase 2-4)
    01_Caries/                  #   15,061 images
    02_Gingivitis/              #    7,107 images
    03_Mouth_Ulcer/             #    3,119 images
    ...
    11_Healthy/                 #    8,204 images
  Training_Ready/               # Preprocessed splits (Phase 5)
    train/<disease>/*.jpg       #   14,362 images
    val/<disease>/*.jpg         #    1,796 images
    test/<disease>/*.jpg        #    1,800 images
  Models/                       # Trained model artifacts (Phase 6-7)
    best_model.pth              #   PyTorch checkpoint
    dental_classifier.onnx      #   ONNX model for production
    model_metadata.json         #   Class names, normalization params
    evaluation_report.txt       #   Full test evaluation
    training_history.json       #   Loss/accuracy per epoch
```

---

## Scripts Overview

### 1. `organize_datasets.py`
Unzips and organizes raw dental datasets into a format-based folder structure. Run this first on fresh data.

### 2. `reorganize_by_disease.py`
Reorganizes the format-based structure into disease-based folders. Handles all annotation formats (YOLO, COCO, VOC, segmentation masks, CSV labels).

### 3. `fix_phases.py`
Fixes edge cases discovered after initial reorganization:
- Multi-disease YOLO datasets with non-standard directory structures
- Segmentation masks with `_mask` suffix naming convention

### 4. `integrate_new_datasets.py`
Integrates externally downloaded datasets (DENTEX panoramic X-rays, SMART-OM oral mucosa photos) into the disease-based structure.

### 5. `preprocess_for_training.py`
Creates the final training dataset:
- Collects classification-suitable images per disease
- Extracts crack crops from segmentation masks
- Deduplicates, resizes to 224x224
- Creates stratified 80/10/10 splits

### 6. `train_classifier.py`
Full training pipeline:
- EfficientNet-B0 with transfer learning
- Two-phase training (frozen head, then full fine-tune)
- Weighted sampling + class-weighted loss for imbalance
- Exports to ONNX for AMD GPU inference

### 7. `inference.py`
Production inference script:
- Loads ONNX model with DirectML (AMD GPU) or CPU fallback
- Single image or batch folder processing
- Returns disease predictions with confidence scores
- Exports results to CSV

---

## How to Use the Model

### Command-Line Inference

```bash
# Single image
python inference.py path/to/dental_image.jpg

# Single image with top-5 predictions
python inference.py path/to/dental_image.jpg --top-k 5

# Batch process a folder
python inference.py --folder path/to/images/ --output results.csv
```

### Python API Usage

```python
from inference import load_model, predict

# Load model (uses AMD GPU via DirectML if available)
session, metadata = load_model()

# Classify an image
results, inference_time_ms = predict(session, metadata, "dental_photo.jpg", top_k=3)

for r in results:
    print(f"{r['class']}: {r['confidence']:.1%}")
# Output:
#   Caries: 92.3%
#   Gingivitis: 4.1%
#   Healthy: 2.8%
```

### Model Files Required

The following files must exist at `E:\ECG and Dental Images\Dental data set\Models\`:
- `dental_classifier.onnx` - The model weights
- `model_metadata.json` - Class names and preprocessing parameters

To use a different path, modify `MODEL_DIR` in `inference.py`.

---

## Integrating with OraScan App

The OraScan app captures 13 intraoral photos per scan session. Here is how to integrate the disease classifier:

### Option A: Local Inference After Capture (Recommended)

Modify `oral_photo_acquisition_page.py` in `OraScan_Automated_Scanning/`:

```python
# Add to imports at the top of oral_photo_acquisition_page.py
import sys
sys.path.insert(0, "path/to/OraScan_Disease_Model_Training")
from inference import load_model, predict

# Load model once at module level
_session, _metadata = load_model()

# In handle_submit() after collecting local_paths:
def handle_submit():
    # ... existing code to collect local_paths ...

    # Run disease classification on each captured image
    scan_results = []
    for img_path in local_paths:
        results, elapsed = predict(_session, _metadata, str(img_path), top_k=3)
        scan_results.append({
            "image": img_path.name,
            "top_prediction": results[0]["class"],
            "confidence": results[0]["confidence"],
            "all_predictions": results,
        })

    # Generate report summary
    diseases_found = set()
    for r in scan_results:
        if r["confidence"] > 0.5 and r["top_prediction"] != "Healthy":
            diseases_found.add(r["top_prediction"])

    if diseases_found:
        summary = "Potential conditions detected: " + ", ".join(sorted(diseases_found))
    else:
        summary = "No significant conditions detected. Oral health appears normal."

    # Store in reports table
    from database_utils import add_report
    add_report(patient_id, summary, scan_results)

    # ... existing upload code ...
```

### Option B: Backend Processing

Add an inference endpoint to the Go backend (`OraScan_backend/`):

1. Copy `dental_classifier.onnx` and `model_metadata.json` to the backend's model directory
2. Use the Go ONNX Runtime bindings or call the Python inference script as a subprocess
3. Process images after they arrive at the `/upload` endpoint
4. Store results in the database and return them to the frontend

### Option C: Dedicated FastAPI Inference Server

Create a lightweight API server alongside the main app:

```python
# inference_server.py
from fastapi import FastAPI, UploadFile
from inference import load_model, predict
import tempfile

app = FastAPI()
session, metadata = load_model()

@app.post("/classify")
async def classify(file: UploadFile):
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    results, elapsed = predict(session, metadata, tmp_path, top_k=3)
    return {"predictions": results, "inference_time_ms": elapsed}
```

Then call from the existing app:
```python
import httpx
async with httpx.AsyncClient() as client:
    response = await client.post("http://localhost:8001/classify",
                                  files={"file": open(img_path, "rb")})
    predictions = response.json()["predictions"]
```

---

## Retraining the Model

### Adding New Data

1. Place new images in the appropriate disease folder under `By_Disease/`:
   ```
   E:\ECG and Dental Images\Dental data set\By_Disease\01_Caries\classification\new_images\
   ```

2. Re-run preprocessing:
   ```bash
   python preprocess_for_training.py
   ```

3. Retrain:
   ```bash
   python train_classifier.py --epochs-frozen 3 --epochs-finetune 15
   ```

### Resuming Training

```bash
python train_classifier.py --resume "E:\ECG and Dental Images\Dental data set\Models\best_model.pth"
```

### Adjusting Hyperparameters

```bash
# Larger batch size (if RAM allows)
python train_classifier.py --batch-size 128

# More fine-tuning epochs
python train_classifier.py --epochs-finetune 30

# Lower learning rate for stability
python train_classifier.py --lr-finetune 1e-4

# More patience before early stopping
python train_classifier.py --patience 10
```

---

## Future Improvements

### High Priority

1. **More Mucocele data (currently 48 images, 60% recall)**
   - Collect more mucocele clinical photos
   - Try data augmentation specifically for this class (CutMix, MixUp)
   - Consider merging with a broader "oral lesion" category

2. **Gingivitis/Calculus confusion (39 gingivitis images misclassified as calculus)**
   - These conditions are visually similar and often co-occur
   - Collect higher-resolution, better-labeled images distinguishing the two
   - Consider a hierarchical classifier: first detect "gum disease" then sub-classify

3. **Oral Cancer accuracy (87.4% F1)**
   - This is the most clinically critical class
   - Augment with more OPMD/cancer datasets
   - Consider a binary "cancer screening" model with high sensitivity

### Medium Priority

4. **Upgrade to EfficientNet-B2 or B4 for higher accuracy**
   - Requires GPU training (not practical on CPU)
   - Install PyTorch with ROCm on Linux, or use cloud GPU
   - Expected improvement: 2-5% accuracy gain

5. **Add object detection for localization**
   - The current model classifies whole images
   - YOLO-based detection would highlight WHERE the disease is in the image
   - Useful for the kiosk to show patients which area has an issue
   - The `By_Disease/` folders already contain YOLO and COCO annotations for this

6. **Ensemble multiple models**
   - Train separate specialist models for confusing pairs (gingivitis vs calculus)
   - Combine predictions with a meta-classifier

7. **Test-time augmentation (TTA)**
   - Run inference on multiple augmented versions of each image
   - Average predictions for more robust results
   - Easy to implement, typically gives 1-2% accuracy boost

### Low Priority / Long-Term

8. **GPU-accelerated training**
   - Current training runs on CPU (~4.5 hours for 25 epochs)
   - Options: Linux with ROCm, cloud GPU (Colab/AWS/Azure), or NVIDIA GPU
   - Would enable larger models, larger images, and faster experimentation

9. **Explainability (Grad-CAM)**
   - Add heatmap visualization showing which parts of the image the model focuses on
   - Crucial for clinical trust and regulatory approval
   - PyTorch hooks make this straightforward to add

10. **Confidence calibration**
    - Current confidence scores may not be well-calibrated
    - Apply temperature scaling on the validation set
    - Important for clinical decision support (knowing when the model is uncertain)

11. **Additional disease categories**
    - Dental fluorosis, bruxism damage, enamel erosion
    - Requires new datasets and clinical annotation

12. **Cross-validation**
    - Current evaluation uses a single train/val/test split
    - K-fold cross-validation would give more robust accuracy estimates

---

## Requirements

```
torch>=2.0
torchvision>=0.15
onnxruntime-directml>=1.15
numpy>=1.21
Pillow>=9.0
scikit-learn>=1.0
opencv-python>=4.5
```

Install:
```bash
pip install torch torchvision onnxruntime-directml numpy Pillow scikit-learn opencv-python
```

**Hardware used for training:**
- CPU: AMD (48GB RAM)
- GPU: AMD (used via DirectML for ONNX inference only)
- Training: CPU-only (PyTorch 2.7.1+cpu)
- OS: Windows 11

---

## License

This training pipeline is part of the OraScan project. The datasets used have their own licenses:
- DENTEX: CC BY-NC 4.0 (HuggingFace)
- SMART-OM: CC BY 4.0 (Figshare)
- Other datasets: Various open/research licenses from Kaggle and Roboflow
