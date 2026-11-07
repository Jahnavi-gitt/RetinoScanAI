# Module 01: Dataset Validation and Reproducible Preprocessing Report

**Project**: Deep Learning for Diabetic Retinopathy Grading  
**Target Benchmark Dataset**: DDR (`mariaherrerot/ddrdataset`)  
**External Pretraining Dataset**: EyePACS (`mohlamin/resized-eyepacs-diabetic-retinopathy-dataset`)  
**Deliverable Path**: `outputs/Module_01_Dataset_and_Preprocessing/`  
**Execution Date**: October 26, 2026  
**Status**: Completed and Verified with Real DDR Metadata and Schema  

---

## 1. Executive Summary

Module 01 establishes a validated, reproducible, leak-free dataset pipeline based on the real DDR dataset (`mariaherrerot/ddrdataset`) and EyePACS (`mohlamin/resized-eyepacs-diabetic-retinopathy-dataset`).

Using the Kaggle API access token provided by the user (`jah2navi`), the official metadata file [`data/ddr/DR_grading.csv`](file:///Users/jahnavi/Documents/DL_proj/data/ddr/DR_grading.csv) containing all **12,522 real retinal fundus records** was downloaded and audited. The canonical 70% Train (8,765), 15% Val (1,878), and 15% Test (1,879) stratified split was produced, matching Cell 312 of [`dr-ansh (11).ipynb`](file:///Users/jahnavi/Documents/DL_proj/dr-ansh%20(11).ipynb).

---

## 2. Dataset Roles & Architecture

### 2.1 Primary Target Benchmark: DDR (12,522 images)
- **Kaggle Source**: `https://www.kaggle.com/datasets/mariaherrerot/ddrdataset`
- **Role**: Primary dataset for the baseline model competition (ConvNeXt-Small, Swin-Tiny, EfficientNet-B3), local radiomics feature extraction, spatial cross-attention fusion, class imbalance handling, and held-out final evaluation.
- **Why DDR**: In Cell 206 of the project notebook (`### misaligned as the models(eyepac) and radiomics(DDR)`), the team found that radiomics had been extracted from DDR images. Fusing visual tokens and radiomic descriptors requires that both representations originate from the **exact same retinal photograph**.

### 2.2 Domain Pretraining Dataset: EyePACS
- **Kaggle Source**: `https://www.kaggle.com/datasets/mohlamin/resized-eyepacs-diabetic-retinopathy-dataset`
- **Role**: Optional source of retinal domain pretraining weights before fine-tuning on DDR.

---

## 3. Real DDR Class Distribution & Imbalance Audit

Analysis of the ground-truth annotations in `DR_grading.csv` reveals the true distribution across all 12,522 fundus images:

| Numeric Grade | Clinical Diagnosis | Total Samples | Total % | Train (70%) | Val (15%) | Test (15%) |
|:---:|---|:---:|:---:|:---:|:---:|:---:|
| **0** | **No DR** | 6,266 | 50.04% | 4,386 | 940 | 940 |
| **1** | **Mild NPDR** | 630 | 5.03% | 441 | 94 | 95 |
| **2** | **Moderate NPDR** | 4,477 | 35.75% | 3,134 | 671 | 672 |
| **3** | **Severe NPDR** | 236 | 1.88% | 165 | 36 | 35 |
| **4** | **Proliferative DR (PDR)** | 913 | 7.29% | 639 | 137 | 137 |
| **Total** | | **12,522** | **100.0%** | **8,765** | **1,878** | **1,879** |

### Key Clinical & Machine Learning Observations:
1. **Severe Imbalance on Grade 3 (Severe NPDR)**: Grade 3 comprises only 1.88% (236 samples) of the entire dataset. This severely penalizes standard Cross-Entropy loss and empirically justifies **Module 08 (Class-Imbalance Strategies)**.
2. **Clinical Severity Distance**: A standard multiclass model penalizes predicting Grade 0 instead of Grade 1 identically to predicting Grade 0 instead of Grade 4. This medical reality justifies our proposed **Module 09 (CORAL Ordinal Grading)** to enforce monotonic grade boundaries.

---

## 4. Reproducible Stratified Split Strategy

The dataset was split using the exact methodology from Cell 312 of the notebook:
- **Deterministic Pre-sorting**: Sorted alphabetically by `image_id` to guarantee reproducible ordering across platforms.
- **Stratification**: Enforced via `sklearn.model_selection.train_test_split` with `stratify=numeric_label`.
- **Random Seed**: Fixed to `SEED = 42`.
- **Manifests Generated**:
  - `data_manifests/ddr_master_manifest.csv` (12,522 rows)
  - `data_manifests/train_manifest.csv` / `outputs/Module_01_Dataset_and_Preprocessing/split_manifest_train.csv` (8,765 rows)
  - `data_manifests/val_manifest.csv` / `outputs/Module_01_Dataset_and_Preprocessing/split_manifest_val.csv` (1,878 rows)
  - `data_manifests/test_manifest.csv` / `outputs/Module_01_Dataset_and_Preprocessing/split_manifest_test.csv` (1,879 rows)
- **Leakage Protection**:
  - The test manifest (`1,879 samples`) is strictly isolated and will remain untouched until final evaluation (Module 12).
  - Data augmentations (random resized crop, flips, rotation, color jitter) are strictly restricted to the training split.

---

## 5. Standardized Preprocessing Pipeline

```
[Raw Fundus Image]
        │
        ▼
[Retina Circle Crop] ──> Eliminates uninformative black scanner margins (threshold=10, margin=5)
        │
        ▼
[CLAHE in LAB Space] ──> Enhances L-channel contrast (clipLimit=2.0, tileGridSize=(8,8))
        │
        ├───────────────────────────────────────┬───────────────────────────────────────┐
        ▼ (Training Split Only)                 ▼ (Validation & Testing Splits)
[RandomResizedCrop (224×224, scale 0.85-1.0)]   [Resize (224×224) & CenterCrop]
[RandomHorizontalFlip (p=0.5)]                  [ToTensor]
[RandomVerticalFlip (p=0.2)]                    [ImageNet Normalization]
[RandomRotation (±15°)]                                 │
[ColorJitter (B=0.15, C=0.15, S=0.15, H=0.02)]          │
[ToTensor & ImageNet Normalization]                     │
        │                                               │
        ▼                                               ▼
[Train DataLoader: Tensor (B, 3, 224, 224)]    [Val DataLoader: Tensor (B, 3, 224, 224)]
```

---

## 6. Deliverables Inventory

All deliverables for Module 01 have been updated and verified:
- [`outputs/Module_01_Dataset_and_Preprocessing/dataset_audit.csv`](file:///Users/jahnavi/Documents/DL_proj/outputs/Module_01_Dataset_and_Preprocessing/dataset_audit.csv) (12,522 audited records)
- [`outputs/Module_01_Dataset_and_Preprocessing/class_distribution.csv`](file:///Users/jahnavi/Documents/DL_proj/outputs/Module_01_Dataset_and_Preprocessing/class_distribution.csv) (Exact per-grade counts and split allocations)
- [`outputs/Module_01_Dataset_and_Preprocessing/split_manifest_train.csv`](file:///Users/jahnavi/Documents/DL_proj/outputs/Module_01_Dataset_and_Preprocessing/split_manifest_train.csv) (8,765 train samples)
- [`outputs/Module_01_Dataset_and_Preprocessing/split_manifest_val.csv`](file:///Users/jahnavi/Documents/DL_proj/outputs/Module_01_Dataset_and_Preprocessing/split_manifest_val.csv) (1,878 val samples)
- [`outputs/Module_01_Dataset_and_Preprocessing/split_manifest_test.csv`](file:///Users/jahnavi/Documents/DL_proj/outputs/Module_01_Dataset_and_Preprocessing/split_manifest_test.csv) (1,879 test samples)
- [`outputs/Module_01_Dataset_and_Preprocessing/preprocessing_config.yaml`](file:///Users/jahnavi/Documents/DL_proj/outputs/Module_01_Dataset_and_Preprocessing/preprocessing_config.yaml)
- [`outputs/Module_01_Dataset_and_Preprocessing/dataset_inspection.png`](file:///Users/jahnavi/Documents/DL_proj/outputs/Module_01_Dataset_and_Preprocessing/dataset_inspection.png)
- [`outputs/Module_01_Dataset_and_Preprocessing/data_pipeline.py`](file:///Users/jahnavi/Documents/DL_proj/outputs/Module_01_Dataset_and_Preprocessing/data_pipeline.py)
- [`outputs/Module_01_Dataset_and_Preprocessing/STATUS.md`](file:///Users/jahnavi/Documents/DL_proj/outputs/Module_01_Dataset_and_Preprocessing/STATUS.md)
