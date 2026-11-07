# Module 15: Reproducibility & Verification Checklist
**Diabetic Retinopathy Spatial Visual–Radiomic Cross-Attention Project (RetinoScan AI)**

---

## 1. Environment & Hardware Specifications

| Component | Specification | Verification Command / Location |
| :--- | :--- | :--- |
| **Operating System** | macOS (Darwin 24.6.0, Apple Silicon) | `uname -a` |
| **Python Environment** | Python 3.9.6 | `/Users/jahnavi/.venv/bin/python3` |
| **Deep Learning Framework** | PyTorch 2.8.0 + Torchvision 0.23.0 | `torch.__version__` |
| **Image Processing** | OpenCV 4.12.0 + Scikit-Image 0.24.0 | `cv2.__version__`, `skimage.__version__` |
| **Machine Learning** | Scikit-Learn 1.6.1 | `sklearn.__version__` |
| **Hardware Accelerator** | Apple Silicon MPS (`torch.device("mps")`) | `torch.backends.mps.is_available()` |
| **Random Seeds** | Fixed globally: `seed=42` | `torch.manual_seed(42)`, `np.random.seed(42)` |

---

## 2. Dataset Split & Integrity Checklist

- [x] **Stratified Real DDR Split**: Canonical 80/10/10 patient-level split across all 5 DR grades ($N = 12,522$):
  - Training: **10,024 fundus photographs** (`data_manifests/split_80_10_10/train_manifest.csv`)
  - Validation: **1,247 fundus photographs** (`data_manifests/split_80_10_10/val_manifest.csv`)
  - Held-out Test: **1,251 fundus photographs** (`data_manifests/split_80_10_10/test_manifest.csv`)
- [x] **Zero Test Data Leakage**: Held-out test split (`test_manifest.csv` and `radiomics_test.npz`) was strictly preserved and evaluated only once under locked conditions in Module 12.
- [x] **Reproducible Preprocessing Pipeline**: Retina circle cropping, CLAHE enhancement in LAB color space, resolution standardization (224x224), and ImageNet normalization.
- [x] **Feature Normalization**: Radiomics StandardScaler fitted strictly on the training partition (`outputs/gap3_spatial_radiomics_cross_attention/cache/scaler.npz`) and applied identically to validation and test splits without leakage.

---

## 3. Checkpoint & Artifact Directory Checklist

| Module / Model Key | Model Architecture Description | Checkpoint Path | Parameters | Status |
| :--- | :--- | :--- | :---: | :---: |
| **Module 02 Baseline** | ConvNeXt-Small Baseline (Full DDR) | `outputs/Module_02_Baseline_ConvNeXt_Small/full_ddr/best_model.pth` | 49.46M | Verified (566.4 MB) |
| **Module 03 Baseline** | Swin-Tiny + CORAL Clean Baseline | `outputs/Module_03_Baseline_Swin_Tiny/clean_run/best_model_clean.pth` | 27.52M | Verified (315.4 MB) |
| **Module 04 Baseline** | EfficientNet-B3 Baseline (Full DDR) | `outputs/Module_04_Baseline_EfficientNet_B3/full_ddr/best_model.pth` | 10.70M | Verified (47.0 MB) |
| **M0 (Proposed Reference)** | Swin-Tiny + 9 Spatial Radiomic Tokens + Cross-Attn + CORAL | `outputs/gap3_spatial_radiomics_cross_attention/best_spatial_radiomics_coral_model.pth` | 33.28M | Verified (395.1 MB) |
| **M1 (Fusion Ablation)** | Swin-Tiny + Flat Radiomics Late Concat + CORAL | `outputs/Module_11_Ablation_Studies/M1_Late_Concatenation/best_model.pth` | 28.53M | Verified (326.1 MB) |
| **M2 (Spatial Ablation)** | Swin-Tiny + 1 Global Radiomics Token + Cross-Attn + CORAL | `outputs/Module_11_Ablation_Studies/M2_Global_Radiomics/best_model.pth` | 33.28M | Verified (395.1 MB) |
| **M3 (Ordinal Ablation)** | Swin-Tiny + 9 Spatial Radiomic Tokens + Nominal Softmax | `outputs/Module_11_Ablation_Studies/M3_Nominal_Softmax/best_model.pth` | 32.89M | Verified (394.8 MB) |
| **M4a (Visual-Only)** | Swin-Tiny Clean Baseline + CORAL Head | `outputs/Module_11_Ablation_Studies/M4a_Visual_Only/best_model.pth` | 27.52M | Verified (315.4 MB) |
| **M5 (Backbone Ablation)** | ConvNeXt-S (7x7 feat) + 9 Spatial Radiomics + Cross-Attn + CORAL | `outputs/Module_11_Ablation_Studies/M5_ConvNeXt_Multimodal/best_model.pth` | 54.82M | Verified (628.0 MB) |

---

## 4. Master Pipeline Execution Verification

The entire repository pipeline can be verified or re-executed with the modular master runner:
```bash
# Verify specific module (e.g., Module 12 test evaluation)
/Users/jahnavi/.venv/bin/python3 outputs/Module_12_Final_Evaluation_on_Held_Out_Test_Split/evaluate_held_out_test_set.py

# Launch interactive clinical triage web server (Module 14)
/Users/jahnavi/.venv/bin/python3 outputs/Module_14_Interactive_Web_Application/server.py
```

---
*Checklist verified and confirmed under Project Governance Rules.*
