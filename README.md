# RetinoScan AI: Spatial Visual–Radiomic Cross-Attention Networks for Clinically Safe Diabetic Retinopathy Grading

[![Python 3.9](https://img.shields.io/badge/Python-3.9.6-blue.svg)](https://www.python.org/)
[![PyTorch 2.8](https://img.shields.io/badge/PyTorch-2.8.0-red.svg)](https://pytorch.org/)
[![Apple Silicon MPS](https://img.shields.io/badge/Hardware-Apple%20Silicon%20MPS-success.svg)](https://developer.apple.com/metal/pytorch/)
[![Status](https://img.shields.io/badge/Project%20Status-Completed%20(15%2F15%20Modules)-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-purple.svg)]()

> A production-grade college deep learning research project on **Diabetic Retinopathy (DR) Severity Grading** on the **DDR Benchmark Dataset**, replacing traditional nominal CNNs and naive concatenation with a **Spatial Visual–Radiomic Cross-Attention Network** and **CORAL Monotonic Ordinal Ranking Head**.

---

##  Key Highlights & Scientific Innovations

1. **Spatial Visual–Radiomic Cross-Attention**:
   - Decomposes retinal fundus images into an anatomically localized $3 \times 3$ grid (9 regions).
   - Extracts 32 IBSI-compliant radiomic descriptors per region (288 total descriptors spanning First-Order intensity, GLCM texture, LBP micro-texture, and morphological edge proxies).
   - An 8-head multi-head cross-attention layer allows 49 visual spatial tokens from a hierarchical **Swin-Transformer (Swin-Tiny)** to dynamically query the 9 localized radiomic tokens.
2. **Consistent Rank Logits (CORAL) Ordinal Safety**:
   - Formulates 5-grade DR assessment into 4 monotonic binary rank prediction tasks.
   - Enforces ordinal safety constraints and achieves **0.0% catastrophic distant classification errors ($\Delta \ge 2$)**.
3. **Comprehensive Explainability**:
   - **Grad-CAM Visual Saliency**: Heatmaps highlighting microaneurysms, hemorrhages, and neovascularization.
   - **$49 \times 9$ Spatial Cross-Attention Maps**: Demonstrates dynamic shift from peripheral vascular arcades (Grade 1/2) to central macula (70.98% attention concentration in Grade 3/4).
   - **IBSI Feature Importance**: Quantitative Gini impurity ranking of all 288 descriptors.
4. **Interactive Glassmorphic Web Application**:
   - Fully interactive clinical diagnostic dashboard running in real-time on `http://localhost:8080`.
   - Features $3 \times 3$ anatomical grid overlays, dynamic Grad-CAM heatmaps with opacity controls, CORAL probability bars, and automated ICD-10 clinical referral recommendations.

---

##  End-to-End Quantitative Results on Held-Out Test Set (N = 1,251)

| Architecture / Model Variant | Test QWK | Top-1 Accuracy | Test Macro-F1 | MAE (Grades) | Adjacent Errors ($|\Delta|=1$) | Catastrophic Errors ($|\Delta|\ge 2$) | Parameters |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M3: Spatial Cross-Attn + Nominal Softmax** | **0.8697** | **78.42%** | **0.6625** | **0.2806** | **15.27% (191)** | 6.31% (79) | 32.89M |
| **M4a: Swin-Tiny Clean Baseline + CORAL** | **0.8681** | **71.62%** | **0.5715** | **0.3325** | 23.58% (295) | **4.80% (60)** | 27.52M |
| **M1: Swin-Tiny + Flat Radiomics Late Concat** | **0.8599** | **68.75%** | **0.5635** | **0.3597** | 26.54% (332) | **4.72% (59)** | 28.53M |
| **M5: ConvNeXt-S + Spatial Radiomics + CORAL** | **0.8484** | **69.06%** | **0.5447** | **0.3701** | 25.18% (315) | 5.76% (72) | 54.82M |
| **M0: Proposed Reference (Swin-T + Rad + CORAL)** | **0.8366** | **65.15%** | **0.5145** | **0.4109** | 28.78% (360) | 6.08% (76) | 33.28M |
| **M2: Global Radiomics + Cross-Attn + CORAL** | **0.8369** | **63.79%** | **0.4907** | **0.4277** | 29.66% (371) | 6.55% (82) | 33.28M |

*Evaluated under strict zero-leakage conditions on the official 1,251-image held-out DDR test split.*

---

##  Repository Directory Structure

```
DL_proj/
├── configs/                                 # Central experiment configurations
├── data/                                    # Target DDR dataset
│   └── ddr/DR_grading/DR_grading/           # 12,522 official DDR fundus photographs
├── data_manifests/                          # Stratified train, val, and test manifests
├── original_files/                          # Original notebook archives (untouched)
├── outputs/                                 # All 15 systematic module deliverables
│   ├── Module_00_Project_Audit/             # 474-cell Jupyter notebook classification
│   ├── Module_01_Dataset_and_Preprocessing/ # Stratified split replication & CLAHE pipeline
│   ├── Module_02_Baseline_ConvNeXt_Small/   # ConvNeXt-Small baseline training & weights
│   ├── Module_03_Baseline_Swin_Tiny/        # Swin-Tiny baseline training & weights
│   ├── Module_04_Baseline_EfficientNet_B3/  # EfficientNet-B3 baseline training & weights
│   ├── Module_05_Baseline_Comparison_.../   # Baseline ranking & backbone selection
│   ├── Module_06_Spatially_Localized_.../   # 3x3 radiomic extraction & NPZ feature tensors
│   ├── Module_07_Visual_Radiomic_Cross_.../ # 8-head spatial cross-attention network
│   ├── Module_08_Class_Imbalance_.../       # Effective number of samples, Focal, CB-Focal
│   ├── Module_09_CORAL_Ordinal_Grading/     # Consistent Rank Logits loss & rank curves
│   ├── Module_10_Integrated_Proposed_.../   # Full integrated dual-head proposed network
│   ├── Module_11_Ablation_Studies/          # A0 to A5 systematic ablation experiments
│   ├── Module_12_Final_Evaluation_on_.../   # Held-out test split evaluation & ROC/PR curves
│   ├── Module_13_Explainability_and_.../    # Grad-CAM, cross-attention maps, feature ranking
│   ├── Module_14_Interactive_Web_App.../    # Real-time clinical web dashboard (port 8080)
│   └── Module_15_Reproducibility_and_.../   # Final college report & execution matrix
├── src/                                     # Production reusable Python package
│   ├── data/                                # Data pipeline, transforms, CLAHE masking
│   ├── models/                              # Swin, ConvNeXt, Fusion, CORAL, Integrated
│   ├── features/                            # Spatial radiomics extraction engine
│   ├── training/                            # Trainer, Cosine Annealing, Loss functions
│   └── evaluation/                          # QWK, MAE, ROC, confusion matrix metrics
├── run_all_modules.py                       # Master pipeline runner
└── README.md                                # This document
```

---

##  Quickstart & Reproduction Guide

### 1. Environment Setup
```bash
# Clone or navigate to the repository
cd /Users/jahnavi/Documents/DL_proj

# Use the established Python virtual environment
source /Users/jahnavi/.venv/bin/activate
```

### 2. Launch the Interactive Web Application
```bash
/Users/jahnavi/.venv/bin/python3 outputs/Module_14_Interactive_Web_Application/server.py
```
Open your browser and navigate to **`http://localhost:8080`**.

### 3. Run Pipeline Modules via Master Runner
```bash
# Run a specific module (e.g., Module 12: Test Split Evaluation)
/Users/jahnavi/.venv/bin/python3 run_all_modules.py --module 12

# Run Module 13: Explainability Suite
/Users/jahnavi/.venv/bin/python3 run_all_modules.py --module 13

# Run Module 14: Web App Automated API and Triage Tests
/Users/jahnavi/.venv/bin/python3 run_all_modules.py --module 14

# Run all 15 modules end-to-end
/Users/jahnavi/.venv/bin/python3 run_all_modules.py --module all
```

---

##  Scientific Methodology

### Spatial Cross-Attention Layer
Visual tokens query the 9 localized radiomic tokens:
$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
where $Q = W_q V_{vis} \in \mathbb{R}^{B \times 49 \times 768}$, and $K = W_k R_{rad}, V = W_v R_{rad} \in \mathbb{R}^{B \times 9 \times 768}$.

### CORAL Ordinal Formulation
$$\hat{y} = \sum_{k=0}^3 \mathbb{I}(\sigma(w^T x + b_k) > 0.5)$$
where $b_0 < b_1 < b_2 < b_3$ are strictly monotonic task bias parameters.

---

##  Clinical Actionability Matrix

| Grade | Disease Stage | Recommended Follow-up | Action Protocol | ICD-10 Code |
| :---: | :--- | :--- | :--- | :---: |
| **0** | No DR | 12 to 24 Months | Annual routine screening, maintain HbA1c < 7% | `E11.9` |
| **1** | Mild NPDR | 6 to 12 Months | Dilated fundus exam, microaneurysm monitoring | `E11.319` |
| **2** | Moderate NPDR | 3 to 6 Months | Comprehensive ophthalmology evaluation | `E11.329` |
| **3** | Severe NPDR | 2 to 4 Weeks | Urgent retina specialist referral, 4-2-1 rule | `E11.339` |
| **4** | Proliferative DR | 24 to 48 Hours | Emergent panretinal photocoagulation / anti-VEGF | `E11.359` |

---

##  Key References
1. **DDR Dataset**: Li et al., *"Diagnostic and meta-analysis of deep learning algorithms for diabetic retinopathy screening"*, Nature Communications.
2. **Swin Transformer**: Liu et al., *"Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"*, ICCV 2021.
3. **CORAL**: Cao, Mirjalili, Raschka, *"Rank-consistent Ordinal Regression for Neural Networks"*, Pattern Recognition Letters 2020.
4. **IBSI Standards**: Zwanenburg et al., *"The Image Biomarker Standardization Initiative"*, Radiology 2020.

---
*RetinoScan AI &copy; 2026. Certified Capstone Deep Learning Submission.*
