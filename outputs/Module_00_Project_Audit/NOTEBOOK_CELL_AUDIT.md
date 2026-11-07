# Notebook Cell-by-Cell Audit: `dr-ansh (11).ipynb`
## Executive Summary
- **Total Cells**: 474 (Markdown: 32, Code: 441)
- **Executed Code Cells**: 44
- **Unexecuted Code Cells**: 397
- **Execution Errors**: 3 (Cells 419, 422, 445 due to interactive session variable state)
- **Origin Environment**: Kaggle Notebook (Linux, CUDA GPU, `/kaggle/input`, `/kaggle/working`)

## Functional Section Breakdown

| Cell Range | Primary Topic / Functional Group | Mapped Project Module | Status & Usability |
|---|---|---|---|
| 0–15 | Kaggle Environment Setup & Dataset Upload | Module 00 / External | Environment-specific scripts; not runnable locally without modification. |
| 16–44 | EyePACS Pretraining & Data Loader (Legacy) | Module 01 (Pretraining) | Implements EyePACS loader, FocalLoss, cosine warmup scheduler. Superseded when dataset shifted to DDR. |
| 45–93 | Attention Maps & ResNet-50 CAMs (Legacy) | Module 13 (Explainability) | Circular crop, CLAHE, mean CAM calculation. Useful reference for Grad-CAM. |
| 94–133 | Radiomics, Vessel Extraction, CLAHE & SMOTE | Module 01 & Module 06 | Vessel segmentation, skeletonization, PyRadiomics (107 features), SMOTE. Reusable logic for radiomics. |
| 134–163 | MLP Feature Fusion & 5-Fold Cross-Validation | Module 08 & Module 10 | Early tabular MLP fusion on static features; t-SNE evaluation. Incomplete end-to-end integration. |
| 164–205 | Multi-Backbone Exploration & Embedding Extraction | Modules 02, 03, 04 | Extracts penultimate embeddings from backbones into `.npz` files. |
| 206–219 | Dataset Realignment Notice (EyePACS to DDR) | Module 01 (Dataset Audit) | Explicitly documents the misalignment between EyePACS models and DDR radiomics. Transition to DDR. |
| 220–229 | Baseline: EfficientNet-B3 Fine-Tuning on DDR | Module 04 | Complete fine-tuning pipeline: Config, DDR dataset, timm model init, QWK metric, train/val loops. |
| 230–235 | Baseline: ConvNeXt-Small Fine-Tuning on DDR | Module 02 | Complete fine-tuning pipeline for ConvNeXt-Small on DDR with QWK optimization. |
| 236–243 | Baseline: Swin-Tiny Fine-Tuning on DDR | Module 03 | Complete fine-tuning pipeline for Swin-Tiny on DDR with QWK optimization. |
| 244–259 | Multi-Backbone Embeddings + Radiomics + CORAL | Module 09 & Module 10 | FusionDataset concatenating 3 deep embeddings (1536+768+768) + 107 radiomics, passed to CORAL head. |
| 260–310 | Full Ensemble Grad-CAM++ & Lesion Patch Extraction | Module 06 & Module 13 | Extracts bounding boxes/patches around high-attention retinal regions using Grad-CAM++. |
| 311–336 | Fusion Ablations & Imbalance Weighted Loss | Module 08 & Module 11 | Explores class weighting and removes mean centering from embeddings. |
| 337–358 | Patch-Level Radiomics | Module 06 (Local Radiomics) | Computes radiomic descriptors specifically on extracted lesion patches. Highly relevant to spatial radiomics. |
| 359–390 | Lesion Graph Construction & GATv2 GNN | Module 07 (Alternative) | Builds lesion graphs with torch_geometric; trains GATv2 model. Complex and fragile; cross-attention is cleaner. |
| 391–418 | Exploratory Data Analysis & UMAP Embeddings | Module 01 & Module 12 | 3D UMAP projections and feature distributions across DR grades. |
| 419–455 | Backbone Comparison, Logistic Regression, CORAL | Module 05 & Module 09 | Evaluates backbone representations via Logistic Regression and CORAL; contains runtime errors in interactive state. |
| 456–473 | Clinical Decision Rules, Error Analysis & Plots | Module 12 & Module 14 | Decision rule: >0.85 Auto-accept, 0.65-0.85 Review, <0.65 Specialist; Plotly ordinal error gauge visualizer. |

## Detailed Cell Table

| Cell # | Type | Exec # | Functional Group | Mapped Module | Status | Line Count | First Line Preview |
|---|---|---|---|---|---|---|---|
| 000 | code |  | Kaggle Setup & Dataset Upload | Module 00 / External | Unexecuted Code | 7 | `import os` |
| 001 | code |  | Kaggle Setup & Dataset Upload | Module 00 / External | Unexecuted Code | 21 | `import shutil` |
| 002 | markdown |  | Kaggle Setup & Dataset Upload | Module 00 / External | Markdown Documentation | 1 | `# publishing as data set` |
| 003 | code |  | Kaggle Setup & Dataset Upload | Module 00 / External | Unexecuted Code | 0 | `<empty>` |
| 004 | code |  | Kaggle Setup & Dataset Upload | Module 00 / External | Unexecuted Code | 4 | `%%bash` |
| 005 | code |  | Kaggle Setup & Dataset Upload | Module 00 / External | Unexecuted Code | 2 | `%%bash` |
| 006 | code |  | Kaggle Setup & Dataset Upload | Module 00 / External | Unexecuted Code | 2 | `%%bash` |
| 007 | code |  | Kaggle Setup & Dataset Upload | Module 00 / External | Unexecuted Code | 4 | `%%bash` |
| 008 | code |  | Kaggle Setup & Dataset Upload | Module 00 / External | Unexecuted Code | 2 | `%%bash` |
| 009 | code |  | Kaggle Setup & Dataset Upload | Module 00 / External | Unexecuted Code | 10 | `%%bash` |
| 010 | code |  | Kaggle Setup & Dataset Upload | Module 00 / External | Unexecuted Code | 2 | `%%bash` |
| 011 | code |  | Kaggle Setup & Dataset Upload | Module 00 / External | Unexecuted Code | 0 | `<empty>` |
| 012 | markdown |  | Kaggle Setup & Dataset Upload | Module 00 / External | Markdown Documentation | 1 | `# cross checking upload` |
| 013 | markdown |  | Kaggle Setup & Dataset Upload | Module 00 / External | Markdown Documentation | 0 | `<empty>` |
| 014 | code |  | Kaggle Setup & Dataset Upload | Module 00 / External | Unexecuted Code | 2 | `import os` |
| 015 | code |  | Kaggle Setup & Dataset Upload | Module 00 / External | Unexecuted Code | 4 | `import os` |
| 016 | code |  | EyePACS Exploration & Pretraining (Legacy) | Module 01 (Pretraining / Legacy) | Unexecuted Code | 3 | `import os` |
| 017 | code |  | EyePACS Exploration & Pretraining (Legacy) | Module 01 (Pretraining / Legacy) | Unexecuted Code | 4 | `import os` |
| 018 | code |  | EyePACS Exploration & Pretraining (Legacy) | Module 01 (Pretraining / Legacy) | Unexecuted Code | 20 | `import os, torch` |
| 019 | code |  | EyePACS Exploration & Pretraining (Legacy) | Module 01 (Pretraining / Legacy) | Unexecuted Code | 22 | `import torch, numpy as np` |
| 020 | code |  | EyePACS Exploration & Pretraining (Legacy) | Module 01 (Pretraining / Legacy) | Unexecuted Code | 1 | `!pip install -q timm==0.9.16` |
| 021 | code |  | EyePACS Exploration & Pretraining (Legacy) | Module 01 (Pretraining / Legacy) | Unexecuted Code | 1 | `!ls /kaggle/working/checkpoints_effb3` |
| 022 | code |  | EyePACS Exploration & Pretraining (Legacy) | Module 01 (Pretraining / Legacy) | Unexecuted Code | 34 | `import os` |
| 023 | code |  | EyePACS Exploration & Pretraining (Legacy) | Module 01 (Pretraining / Legacy) | Unexecuted Code | 27 | `import os` |
| 024 | code |  | EyePACS Exploration & Pretraining (Legacy) | Module 01 (Pretraining / Legacy) | Unexecuted Code | 22 | `import os` |
| 025 | code |  | EyePACS Exploration & Pretraining (Legacy) | Module 01 (Pretraining / Legacy) | Unexecuted Code | 25 | `import os` |
| 026 | code |  | EyePACS Exploration & Pretraining (Legacy) | Module 01 (Pretraining / Legacy) | Unexecuted Code | 8 | `import shutil` |
| 027 | code |  | EyePACS Exploration & Pretraining (Legacy) | Module 01 (Pretraining / Legacy) | Unexecuted Code | 85 | `# =============================================================` |
| 028 | code |  | EyePACS Exploration & Pretraining (Legacy) | Module 01 (Pretraining / Legacy) | Unexecuted Code | 107 | `# ======================================================` |
| 029 | code |  | EyePACS Exploration & Pretraining (Legacy) | Module 01 (Pretraining / Legacy) | Unexecuted Code | 70 | `# ======================================================` |
| 030 | code |  | EyePACS Exploration & Pretraining (Legacy) | Module 01 (Pretraining / Legacy) | Unexecuted Code | 13 | `import numpy as np` |
| 031 | code |  | EyePACS Exploration & Pretraining (Legacy) | Module 01 (Pretraining / Legacy) | Unexecuted Code | 29 | `@torch.no_grad()` |
| 032 | code |  | EyePACS Exploration & Pretraining (Legacy) | Module 01 (Pretraining / Legacy) | Unexecuted Code | 313 | `# ======= FULLY FIXED TRAINING CELL (deterministic resume + atomic ckpt + scaler/RNG save/restore) =` |
| 033 | code |  | EyePACS Exploration & Pretraining (Legacy) | Module 01 (Pretraining / Legacy) | Unexecuted Code | 1 | `!pip install -U scikit-learn==1.4.2` |
| 034 | code |  | EyePACS Exploration & Pretraining (Legacy) | Module 01 (Pretraining / Legacy) | Unexecuted Code | 13 | `import pandas as pd` |
| 035 | code |  | EyePACS Exploration & Pretraining (Legacy) | Module 01 (Pretraining / Legacy) | Unexecuted Code | 2 | `!pip install -q git+https://github.com/jacobgil/pytorch-grad-cam.git@master` |
| 036 | code |  | EyePACS Exploration & Pretraining (Legacy) | Module 01 (Pretraining / Legacy) | Unexecuted Code | 1 | `!pip install -U scikit-learn==1.5.2 --quiet` |
| 037 | code |  | EyePACS Exploration & Pretraining (Legacy) | Module 01 (Pretraining / Legacy) | Unexecuted Code | 1 | `# !pip install -U numpy==1.26.4 opencv-python-headless==4.8.1.78 scikit-learn==1.3.2 matplotlib==3.8` |
| 038 | code |  | EyePACS Exploration & Pretraining (Legacy) | Module 01 (Pretraining / Legacy) | Unexecuted Code | 1 | `!pip install -U numpy==1.26.4 scikit-learn==1.4.2 pytorch-grad-cam==1.4.9 --quiet` |
| 039 | code |  | EyePACS Exploration & Pretraining (Legacy) | Module 01 (Pretraining / Legacy) | Unexecuted Code | 2 | `import pytorch_grad_cam` |
| 040 | code |  | EyePACS Exploration & Pretraining (Legacy) | Module 01 (Pretraining / Legacy) | Unexecuted Code | 15 | `import torch` |
| 041 | code |  | EyePACS Exploration & Pretraining (Legacy) | Module 01 (Pretraining / Legacy) | Unexecuted Code | 7 | `# Pick 1–2 random examples per DR grade` |
| 042 | code |  | EyePACS Exploration & Pretraining (Legacy) | Module 01 (Pretraining / Legacy) | Unexecuted Code | 69 | `import os` |
| 043 | code |  | EyePACS Exploration & Pretraining (Legacy) | Module 01 (Pretraining / Legacy) | Unexecuted Code | 47 | `# ======================================================` |
| 044 | code |  | EyePACS Exploration & Pretraining (Legacy) | Module 01 (Pretraining / Legacy) | Unexecuted Code | 133 | `# ======================================================` |
| 045 | markdown |  | Attention Maps / ResNet50 CAMs (Legacy) | Module 13 (Legacy CAM) | Markdown Documentation | 1 | `# attention maps` |
| 046 | code |  | Attention Maps / ResNet50 CAMs (Legacy) | Module 13 (Legacy CAM) | Unexecuted Code | 188 | `# Save this as make_mean_gradcams.py and run it (python make_mean_gradcams.py)` |
| 047 | code |  | Attention Maps / ResNet50 CAMs (Legacy) | Module 13 (Legacy CAM) | Unexecuted Code | 187 | `# =====================================================` |
| 048 | code |  | Attention Maps / ResNet50 CAMs (Legacy) | Module 13 (Legacy CAM) | Unexecuted Code | 25 | `import os` |
| 049 | code |  | Attention Maps / ResNet50 CAMs (Legacy) | Module 13 (Legacy CAM) | Unexecuted Code | 24 | `# ======================================================` |
| 050 | code |  | Attention Maps / ResNet50 CAMs (Legacy) | Module 13 (Legacy CAM) | Unexecuted Code | 27 | `# ======================================================` |
| 051 | code |  | Attention Maps / ResNet50 CAMs (Legacy) | Module 13 (Legacy CAM) | Unexecuted Code | 22 | `# ======================================================` |
| 052 | code |  | Attention Maps / ResNet50 CAMs (Legacy) | Module 13 (Legacy CAM) | Unexecuted Code | 7 | `# ======================================================` |
| 053 | code |  | Attention Maps / ResNet50 CAMs (Legacy) | Module 13 (Legacy CAM) | Unexecuted Code | 11 | `# ======================================================` |
| 054 | code |  | Attention Maps / ResNet50 CAMs (Legacy) | Module 13 (Legacy CAM) | Unexecuted Code | 119 | `# ======================================================` |
| 055 | code |  | Attention Maps / ResNet50 CAMs (Legacy) | Module 13 (Legacy CAM) | Unexecuted Code | 42 | `import cv2, numpy as np, glob, os` |
| 056 | code |  | Attention Maps / ResNet50 CAMs (Legacy) | Module 13 (Legacy CAM) | Unexecuted Code | 9 | `from glob import glob` |
| 057 | code |  | Attention Maps / ResNet50 CAMs (Legacy) | Module 13 (Legacy CAM) | Unexecuted Code | 66 | `import os` |
| 058 | code |  | Attention Maps / ResNet50 CAMs (Legacy) | Module 13 (Legacy CAM) | Unexecuted Code | 56 | `import os` |
| 059 | code |  | Attention Maps / ResNet50 CAMs (Legacy) | Module 13 (Legacy CAM) | Unexecuted Code | 22 | `import os` |
| 060 | code |  | Attention Maps / ResNet50 CAMs (Legacy) | Module 13 (Legacy CAM) | Unexecuted Code | 9 | `mean_cams = {}` |
| 061 | code |  | Attention Maps / ResNet50 CAMs (Legacy) | Module 13 (Legacy CAM) | Unexecuted Code | 11 | `import matplotlib.pyplot as plt` |
| 062 | code |  | Attention Maps / ResNet50 CAMs (Legacy) | Module 13 (Legacy CAM) | Unexecuted Code | 65 | `# ======================================================` |
| 063 | code |  | Attention Maps / ResNet50 CAMs (Legacy) | Module 13 (Legacy CAM) | Unexecuted Code | 57 | `# ======================================================` |
| 064 | code |  | Attention Maps / ResNet50 CAMs (Legacy) | Module 13 (Legacy CAM) | Unexecuted Code | 12 | `# ======================================================` |
| 065 | code |  | Attention Maps / ResNet50 CAMs (Legacy) | Module 13 (Legacy CAM) | Unexecuted Code | 11 | `# ======================================================` |
| 066 | code |  | Attention Maps / ResNet50 CAMs (Legacy) | Module 13 (Legacy CAM) | Unexecuted Code | 36 | `# ======================================================` |
| 067 | code |  | Attention Maps / ResNet50 CAMs (Legacy) | Module 13 (Legacy CAM) | Unexecuted Code | 33 | `# ======================================================` |
| 068 | code |  | Attention Maps / ResNet50 CAMs (Legacy) | Module 13 (Legacy CAM) | Unexecuted Code | 52 | `# ======================================================` |
| 069 | code |  | Attention Maps / ResNet50 CAMs (Legacy) | Module 13 (Legacy CAM) | Unexecuted Code | 38 | `# ======================================================` |
| 070 | code |  | Attention Maps / ResNet50 CAMs (Legacy) | Module 13 (Legacy CAM) | Unexecuted Code | 87 | `# ======================================================` |
| 071 | code |  | Attention Maps / ResNet50 CAMs (Legacy) | Module 13 (Legacy CAM) | Unexecuted Code | 0 | `<empty>` |
| 072 | code |  | Attention Maps / ResNet50 CAMs (Legacy) | Module 13 (Legacy CAM) | Unexecuted Code | 107 | `# ======================================================` |
| 073 | code |  | Attention Maps / ResNet50 CAMs (Legacy) | Module 13 (Legacy CAM) | Unexecuted Code | 114 | `# ======================================================` |
| 074 | code |  | Attention Maps / ResNet50 CAMs (Legacy) | Module 13 (Legacy CAM) | Unexecuted Code | 128 | `# ======================================================` |
| 075 | code |  | Attention Maps / ResNet50 CAMs (Legacy) | Module 13 (Legacy CAM) | Unexecuted Code | 46 | `# ======================================================` |
| 076 | markdown |  | Attention Maps / ResNet50 CAMs (Legacy) | Module 13 (Legacy CAM) | Markdown Documentation | 1 | `# Loading the finetuned model` |
| 077 | code |  | Attention Maps / ResNet50 CAMs (Legacy) | Module 13 (Legacy CAM) | Unexecuted Code | 58 | `# ======================================================` |
| 078 | code |  | Attention Maps / ResNet50 CAMs (Legacy) | Module 13 (Legacy CAM) | Unexecuted Code | 3 | `!pip install -q scikit-learn==1.5.2` |
| 079 | code |  | Attention Maps / ResNet50 CAMs (Legacy) | Module 13 (Legacy CAM) | Unexecuted Code | 2 | `!pip uninstall -y scikit-learn` |
| 080 | code |  | Attention Maps / ResNet50 CAMs (Legacy) | Module 13 (Legacy CAM) | Unexecuted Code | 4 | `import sklearn` |
| 081 | code |  | Attention Maps / ResNet50 CAMs (Legacy) | Module 13 (Legacy CAM) | Unexecuted Code | 5 | `for i, row in enumerate(samples.itertuples()):` |
| 082 | code |  | Attention Maps / ResNet50 CAMs (Legacy) | Module 13 (Legacy CAM) | Unexecuted Code | 70 | `import torch, cv2, numpy as np, os` |
| 083 | code |  | Attention Maps / ResNet50 CAMs (Legacy) | Module 13 (Legacy CAM) | Unexecuted Code | 108 | `import torch, cv2, numpy as np, os, random` |
| 084 | markdown |  | Attention Maps / ResNet50 CAMs (Legacy) | Module 13 (Legacy CAM) | Markdown Documentation | 1 | `balanced grad cam run` |
| 085 | code |  | Attention Maps / ResNet50 CAMs (Legacy) | Module 13 (Legacy CAM) | Unexecuted Code | 72 | `# ======================================================` |
| 086 | code |  | Attention Maps / ResNet50 CAMs (Legacy) | Module 13 (Legacy CAM) | Unexecuted Code | 182 | `# ======================================================` |
| 087 | code |  | Attention Maps / ResNet50 CAMs (Legacy) | Module 13 (Legacy CAM) | Unexecuted Code | 35 | `import pandas as pd` |
| 088 | code |  | Attention Maps / ResNet50 CAMs (Legacy) | Module 13 (Legacy CAM) | Unexecuted Code | 74 | `# ======================================================` |
| 089 | code |  | Attention Maps / ResNet50 CAMs (Legacy) | Module 13 (Legacy CAM) | Unexecuted Code | 5 | `import os` |
| 090 | code |  | Attention Maps / ResNet50 CAMs (Legacy) | Module 13 (Legacy CAM) | Unexecuted Code | 146 | `# ======================================================` |
| 091 | code |  | Attention Maps / ResNet50 CAMs (Legacy) | Module 13 (Legacy CAM) | Unexecuted Code | 44 | `# ======================================================` |
| 092 | code |  | Attention Maps / ResNet50 CAMs (Legacy) | Module 13 (Legacy CAM) | Unexecuted Code | 6 | `# Create a new alias folder in /kaggle/working (writable)` |
| 093 | code |  | Attention Maps / ResNet50 CAMs (Legacy) | Module 13 (Legacy CAM) | Unexecuted Code | 27 | `import os, cv2, random` |
| 094 | code |  | Radiomics, Vessel Extraction & CLAHE | Module 01 / Module 06 (Radiomics) | Unexecuted Code | 78 | `# ======================================================` |
| 095 | code |  | Radiomics, Vessel Extraction & CLAHE | Module 01 / Module 06 (Radiomics) | Unexecuted Code | 27 | `import os, cv2, random` |
| 096 | code |  | Radiomics, Vessel Extraction & CLAHE | Module 01 / Module 06 (Radiomics) | Unexecuted Code | 25 | `# ======================================================` |
| 097 | code |  | Radiomics, Vessel Extraction & CLAHE | Module 01 / Module 06 (Radiomics) | Unexecuted Code | 46 | `# ======================================================` |
| 098 | code |  | Radiomics, Vessel Extraction & CLAHE | Module 01 / Module 06 (Radiomics) | Unexecuted Code | 30 | `import pandas as pd` |
| 099 | code |  | Radiomics, Vessel Extraction & CLAHE | Module 01 / Module 06 (Radiomics) | Unexecuted Code | 1 | `!pip install -q pyradiomics scikit-image opencv-python-headless SimpleITK tqdm` |
| 100 | code |  | Radiomics, Vessel Extraction & CLAHE | Module 01 / Module 06 (Radiomics) | Unexecuted Code | 111 | `# === radiomics_extract_demo.py ===` |
| 101 | code |  | Radiomics, Vessel Extraction & CLAHE | Module 01 / Module 06 (Radiomics) | Unexecuted Code | 9 | `import pandas as pd` |
| 102 | code |  | Radiomics, Vessel Extraction & CLAHE | Module 01 / Module 06 (Radiomics) | Unexecuted Code | 111 | `!pip install -q pyradiomics scikit-image opencv-python-headless scipy tqdm` |
| 103 | code |  | Radiomics, Vessel Extraction & CLAHE | Module 01 / Module 06 (Radiomics) | Unexecuted Code | 26 | `# --------------------------------------------------------------` |
| 104 | code |  | Radiomics, Vessel Extraction & CLAHE | Module 01 / Module 06 (Radiomics) | Unexecuted Code | 1 | `!find /kaggle/input/ddr-final -type f -name "*.jpg" \| head -n 20` |
| 105 | code |  | Radiomics, Vessel Extraction & CLAHE | Module 01 / Module 06 (Radiomics) | Unexecuted Code | 4 | `IMG_DIR = "/kaggle/input/ddr-final/final"` |
| 106 | code |  | Radiomics, Vessel Extraction & CLAHE | Module 01 / Module 06 (Radiomics) | Unexecuted Code | 33 | `import os, cv2, pandas as pd` |
| 107 | code |  | Radiomics, Vessel Extraction & CLAHE | Module 01 / Module 06 (Radiomics) | Unexecuted Code | 75 | `# ==========================================================` |
| 108 | code |  | Radiomics, Vessel Extraction & CLAHE | Module 01 / Module 06 (Radiomics) | Unexecuted Code | 66 | `# ==========================================================` |
| 109 | code |  | Radiomics, Vessel Extraction & CLAHE | Module 01 / Module 06 (Radiomics) | Unexecuted Code | 53 | `# ==========================================================` |
| 110 | code |  | Radiomics, Vessel Extraction & CLAHE | Module 01 / Module 06 (Radiomics) | Unexecuted Code | 72 | `# ==========================================================` |
| 111 | code |  | Radiomics, Vessel Extraction & CLAHE | Module 01 / Module 06 (Radiomics) | Unexecuted Code | 72 | `import os, json, joblib` |
| 112 | markdown |  | Radiomics, Vessel Extraction & CLAHE | Module 01 / Module 06 (Radiomics) | Markdown Documentation | 1 | `# now for train and test` |
| 113 | code |  | Radiomics, Vessel Extraction & CLAHE | Module 01 / Module 06 (Radiomics) | Unexecuted Code | 76 | `import os, json, joblib` |
| 114 | code |  | Radiomics, Vessel Extraction & CLAHE | Module 01 / Module 06 (Radiomics) | Unexecuted Code | 17 | `import seaborn as sns, matplotlib.pyplot as plt` |
| 115 | code |  | Radiomics, Vessel Extraction & CLAHE | Module 01 / Module 06 (Radiomics) | Unexecuted Code | 12 | `corr_pairs = (` |
| 116 | code |  | Radiomics, Vessel Extraction & CLAHE | Module 01 / Module 06 (Radiomics) | Unexecuted Code | 26 | `import seaborn as sns, matplotlib.pyplot as plt` |
| 117 | code |  | Radiomics, Vessel Extraction & CLAHE | Module 01 / Module 06 (Radiomics) | Unexecuted Code | 66 | `import os` |
| 118 | code |  | Radiomics, Vessel Extraction & CLAHE | Module 01 / Module 06 (Radiomics) | Unexecuted Code | 22 | `import seaborn as sns, matplotlib.pyplot as plt` |
| 119 | code |  | Radiomics, Vessel Extraction & CLAHE | Module 01 / Module 06 (Radiomics) | Unexecuted Code | 30 | `import shap, joblib, json, matplotlib.pyplot as plt` |
| 120 | code |  | Radiomics, Vessel Extraction & CLAHE | Module 01 / Module 06 (Radiomics) | Unexecuted Code | 137 | `# # ==========================================================` |
| 121 | code |  | Radiomics, Vessel Extraction & CLAHE | Module 01 / Module 06 (Radiomics) | Unexecuted Code | 1 | `# !pip install -U scikit-learn==1.5.2 imbalanced-learn==0.12.3 lightgbm==4.3.0 xgboost==2.1.0 torch ` |
| 122 | code |  | Radiomics, Vessel Extraction & CLAHE | Module 01 / Module 06 (Radiomics) | Unexecuted Code | 1 | `!pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch` |
| 123 | code |  | Radiomics, Vessel Extraction & CLAHE | Module 01 / Module 06 (Radiomics) | Unexecuted Code | 1 | `!pip install -U scikit-learn==1.5.2 imbalanced-learn==0.12.0` |
| 124 | code |  | Radiomics, Vessel Extraction & CLAHE | Module 01 / Module 06 (Radiomics) | Unexecuted Code | 6 | `!pip install --force-reinstall --no-cache-dir \` |
| 125 | code |  | Radiomics, Vessel Extraction & CLAHE | Module 01 / Module 06 (Radiomics) | Unexecuted Code | 1 | `!pip install --force-reinstall --no-cache-dir "numpy<2.0" "scipy<1.12"` |
| 126 | code |  | Radiomics, Vessel Extraction & CLAHE | Module 01 / Module 06 (Radiomics) | Unexecuted Code | 5 | `import numpy as np, torch` |
| 127 | code |  | Radiomics, Vessel Extraction & CLAHE | Module 01 / Module 06 (Radiomics) | Unexecuted Code | 133 | `# import os, numpy as np, pandas as pd` |
| 128 | code |  | Radiomics, Vessel Extraction & CLAHE | Module 01 / Module 06 (Radiomics) | Unexecuted Code | 159 | `import os, numpy as np, pandas as pd, json` |
| 129 | code |  | Radiomics, Vessel Extraction & CLAHE | Module 01 / Module 06 (Radiomics) | Unexecuted Code | 0 | `<empty>` |
| 130 | code |  | Radiomics, Vessel Extraction & CLAHE | Module 01 / Module 06 (Radiomics) | Unexecuted Code | 133 | `import os, numpy as np, pandas as pd, json` |
| 131 | code |  | Radiomics, Vessel Extraction & CLAHE | Module 01 / Module 06 (Radiomics) | Unexecuted Code | 68 | `import numpy as np` |
| 132 | code |  | Radiomics, Vessel Extraction & CLAHE | Module 01 / Module 06 (Radiomics) | Unexecuted Code | 59 | `import os` |
| 133 | code |  | Radiomics, Vessel Extraction & CLAHE | Module 01 / Module 06 (Radiomics) | Unexecuted Code | 59 | `import os` |
| 134 | markdown |  | Feature Fusion MLP & SMOTE Experiments | Module 08 (SMOTE) / Module 10 (MLP Fusion) | Markdown Documentation | 1 | `# fusion model` |
| 135 | code |  | Feature Fusion MLP & SMOTE Experiments | Module 08 (SMOTE) / Module 10 (MLP Fusion) | Unexecuted Code | 38 | `import os` |
| 136 | code |  | Feature Fusion MLP & SMOTE Experiments | Module 08 (SMOTE) / Module 10 (MLP Fusion) | Unexecuted Code | 81 | `# ======================================================` |
| 137 | code |  | Feature Fusion MLP & SMOTE Experiments | Module 08 (SMOTE) / Module 10 (MLP Fusion) | Unexecuted Code | 4 | `import pandas as pd` |
| 138 | code |  | Feature Fusion MLP & SMOTE Experiments | Module 08 (SMOTE) / Module 10 (MLP Fusion) | Unexecuted Code | 17 | `import pandas as pd` |
| 139 | code |  | Feature Fusion MLP & SMOTE Experiments | Module 08 (SMOTE) / Module 10 (MLP Fusion) | Unexecuted Code | 5 | `# Check a few sample IDs from each file` |
| 140 | code |  | Feature Fusion MLP & SMOTE Experiments | Module 08 (SMOTE) / Module 10 (MLP Fusion) | Unexecuted Code | 5 | `# Check how cleaned IDs look after normalization` |
| 141 | code |  | Feature Fusion MLP & SMOTE Experiments | Module 08 (SMOTE) / Module 10 (MLP Fusion) | Unexecuted Code | 36 | `import pandas as pd` |
| 142 | code |  | Feature Fusion MLP & SMOTE Experiments | Module 08 (SMOTE) / Module 10 (MLP Fusion) | Unexecuted Code | 9 | `# Merge again cleanly and save` |
| 143 | code |  | Feature Fusion MLP & SMOTE Experiments | Module 08 (SMOTE) / Module 10 (MLP Fusion) | Unexecuted Code | 21 | `import pandas as pd, numpy as np` |
| 144 | code |  | Feature Fusion MLP & SMOTE Experiments | Module 08 (SMOTE) / Module 10 (MLP Fusion) | Unexecuted Code | 130 | `# ==========================================================` |
| 145 | markdown |  | Feature Fusion MLP & SMOTE Experiments | Module 08 (SMOTE) / Module 10 (MLP Fusion) | Markdown Documentation | 1 | `# cross check` |
| 146 | code |  | Feature Fusion MLP & SMOTE Experiments | Module 08 (SMOTE) / Module 10 (MLP Fusion) | Unexecuted Code | 138 | `# ==========================================================` |
| 147 | code |  | Feature Fusion MLP & SMOTE Experiments | Module 08 (SMOTE) / Module 10 (MLP Fusion) | Unexecuted Code | 19 | `import json, pandas as pd, joblib, numpy as np` |
| 148 | markdown |  | Feature Fusion MLP & SMOTE Experiments | Module 08 (SMOTE) / Module 10 (MLP Fusion) | Markdown Documentation | 1 | `# 5 fold cv` |
| 149 | code |  | Feature Fusion MLP & SMOTE Experiments | Module 08 (SMOTE) / Module 10 (MLP Fusion) | Unexecuted Code | 97 | `# ==========================================================` |
| 150 | code |  | Feature Fusion MLP & SMOTE Experiments | Module 08 (SMOTE) / Module 10 (MLP Fusion) | Unexecuted Code | 2 | `print(df[label_col].unique())` |
| 151 | code |  | Feature Fusion MLP & SMOTE Experiments | Module 08 (SMOTE) / Module 10 (MLP Fusion) | Unexecuted Code | 129 | `# ==========================================================` |
| 152 | code |  | Feature Fusion MLP & SMOTE Experiments | Module 08 (SMOTE) / Module 10 (MLP Fusion) | Unexecuted Code | 111 | `# ==========================================================` |
| 153 | code |  | Feature Fusion MLP & SMOTE Experiments | Module 08 (SMOTE) / Module 10 (MLP Fusion) | Unexecuted Code | 100 | `# ==========================================================` |
| 154 | code |  | Feature Fusion MLP & SMOTE Experiments | Module 08 (SMOTE) / Module 10 (MLP Fusion) | Unexecuted Code | 113 | `# ==========================================================` |
| 155 | code |  | Feature Fusion MLP & SMOTE Experiments | Module 08 (SMOTE) / Module 10 (MLP Fusion) | Unexecuted Code | 103 | `# # ==========================================================` |
| 156 | code |  | Feature Fusion MLP & SMOTE Experiments | Module 08 (SMOTE) / Module 10 (MLP Fusion) | Unexecuted Code | 131 | `# ==========================================================` |
| 157 | code |  | Feature Fusion MLP & SMOTE Experiments | Module 08 (SMOTE) / Module 10 (MLP Fusion) | Unexecuted Code | 4 | `from collections import Counter` |
| 158 | code |  | Feature Fusion MLP & SMOTE Experiments | Module 08 (SMOTE) / Module 10 (MLP Fusion) | Unexecuted Code | 34 | `import pandas as pd, re, os` |
| 159 | code |  | Feature Fusion MLP & SMOTE Experiments | Module 08 (SMOTE) / Module 10 (MLP Fusion) | Unexecuted Code | 37 | `# ==========================================================` |
| 160 | code |  | Feature Fusion MLP & SMOTE Experiments | Module 08 (SMOTE) / Module 10 (MLP Fusion) | Unexecuted Code | 28 | `# ==========================================================` |
| 161 | code |  | Feature Fusion MLP & SMOTE Experiments | Module 08 (SMOTE) / Module 10 (MLP Fusion) | Unexecuted Code | 42 | `# ==========================================================` |
| 162 | code |  | Feature Fusion MLP & SMOTE Experiments | Module 08 (SMOTE) / Module 10 (MLP Fusion) | Unexecuted Code | 112 | `# ==========================================================` |
| 163 | code |  | Feature Fusion MLP & SMOTE Experiments | Module 08 (SMOTE) / Module 10 (MLP Fusion) | Unexecuted Code | 70 | `# ==========================================================` |
| 164 | markdown |  | Multi-Backbone Exploration & Embedding Extraction | Module 02/03/04 (Embeddings) | Markdown Documentation | 1 | `# ensemble model` |
| 165 | code |  | Multi-Backbone Exploration & Embedding Extraction | Module 02/03/04 (Embeddings) | Unexecuted Code | 4 | `import os` |
| 166 | code |  | Multi-Backbone Exploration & Embedding Extraction | Module 02/03/04 (Embeddings) | Unexecuted Code | 15 | `import os` |
| 167 | code |  | Multi-Backbone Exploration & Embedding Extraction | Module 02/03/04 (Embeddings) | Unexecuted Code | 41 | `import os` |
| 168 | code |  | Multi-Backbone Exploration & Embedding Extraction | Module 02/03/04 (Embeddings) | Unexecuted Code | 21 | `import pandas as pd` |
| 169 | code |  | Multi-Backbone Exploration & Embedding Extraction | Module 02/03/04 (Embeddings) | Unexecuted Code | 15 | `clean_manifest = pd.read_csv("/kaggle/working/manifest_eyepacs_cleaned.csv")` |
| 170 | code |  | Multi-Backbone Exploration & Embedding Extraction | Module 02/03/04 (Embeddings) | Unexecuted Code | 23 | `import pandas as pd` |
| 171 | code |  | Multi-Backbone Exploration & Embedding Extraction | Module 02/03/04 (Embeddings) | Unexecuted Code | 18 | `# =============================================================` |
| 172 | code |  | Multi-Backbone Exploration & Embedding Extraction | Module 02/03/04 (Embeddings) | Unexecuted Code | 29 | `# =============================================================` |
| 173 | markdown |  | Multi-Backbone Exploration & Embedding Extraction | Module 02/03/04 (Embeddings) | Markdown Documentation | 1 | `# Ensemble model setup` |
| 174 | code |  | Multi-Backbone Exploration & Embedding Extraction | Module 02/03/04 (Embeddings) | Unexecuted Code | 64 | `# =============================================================` |
| 175 | code |  | Multi-Backbone Exploration & Embedding Extraction | Module 02/03/04 (Embeddings) | Unexecuted Code | 123 | `# =============================================================` |
| 176 | code |  | Multi-Backbone Exploration & Embedding Extraction | Module 02/03/04 (Embeddings) | Unexecuted Code | 18 | `# =============================================================` |
| 177 | code |  | Multi-Backbone Exploration & Embedding Extraction | Module 02/03/04 (Embeddings) | Unexecuted Code | 48 | `# =============================================================` |
| 178 | code |  | Multi-Backbone Exploration & Embedding Extraction | Module 02/03/04 (Embeddings) | Unexecuted Code | 21 | `# =============================================================` |
| 179 | code |  | Multi-Backbone Exploration & Embedding Extraction | Module 02/03/04 (Embeddings) | Unexecuted Code | 45 | `# =============================================================` |
| 180 | code |  | Multi-Backbone Exploration & Embedding Extraction | Module 02/03/04 (Embeddings) | Unexecuted Code | 146 | `# =============================================================` |
| 181 | code |  | Multi-Backbone Exploration & Embedding Extraction | Module 02/03/04 (Embeddings) | Unexecuted Code | 77 | `import torch` |
| 182 | code |  | Multi-Backbone Exploration & Embedding Extraction | Module 02/03/04 (Embeddings) | Unexecuted Code | 134 | `# === tidy_week2_files.py ===` |
| 183 | code |  | Multi-Backbone Exploration & Embedding Extraction | Module 02/03/04 (Embeddings) | Unexecuted Code | 15 | `import numpy as np` |
| 184 | code |  | Multi-Backbone Exploration & Embedding Extraction | Module 02/03/04 (Embeddings) | Unexecuted Code | 43 | `import os` |
| 185 | code |  | Multi-Backbone Exploration & Embedding Extraction | Module 02/03/04 (Embeddings) | Unexecuted Code | 3 | `df = pd.read_csv("/kaggle/working/manifest_fixed.csv")` |
| 186 | code |  | Multi-Backbone Exploration & Embedding Extraction | Module 02/03/04 (Embeddings) | Unexecuted Code | 3 | `import pandas as pd` |
| 187 | code |  | Multi-Backbone Exploration & Embedding Extraction | Module 02/03/04 (Embeddings) | Unexecuted Code | 2 | `missing = df[df["filepath"].isna()]` |
| 188 | code |  | Multi-Backbone Exploration & Embedding Extraction | Module 02/03/04 (Embeddings) | Unexecuted Code | 3 | `df2 = df[df["image_id"] != "eyepacs_06934.png"].reset_index(drop=True)` |
| 189 | code |  | Multi-Backbone Exploration & Embedding Extraction | Module 02/03/04 (Embeddings) | Unexecuted Code | 70 | `import torch` |
| 190 | code |  | Multi-Backbone Exploration & Embedding Extraction | Module 02/03/04 (Embeddings) | Unexecuted Code | 41 | `import os` |
| 191 | code |  | Multi-Backbone Exploration & Embedding Extraction | Module 02/03/04 (Embeddings) | Unexecuted Code | 16 | `import numpy as np` |
| 192 | code |  | Multi-Backbone Exploration & Embedding Extraction | Module 02/03/04 (Embeddings) | Unexecuted Code | 16 | `import numpy as np` |
| 193 | code |  | Multi-Backbone Exploration & Embedding Extraction | Module 02/03/04 (Embeddings) | Unexecuted Code | 11 | `import os` |
| 194 | code |  | Multi-Backbone Exploration & Embedding Extraction | Module 02/03/04 (Embeddings) | Unexecuted Code | 13 | `import os` |
| 195 | code |  | Multi-Backbone Exploration & Embedding Extraction | Module 02/03/04 (Embeddings) | Unexecuted Code | 39 | `import os` |
| 196 | code |  | Multi-Backbone Exploration & Embedding Extraction | Module 02/03/04 (Embeddings) | Unexecuted Code | 3 | `import shutil` |
| 197 | code |  | Multi-Backbone Exploration & Embedding Extraction | Module 02/03/04 (Embeddings) | Unexecuted Code | 3 | `import os` |
| 198 | code |  | Multi-Backbone Exploration & Embedding Extraction | Module 02/03/04 (Embeddings) | Unexecuted Code | 14 | `import os` |
| 199 | code |  | Multi-Backbone Exploration & Embedding Extraction | Module 02/03/04 (Embeddings) | Unexecuted Code | 19 | `import os` |
| 200 | code |  | Multi-Backbone Exploration & Embedding Extraction | Module 02/03/04 (Embeddings) | Unexecuted Code | 8 | `import pandas as pd` |
| 201 | code |  | Multi-Backbone Exploration & Embedding Extraction | Module 02/03/04 (Embeddings) | Unexecuted Code | 1 | `df.head(5)` |
| 202 | code |  | Multi-Backbone Exploration & Embedding Extraction | Module 02/03/04 (Embeddings) | Unexecuted Code | 26 | `import os` |
| 203 | code |  | Multi-Backbone Exploration & Embedding Extraction | Module 02/03/04 (Embeddings) | Unexecuted Code | 25 | `import pandas as pd` |
| 204 | markdown |  | Multi-Backbone Exploration & Embedding Extraction | Module 02/03/04 (Embeddings) | Markdown Documentation | 1 | `# multi backbone fusion deep learning model` |
| 205 | code |  | Multi-Backbone Exploration & Embedding Extraction | Module 02/03/04 (Embeddings) | Unexecuted Code | 19 | `# =========================` |
| 206 | markdown |  | Dataset Realignment Notice (EyePACS -> DDR) | Module 01 (Dataset Validation) | Markdown Documentation | 1 | `### misaligned as the models(eyepac) and radiomics(DDR)` |
| 207 | code |  | Dataset Realignment Notice (EyePACS -> DDR) | Module 01 (Dataset Validation) | Unexecuted Code | 22 | `# =========================` |
| 208 | code |  | Dataset Realignment Notice (EyePACS -> DDR) | Module 01 (Dataset Validation) | Unexecuted Code | 4 | `print("EffB3:", eff_df["image_id"].head())` |
| 209 | code |  | Dataset Realignment Notice (EyePACS -> DDR) | Module 01 (Dataset Validation) | Unexecuted Code | 11 | `def clean_id(x):` |
| 210 | code |  | Dataset Realignment Notice (EyePACS -> DDR) | Module 01 (Dataset Validation) | Unexecuted Code | 5 | `merged = (eff_df` |
| 211 | markdown |  | Dataset Realignment Notice (EyePACS -> DDR) | Module 01 (Dataset Validation) | Markdown Documentation | 1 | `# finetuning using muti fusion of deep learning followed by mlp and concat with radiomics later#` |
| 212 | code |  | Dataset Realignment Notice (EyePACS -> DDR) | Module 01 (Dataset Validation) | Unexecuted Code | 38 | `# BLOCK A - utilities: QWK, seed, focal loss, small helpers` |
| 213 | code |  | Dataset Realignment Notice (EyePACS -> DDR) | Module 01 (Dataset Validation) | Unexecuted Code | 17 | `import pandas as pd` |
| 214 | code |  | Dataset Realignment Notice (EyePACS -> DDR) | Module 01 (Dataset Validation) | Unexecuted Code | 1 | `data = pd.read_csv("/kaggle/working/manifest_ddr_aligned.csv")` |
| 215 | code |  | Dataset Realignment Notice (EyePACS -> DDR) | Module 01 (Dataset Validation) | Unexecuted Code | 1 | `data.head(5)` |
| 216 | raw |  | Dataset Realignment Notice (EyePACS -> DDR) | Module 01 (Dataset Validation) | Unexecuted Code | 0 | `<empty>` |
| 217 | code |  | Dataset Realignment Notice (EyePACS -> DDR) | Module 01 (Dataset Validation) | Unexecuted Code | 1 | `print(len(data))` |
| 218 | code |  | Dataset Realignment Notice (EyePACS -> DDR) | Module 01 (Dataset Validation) | Unexecuted Code | 0 | `<empty>` |
| 219 | code |  | Dataset Realignment Notice (EyePACS -> DDR) | Module 01 (Dataset Validation) | Unexecuted Code | 4 | `import os` |
| 220 | code |  | Baseline: EfficientNet-B3 Fine-Tuning on DDR | Module 04 (EfficientNet-B3) | Unexecuted Code | 162 | `# BLOCK B — DDR Dataset + Transforms (Robust, ready-to-run)` |
| 221 | code |  | Baseline: EfficientNet-B3 Fine-Tuning on DDR | Module 04 (EfficientNet-B3) | Unexecuted Code | 1 | `print(manifest_df["numeric_label"].value_counts())` |
| 222 | markdown |  | Baseline: EfficientNet-B3 Fine-Tuning on DDR | Module 04 (EfficientNet-B3) | Markdown Documentation | 1 | `#### effb3` |
| 223 | code |  | Baseline: EfficientNet-B3 Fine-Tuning on DDR | Module 04 (EfficientNet-B3) | Unexecuted Code | 30 | `# BLOCK 0 - CONFIG` |
| 224 | code |  | Baseline: EfficientNet-B3 Fine-Tuning on DDR | Module 04 (EfficientNet-B3) | Unexecuted Code | 55 | `# BLOCK 1 - DATASET & DATALOADERS` |
| 225 | code |  | Baseline: EfficientNet-B3 Fine-Tuning on DDR | Module 04 (EfficientNet-B3) | Unexecuted Code | 28 | `# BLOCK 2 - MODEL INIT and weight loading` |
| 226 | code |  | Baseline: EfficientNet-B3 Fine-Tuning on DDR | Module 04 (EfficientNet-B3) | Unexecuted Code | 33 | `# BLOCK 3 - LOSS, OPTIMIZER, SCHEDULER & METRIC (QWK)` |
| 227 | code |  | Baseline: EfficientNet-B3 Fine-Tuning on DDR | Module 04 (EfficientNet-B3) | Unexecuted Code | 28 | `# BLOCK 4 - VALIDATION FUNCTION` |
| 228 | code |  | Baseline: EfficientNet-B3 Fine-Tuning on DDR | Module 04 (EfficientNet-B3) | Unexecuted Code | 0 | `<empty>` |
| 229 | code |  | Baseline: EfficientNet-B3 Fine-Tuning on DDR | Module 04 (EfficientNet-B3) | Unexecuted Code | 104 | `# ================================================================` |
| 230 | code |  | Baseline: ConvNeXt-Small Fine-Tuning on DDR | Module 02 (ConvNeXt-Small) | Unexecuted Code | 46 | `# ============================================================` |
| 231 | code |  | Baseline: ConvNeXt-Small Fine-Tuning on DDR | Module 02 (ConvNeXt-Small) | Unexecuted Code | 95 | `# ================================================================` |
| 232 | code |  | Baseline: ConvNeXt-Small Fine-Tuning on DDR | Module 02 (ConvNeXt-Small) | Unexecuted Code | 43 | `# ============================================================` |
| 233 | code |  | Baseline: ConvNeXt-Small Fine-Tuning on DDR | Module 02 (ConvNeXt-Small) | Unexecuted Code | 53 | `# ================================================================` |
| 234 | code |  | Baseline: ConvNeXt-Small Fine-Tuning on DDR | Module 02 (ConvNeXt-Small) | Unexecuted Code | 29 | `# BLOCK 4 - VALIDATION FUNCTION` |
| 235 | code |  | Baseline: ConvNeXt-Small Fine-Tuning on DDR | Module 02 (ConvNeXt-Small) | Unexecuted Code | 109 | `# ================================================================` |
| 236 | markdown |  | Baseline: Swin-Tiny Fine-Tuning on DDR | Module 03 (Swin-Tiny) | Markdown Documentation | 1 | `### SWIN TINY` |
| 237 | code |  | Baseline: Swin-Tiny Fine-Tuning on DDR | Module 03 (Swin-Tiny) | Unexecuted Code | 46 | `# ============================================================` |
| 238 | code |  | Baseline: Swin-Tiny Fine-Tuning on DDR | Module 03 (Swin-Tiny) | Unexecuted Code | 100 | `# ================================================================` |
| 239 | code |  | Baseline: Swin-Tiny Fine-Tuning on DDR | Module 03 (Swin-Tiny) | Unexecuted Code | 42 | `# ============================================================` |
| 240 | code |  | Baseline: Swin-Tiny Fine-Tuning on DDR | Module 03 (Swin-Tiny) | Unexecuted Code | 57 | `# ================================================================` |
| 241 | code |  | Baseline: Swin-Tiny Fine-Tuning on DDR | Module 03 (Swin-Tiny) | Unexecuted Code | 35 | `# ================================================================` |
| 242 | code |  | Baseline: Swin-Tiny Fine-Tuning on DDR | Module 03 (Swin-Tiny) | Unexecuted Code | 108 | `from tqdm import tqdm` |
| 243 | code |  | Baseline: Swin-Tiny Fine-Tuning on DDR | Module 03 (Swin-Tiny) | Unexecuted Code | 6 | `# Load manifest for embedding extraction` |
| 244 | code |  | Multi-Backbone + Global Radiomics + CORAL | Module 09 (CORAL) / Module 10 (Fusion) | Unexecuted Code | 74 | `import os` |
| 245 | code |  | Multi-Backbone + Global Radiomics + CORAL | Module 09 (CORAL) / Module 10 (Fusion) | Unexecuted Code | 7 | `shape_eff = extract_embeddings(` |
| 246 | code |  | Multi-Backbone + Global Radiomics + CORAL | Module 09 (CORAL) / Module 10 (Fusion) | Unexecuted Code | 7 | `shape_conv = extract_embeddings(` |
| 247 | code |  | Multi-Backbone + Global Radiomics + CORAL | Module 09 (CORAL) / Module 10 (Fusion) | Unexecuted Code | 96 | `import torch` |
| 248 | markdown |  | Multi-Backbone + Global Radiomics + CORAL | Module 09 (CORAL) / Module 10 (Fusion) | Markdown Documentation | 0 | `<empty>` |
| 249 | markdown |  | Multi-Backbone + Global Radiomics + CORAL | Module 09 (CORAL) / Module 10 (Fusion) | Markdown Documentation | 13 | `## Fusion of` |
| 250 | code |  | Multi-Backbone + Global Radiomics + CORAL | Module 09 (CORAL) / Module 10 (Fusion) | Unexecuted Code | 1 | `rad_df.head()` |
| 251 | code |  | Multi-Backbone + Global Radiomics + CORAL | Module 09 (CORAL) / Module 10 (Fusion) | Unexecuted Code | 51 | `import os` |
| 252 | code |  | Multi-Backbone + Global Radiomics + CORAL | Module 09 (CORAL) / Module 10 (Fusion) | Unexecuted Code | 24 | `from sklearn.model_selection import train_test_split` |
| 253 | code |  | Multi-Backbone + Global Radiomics + CORAL | Module 09 (CORAL) / Module 10 (Fusion) | Unexecuted Code | 4 | `print("\nClass distribution:")` |
| 254 | code |  | Multi-Backbone + Global Radiomics + CORAL | Module 09 (CORAL) / Module 10 (Fusion) | Unexecuted Code | 73 | `import numpy as np` |
| 255 | code |  | Multi-Backbone + Global Radiomics + CORAL | Module 09 (CORAL) / Module 10 (Fusion) | Unexecuted Code | 7 | `train_ds = FusionDataset(train_df, eff_emb, conv_emb, swin_emb, rad_df)` |
| 256 | code |  | Multi-Backbone + Global Radiomics + CORAL | Module 09 (CORAL) / Module 10 (Fusion) | Unexecuted Code | 94 | `import torch` |
| 257 | code |  | Multi-Backbone + Global Radiomics + CORAL | Module 09 (CORAL) / Module 10 (Fusion) | Unexecuted Code | 30 | `# ==============================` |
| 258 | code |  | Multi-Backbone + Global Radiomics + CORAL | Module 09 (CORAL) / Module 10 (Fusion) | Unexecuted Code | 164 | `import os` |
| 259 | code |  | Multi-Backbone + Global Radiomics + CORAL | Module 09 (CORAL) / Module 10 (Fusion) | Unexecuted Code | 3 | `val_missing = (val_df["image_id"].isin(rad_df["image_id"]) == False).mean()` |
| 260 | markdown |  | Grad-CAM++ & Lesion Patch Extraction | Module 06 (Local Patches) / Module 13 (GradCAM) | Markdown Documentation | 1 | `# grad cam group 4` |
| 261 | code |  | Grad-CAM++ & Lesion Patch Extraction | Module 06 (Local Patches) / Module 13 (GradCAM) | Unexecuted Code | 3 | `!pip install -q scikit-learn==1.5.2` |
| 262 | code |  | Grad-CAM++ & Lesion Patch Extraction | Module 06 (Local Patches) / Module 13 (GradCAM) | Unexecuted Code | 4 | `import numpy, sklearn, pytorch_grad_cam` |
| 263 | code |  | Grad-CAM++ & Lesion Patch Extraction | Module 06 (Local Patches) / Module 13 (GradCAM) | Unexecuted Code | 0 | `<empty>` |
| 264 | code |  | Grad-CAM++ & Lesion Patch Extraction | Module 06 (Local Patches) / Module 13 (GradCAM) | Unexecuted Code | 5 | `import os` |
| 265 | code |  | Grad-CAM++ & Lesion Patch Extraction | Module 06 (Local Patches) / Module 13 (GradCAM) | Unexecuted Code | 8 | `# ---------------- CONFIG ----------------` |
| 266 | code |  | Grad-CAM++ & Lesion Patch Extraction | Module 06 (Local Patches) / Module 13 (GradCAM) | Unexecuted Code | 4 | `import torch` |
| 267 | code |  | Grad-CAM++ & Lesion Patch Extraction | Module 06 (Local Patches) / Module 13 (GradCAM) | Unexecuted Code | 8 | `# import os` |
| 268 | code |  | Grad-CAM++ & Lesion Patch Extraction | Module 06 (Local Patches) / Module 13 (GradCAM) | Unexecuted Code | 1 | `# rm -rf /kaggle/working/gradcam_ddr_fixed` |
| 269 | code |  | Grad-CAM++ & Lesion Patch Extraction | Module 06 (Local Patches) / Module 13 (GradCAM) | Unexecuted Code | 7 | `import os` |
| 270 | code |  | Grad-CAM++ & Lesion Patch Extraction | Module 06 (Local Patches) / Module 13 (GradCAM) | Unexecuted Code | 113 | `# ================== FULL ENSEMBLE GRAD-CAM++ FOR DDR ==================` |
| 271 | code |  | Grad-CAM++ & Lesion Patch Extraction | Module 06 (Local Patches) / Module 13 (GradCAM) | Unexecuted Code | 18 | `import os` |
| 272 | code |  | Grad-CAM++ & Lesion Patch Extraction | Module 06 (Local Patches) / Module 13 (GradCAM) | Unexecuted Code | 25 | `import os` |
| 273 | code |  | Grad-CAM++ & Lesion Patch Extraction | Module 06 (Local Patches) / Module 13 (GradCAM) | Unexecuted Code | 139 | `# # ================== ENSEMBLE GRAD-CAM++ FOR DDR ==================` |
| 274 | code |  | Grad-CAM++ & Lesion Patch Extraction | Module 06 (Local Patches) / Module 13 (GradCAM) | Unexecuted Code | 8 | `# import shutil` |
| 275 | code |  | Grad-CAM++ & Lesion Patch Extraction | Module 06 (Local Patches) / Module 13 (GradCAM) | Unexecuted Code | 0 | `<empty>` |
| 276 | code |  | Grad-CAM++ & Lesion Patch Extraction | Module 06 (Local Patches) / Module 13 (GradCAM) | Unexecuted Code | 0 | `<empty>` |
| 277 | code |  | Grad-CAM++ & Lesion Patch Extraction | Module 06 (Local Patches) / Module 13 (GradCAM) | Unexecuted Code | 19 | `import os` |
| 278 | code |  | Grad-CAM++ & Lesion Patch Extraction | Module 06 (Local Patches) / Module 13 (GradCAM) | Unexecuted Code | 0 | `<empty>` |
| 279 | code |  | Grad-CAM++ & Lesion Patch Extraction | Module 06 (Local Patches) / Module 13 (GradCAM) | Unexecuted Code | 0 | `<empty>` |
| 280 | code |  | Grad-CAM++ & Lesion Patch Extraction | Module 06 (Local Patches) / Module 13 (GradCAM) | Unexecuted Code | 5 | `# import torch` |
| 281 | code |  | Grad-CAM++ & Lesion Patch Extraction | Module 06 (Local Patches) / Module 13 (GradCAM) | Unexecuted Code | 11 | `# import os` |
| 282 | code |  | Grad-CAM++ & Lesion Patch Extraction | Module 06 (Local Patches) / Module 13 (GradCAM) | Unexecuted Code | 32 | `# import os` |
| 283 | code |  | Grad-CAM++ & Lesion Patch Extraction | Module 06 (Local Patches) / Module 13 (GradCAM) | Unexecuted Code | 4 | `BASE_DIR = "/kaggle/working/gradcam_ensemble"   # change if needed` |
| 284 | code |  | Grad-CAM++ & Lesion Patch Extraction | Module 06 (Local Patches) / Module 13 (GradCAM) | Unexecuted Code | 13 | `cls = 0` |
| 285 | code |  | Grad-CAM++ & Lesion Patch Extraction | Module 06 (Local Patches) / Module 13 (GradCAM) | Unexecuted Code | 13 | `cls = 1` |
| 286 | code |  | Grad-CAM++ & Lesion Patch Extraction | Module 06 (Local Patches) / Module 13 (GradCAM) | Unexecuted Code | 13 | `cls = 2` |
| 287 | code |  | Grad-CAM++ & Lesion Patch Extraction | Module 06 (Local Patches) / Module 13 (GradCAM) | Unexecuted Code | 13 | `cls = 3` |
| 288 | code |  | Grad-CAM++ & Lesion Patch Extraction | Module 06 (Local Patches) / Module 13 (GradCAM) | Unexecuted Code | 13 | `cls = 4` |
| 289 | code |  | Grad-CAM++ & Lesion Patch Extraction | Module 06 (Local Patches) / Module 13 (GradCAM) | Unexecuted Code | 4 | `import os` |
| 290 | markdown |  | Grad-CAM++ & Lesion Patch Extraction | Module 06 (Local Patches) / Module 13 (GradCAM) | Markdown Documentation | 1 | `# patch extraction` |
| 291 | code |  | Grad-CAM++ & Lesion Patch Extraction | Module 06 (Local Patches) / Module 13 (GradCAM) | Unexecuted Code | 7 | `# import shutil, os` |
| 292 | code |  | Grad-CAM++ & Lesion Patch Extraction | Module 06 (Local Patches) / Module 13 (GradCAM) | Unexecuted Code | 4 | `import os` |
| 293 | code |  | Grad-CAM++ & Lesion Patch Extraction | Module 06 (Local Patches) / Module 13 (GradCAM) | Unexecuted Code | 0 | `<empty>` |
| 294 | code |  | Grad-CAM++ & Lesion Patch Extraction | Module 06 (Local Patches) / Module 13 (GradCAM) | Unexecuted Code | 17 | `# import shutil` |
| 295 | code |  | Grad-CAM++ & Lesion Patch Extraction | Module 06 (Local Patches) / Module 13 (GradCAM) | Unexecuted Code | 13 | `# import shutil` |
| 296 | code |  | Grad-CAM++ & Lesion Patch Extraction | Module 06 (Local Patches) / Module 13 (GradCAM) | Unexecuted Code | 96 | `import os, cv2, numpy as np` |
| 297 | code |  | Grad-CAM++ & Lesion Patch Extraction | Module 06 (Local Patches) / Module 13 (GradCAM) | Unexecuted Code | 0 | `<empty>` |
| 298 | code |  | Grad-CAM++ & Lesion Patch Extraction | Module 06 (Local Patches) / Module 13 (GradCAM) | Unexecuted Code | 96 | `import os, cv2, numpy as np` |
| 299 | code |  | Grad-CAM++ & Lesion Patch Extraction | Module 06 (Local Patches) / Module 13 (GradCAM) | Unexecuted Code | 96 | `import os, cv2, numpy as np` |
| 300 | code |  | Grad-CAM++ & Lesion Patch Extraction | Module 06 (Local Patches) / Module 13 (GradCAM) | Unexecuted Code | 96 | `import os, cv2, numpy as np` |
| 301 | code |  | Grad-CAM++ & Lesion Patch Extraction | Module 06 (Local Patches) / Module 13 (GradCAM) | Unexecuted Code | 96 | `import os, cv2, numpy as np` |
| 302 | code |  | Grad-CAM++ & Lesion Patch Extraction | Module 06 (Local Patches) / Module 13 (GradCAM) | Unexecuted Code | 18 | `import os` |
| 303 | code |  | Grad-CAM++ & Lesion Patch Extraction | Module 06 (Local Patches) / Module 13 (GradCAM) | Unexecuted Code | 0 | `<empty>` |
| 304 | code |  | Grad-CAM++ & Lesion Patch Extraction | Module 06 (Local Patches) / Module 13 (GradCAM) | Unexecuted Code | 0 | `<empty>` |
| 305 | code |  | Grad-CAM++ & Lesion Patch Extraction | Module 06 (Local Patches) / Module 13 (GradCAM) | Unexecuted Code | 0 | `<empty>` |
| 306 | code |  | Grad-CAM++ & Lesion Patch Extraction | Module 06 (Local Patches) / Module 13 (GradCAM) | Unexecuted Code | 7 | `# import shutil` |
| 307 | code |  | Grad-CAM++ & Lesion Patch Extraction | Module 06 (Local Patches) / Module 13 (GradCAM) | Unexecuted Code | 13 | `# !rm -rf /kaggle/working/lesion_patches/` |
| 308 | code |  | Grad-CAM++ & Lesion Patch Extraction | Module 06 (Local Patches) / Module 13 (GradCAM) | Unexecuted Code | 1 | `!du -h /kaggle/working/week2 --max-depth=2 \| sort -h` |
| 309 | code |  | Grad-CAM++ & Lesion Patch Extraction | Module 06 (Local Patches) / Module 13 (GradCAM) | Unexecuted Code | 0 | `<empty>` |
| 310 | code |  | Grad-CAM++ & Lesion Patch Extraction | Module 06 (Local Patches) / Module 13 (GradCAM) | Unexecuted Code | 16 | `import os` |
| 311 | markdown |  | Fusion Ablations & Imbalance Weighted Loss | Module 08 (Class Imbalance) / Module 11 (Ablations) | Markdown Documentation | 1 | `# without mean` |
| 312 | code | 51 | Fusion Ablations & Imbalance Weighted Loss | Module 08 (Class Imbalance) / Module 11 (Ablations) | Executed (Run #51) | 24 | `from sklearn.model_selection import train_test_split` |
| 313 | code | 48 | Fusion Ablations & Imbalance Weighted Loss | Module 08 (Class Imbalance) / Module 11 (Ablations) | Executed (Run #48) | 51 | `import os` |
| 314 | code | 49 | Fusion Ablations & Imbalance Weighted Loss | Module 08 (Class Imbalance) / Module 11 (Ablations) | Executed (Run #49) | 65 | `import numpy as np` |
| 315 | code | 52 | Fusion Ablations & Imbalance Weighted Loss | Module 08 (Class Imbalance) / Module 11 (Ablations) | Executed (Run #52) | 7 | `train_ds = FusionDataset(train_df, eff_emb, conv_emb, swin_emb, rad_df)` |
| 316 | code |  | Fusion Ablations & Imbalance Weighted Loss | Module 08 (Class Imbalance) / Module 11 (Ablations) | Unexecuted Code | 5 | `for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:` |
| 317 | code | 43 | Fusion Ablations & Imbalance Weighted Loss | Module 08 (Class Imbalance) / Module 11 (Ablations) | Executed (Run #43) | 106 | `import torch` |
| 318 | code |  | Fusion Ablations & Imbalance Weighted Loss | Module 08 (Class Imbalance) / Module 11 (Ablations) | Unexecuted Code | 164 | `import os` |
| 319 | code |  | Fusion Ablations & Imbalance Weighted Loss | Module 08 (Class Imbalance) / Module 11 (Ablations) | Unexecuted Code | 0 | `<empty>` |
| 320 | code |  | Fusion Ablations & Imbalance Weighted Loss | Module 08 (Class Imbalance) / Module 11 (Ablations) | Unexecuted Code | 27 | `from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay` |
| 321 | code |  | Fusion Ablations & Imbalance Weighted Loss | Module 08 (Class Imbalance) / Module 11 (Ablations) | Unexecuted Code | 22 | `from collections import Counter` |
| 322 | code |  | Fusion Ablations & Imbalance Weighted Loss | Module 08 (Class Imbalance) / Module 11 (Ablations) | Unexecuted Code | 38 | `import torch` |
| 323 | code |  | Fusion Ablations & Imbalance Weighted Loss | Module 08 (Class Imbalance) / Module 11 (Ablations) | Unexecuted Code | 28 | `# BLOCK 1 — t-SNE of Radiomics features (before fusion)` |
| 324 | code |  | Fusion Ablations & Imbalance Weighted Loss | Module 08 (Class Imbalance) / Module 11 (Ablations) | Unexecuted Code | 0 | `<empty>` |
| 325 | code |  | Fusion Ablations & Imbalance Weighted Loss | Module 08 (Class Imbalance) / Module 11 (Ablations) | Unexecuted Code | 44 | `# BLOCK 2 — t-SNE of Fused Latent Features (after fusion)` |
| 326 | code |  | Fusion Ablations & Imbalance Weighted Loss | Module 08 (Class Imbalance) / Module 11 (Ablations) | Unexecuted Code | 18 | `from sklearn.metrics import cohen_kappa_score` |
| 327 | code |  | Fusion Ablations & Imbalance Weighted Loss | Module 08 (Class Imbalance) / Module 11 (Ablations) | Unexecuted Code | 24 | `from sklearn.metrics import confusion_matrix` |
| 328 | code |  | Fusion Ablations & Imbalance Weighted Loss | Module 08 (Class Imbalance) / Module 11 (Ablations) | Unexecuted Code | 10 | `import seaborn as sns` |
| 329 | markdown |  | Fusion Ablations & Imbalance Weighted Loss | Module 08 (Class Imbalance) / Module 11 (Ablations) | Markdown Documentation | 1 | `# imbalance weighted loss` |
| 330 | code |  | Fusion Ablations & Imbalance Weighted Loss | Module 08 (Class Imbalance) / Module 11 (Ablations) | Unexecuted Code | 51 | `import os` |
| 331 | code |  | Fusion Ablations & Imbalance Weighted Loss | Module 08 (Class Imbalance) / Module 11 (Ablations) | Unexecuted Code | 65 | `import numpy as np` |
| 332 | code |  | Fusion Ablations & Imbalance Weighted Loss | Module 08 (Class Imbalance) / Module 11 (Ablations) | Unexecuted Code | 106 | `import torch` |
| 333 | code |  | Fusion Ablations & Imbalance Weighted Loss | Module 08 (Class Imbalance) / Module 11 (Ablations) | Unexecuted Code | 148 | `import os` |
| 334 | code |  | Fusion Ablations & Imbalance Weighted Loss | Module 08 (Class Imbalance) / Module 11 (Ablations) | Unexecuted Code | 21 | `from sklearn.metrics import confusion_matrix, classification_report` |
| 335 | code |  | Fusion Ablations & Imbalance Weighted Loss | Module 08 (Class Imbalance) / Module 11 (Ablations) | Unexecuted Code | 0 | `<empty>` |
| 336 | code |  | Fusion Ablations & Imbalance Weighted Loss | Module 08 (Class Imbalance) / Module 11 (Ablations) | Unexecuted Code | 26 | `# import os` |
| 337 | markdown |  | Patch-Level Radiomics (pyradiomics) | Module 06 (Local Radiomics) | Markdown Documentation | 1 | `# patch encoding group 4 continuation` |
| 338 | code |  | Patch-Level Radiomics (pyradiomics) | Module 06 (Local Radiomics) | Unexecuted Code | 10 | `import os` |
| 339 | code |  | Patch-Level Radiomics (pyradiomics) | Module 06 (Local Radiomics) | Unexecuted Code | 16 | `# =============================` |
| 340 | code |  | Patch-Level Radiomics (pyradiomics) | Module 06 (Local Radiomics) | Unexecuted Code | 35 | `import pandas as pd` |
| 341 | code |  | Patch-Level Radiomics (pyradiomics) | Module 06 (Local Radiomics) | Unexecuted Code | 30 | `import os` |
| 342 | code |  | Patch-Level Radiomics (pyradiomics) | Module 06 (Local Radiomics) | Unexecuted Code | 14 | `IMAGE_SIZE = 320  # not 224` |
| 343 | code |  | Patch-Level Radiomics (pyradiomics) | Module 06 (Local Radiomics) | Unexecuted Code | 24 | `def load_convnext_backbone(ckpt, num_classes=5):` |
| 344 | code |  | Patch-Level Radiomics (pyradiomics) | Module 06 (Local Radiomics) | Unexecuted Code | 17 | `from torch.utils.data import DataLoader` |
| 345 | code |  | Patch-Level Radiomics (pyradiomics) | Module 06 (Local Radiomics) | Unexecuted Code | 20 | `all_emb = []` |
| 346 | code |  | Patch-Level Radiomics (pyradiomics) | Module 06 (Local Radiomics) | Unexecuted Code | 21 | `import pandas as pd` |
| 347 | code |  | Patch-Level Radiomics (pyradiomics) | Module 06 (Local Radiomics) | Unexecuted Code | 38 | `import numpy as np` |
| 348 | markdown |  | Patch-Level Radiomics (pyradiomics) | Module 06 (Local Radiomics) | Markdown Documentation | 1 | `## patch level radiomics` |
| 349 | code |  | Patch-Level Radiomics (pyradiomics) | Module 06 (Local Radiomics) | Unexecuted Code | 1 | `!pip install pyradiomics -q` |
| 350 | code |  | Patch-Level Radiomics (pyradiomics) | Module 06 (Local Radiomics) | Unexecuted Code | 5 | `import logging` |
| 351 | code |  | Patch-Level Radiomics (pyradiomics) | Module 06 (Local Radiomics) | Unexecuted Code | 112 | `import os, time, warnings, logging` |
| 352 | code |  | Patch-Level Radiomics (pyradiomics) | Module 06 (Local Radiomics) | Unexecuted Code | 150 | `import os, time` |
| 353 | code |  | Patch-Level Radiomics (pyradiomics) | Module 06 (Local Radiomics) | Unexecuted Code | 6 | `import numpy as np` |
| 354 | markdown |  | Patch-Level Radiomics (pyradiomics) | Module 06 (Local Radiomics) | Markdown Documentation | 1 | `### how many patches per image` |
| 355 | code |  | Patch-Level Radiomics (pyradiomics) | Module 06 (Local Radiomics) | Unexecuted Code | 6 | `import pandas as pd` |
| 356 | code |  | Patch-Level Radiomics (pyradiomics) | Module 06 (Local Radiomics) | Unexecuted Code | 7 | `# Count number of patches (lesions) per image` |
| 357 | code |  | Patch-Level Radiomics (pyradiomics) | Module 06 (Local Radiomics) | Unexecuted Code | 26 | `import matplotlib.pyplot as plt` |
| 358 | code |  | Patch-Level Radiomics (pyradiomics) | Module 06 (Local Radiomics) | Unexecuted Code | 2 | `summary = df.groupby("class_id")["lesion_count"].agg(["mean","median","max","min","std"])` |
| 359 | markdown |  | Lesion Graph Construction & GATv2 Training | Module 07 (Alternative Graph Fusion) | Markdown Documentation | 1 | `# Global Lession Graph` |
| 360 | code |  | Lesion Graph Construction & GATv2 Training | Module 07 (Alternative Graph Fusion) | Unexecuted Code | 16 | `import numpy as np` |
| 361 | code |  | Lesion Graph Construction & GATv2 Training | Module 07 (Alternative Graph Fusion) | Unexecuted Code | 5 | `import numpy as np` |
| 362 | code |  | Lesion Graph Construction & GATv2 Training | Module 07 (Alternative Graph Fusion) | Unexecuted Code | 7 | `SAVE_PATH = "/kaggle/working/week2/node_features_full.npz"` |
| 363 | code |  | Lesion Graph Construction & GATv2 Training | Module 07 (Alternative Graph Fusion) | Unexecuted Code | 6 | `from collections import defaultdict` |
| 364 | code |  | Lesion Graph Construction & GATv2 Training | Module 07 (Alternative Graph Fusion) | Unexecuted Code | 2 | `# !pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv \` |
| 365 | code |  | Lesion Graph Construction & GATv2 Training | Module 07 (Alternative Graph Fusion) | Unexecuted Code | 29 | `import torch` |
| 366 | code |  | Lesion Graph Construction & GATv2 Training | Module 07 (Alternative Graph Fusion) | Unexecuted Code | 7 | `import torch` |
| 367 | code |  | Lesion Graph Construction & GATv2 Training | Module 07 (Alternative Graph Fusion) | Unexecuted Code | 26 | `import random` |
| 368 | code |  | Lesion Graph Construction & GATv2 Training | Module 07 (Alternative Graph Fusion) | Unexecuted Code | 31 | `import random` |
| 369 | code |  | Lesion Graph Construction & GATv2 Training | Module 07 (Alternative Graph Fusion) | Unexecuted Code | 35 | `import numpy as np` |
| 370 | code |  | Lesion Graph Construction & GATv2 Training | Module 07 (Alternative Graph Fusion) | Unexecuted Code | 13 | `import numpy as np` |
| 371 | code |  | Lesion Graph Construction & GATv2 Training | Module 07 (Alternative Graph Fusion) | Unexecuted Code | 6 | `coords = np.stack([emb_data["x1"], emb_data["y1"], emb_data["x2"], emb_data["y2"]],1)` |
| 372 | code |  | Lesion Graph Construction & GATv2 Training | Module 07 (Alternative Graph Fusion) | Unexecuted Code | 25 | `import matplotlib.pyplot as plt` |
| 373 | code |  | Lesion Graph Construction & GATv2 Training | Module 07 (Alternative Graph Fusion) | Unexecuted Code | 20 | `import numpy as np` |
| 374 | code |  | Lesion Graph Construction & GATv2 Training | Module 07 (Alternative Graph Fusion) | Unexecuted Code | 55 | `# import numpy as np` |
| 375 | code |  | Lesion Graph Construction & GATv2 Training | Module 07 (Alternative Graph Fusion) | Unexecuted Code | 34 | `# import numpy as np` |
| 376 | code |  | Lesion Graph Construction & GATv2 Training | Module 07 (Alternative Graph Fusion) | Unexecuted Code | 61 | `import numpy as np` |
| 377 | code |  | Lesion Graph Construction & GATv2 Training | Module 07 (Alternative Graph Fusion) | Unexecuted Code | 2 | `npz = np.load("/kaggle/working/week2/patch_embeddings_full.npz")` |
| 378 | markdown |  | Lesion Graph Construction & GATv2 Training | Module 07 (Alternative Graph Fusion) | Markdown Documentation | 1 | `# Group 5` |
| 379 | code |  | Lesion Graph Construction & GATv2 Training | Module 07 (Alternative Graph Fusion) | Unexecuted Code | 67 | `# prepare_node_vectors.py` |
| 380 | code |  | Lesion Graph Construction & GATv2 Training | Module 07 (Alternative Graph Fusion) | Unexecuted Code | 4 | `!pip uninstall -y torch torchvision torchaudio torch-geometric torch-scatter torch-sparse torch-clus` |
| 381 | code |  | Lesion Graph Construction & GATv2 Training | Module 07 (Alternative Graph Fusion) | Unexecuted Code | 4 | `import torch, torch_geometric` |
| 382 | code |  | Lesion Graph Construction & GATv2 Training | Module 07 (Alternative Graph Fusion) | Unexecuted Code | 19 | `import torch` |
| 383 | code |  | Lesion Graph Construction & GATv2 Training | Module 07 (Alternative Graph Fusion) | Unexecuted Code | 12 | `import torch` |
| 384 | code |  | Lesion Graph Construction & GATv2 Training | Module 07 (Alternative Graph Fusion) | Unexecuted Code | 3 | `nan_mask = np.isnan(rad)` |
| 385 | code |  | Lesion Graph Construction & GATv2 Training | Module 07 (Alternative Graph Fusion) | Unexecuted Code | 24 | `import numpy as np` |
| 386 | code |  | Lesion Graph Construction & GATv2 Training | Module 07 (Alternative Graph Fusion) | Unexecuted Code | 14 | `import numpy as np` |
| 387 | code |  | Lesion Graph Construction & GATv2 Training | Module 07 (Alternative Graph Fusion) | Unexecuted Code | 22 | `import torch` |
| 388 | code |  | Lesion Graph Construction & GATv2 Training | Module 07 (Alternative Graph Fusion) | Unexecuted Code | 210 | `import torch` |
| 389 | code |  | Lesion Graph Construction & GATv2 Training | Module 07 (Alternative Graph Fusion) | Unexecuted Code | 63 | `# DIAGNOSTIC CELL — run just this first` |
| 390 | code |  | Lesion Graph Construction & GATv2 Training | Module 07 (Alternative Graph Fusion) | Unexecuted Code | 163 | `# Robust GNN training (GATv2 but stable): LayerNorm, balanced sampler, LR scheduler, gradient clippi` |
| 391 | markdown |  | Exploratory Data Analysis, UMAP & Embeddings | Module 01 / Module 12 (EDA) | Markdown Documentation | 1 | `# EDA` |
| 392 | markdown |  | Exploratory Data Analysis, UMAP & Embeddings | Module 01 / Module 12 (EDA) | Markdown Documentation | 1 | `!mkdir EDA` |
| 393 | code |  | Exploratory Data Analysis, UMAP & Embeddings | Module 01 / Module 12 (EDA) | Unexecuted Code | 2 | `!pip uninstall -y protobuf` |
| 394 | code |  | Exploratory Data Analysis, UMAP & Embeddings | Module 01 / Module 12 (EDA) | Unexecuted Code | 8 | `import numpy as np` |
| 395 | code |  | Exploratory Data Analysis, UMAP & Embeddings | Module 01 / Module 12 (EDA) | Unexecuted Code | 2 | `import google.protobuf` |
| 396 | code |  | Exploratory Data Analysis, UMAP & Embeddings | Module 01 / Module 12 (EDA) | Unexecuted Code | 4 | `import numpy as np` |
| 397 | code | 8 | Exploratory Data Analysis, UMAP & Embeddings | Module 01 / Module 12 (EDA) | Executed (Run #8) | 6 | `EMB_PATHS = {` |
| 398 | code | 2 | Exploratory Data Analysis, UMAP & Embeddings | Module 01 / Module 12 (EDA) | Executed (Run #2) | 6 | `import numpy as np` |
| 399 | code | 5 | Exploratory Data Analysis, UMAP & Embeddings | Module 01 / Module 12 (EDA) | Executed (Run #5) | 9 | `import numpy as np` |
| 400 | code |  | Exploratory Data Analysis, UMAP & Embeddings | Module 01 / Module 12 (EDA) | Unexecuted Code | 3 | `for name, path in EMB_PATHS.items():` |
| 401 | code |  | Exploratory Data Analysis, UMAP & Embeddings | Module 01 / Module 12 (EDA) | Unexecuted Code | 1 | `!pip install -U scikit-learn umap-learn` |
| 402 | code |  | Exploratory Data Analysis, UMAP & Embeddings | Module 01 / Module 12 (EDA) | Unexecuted Code | 4 | `import sklearn` |
| 403 | code |  | Exploratory Data Analysis, UMAP & Embeddings | Module 01 / Module 12 (EDA) | Unexecuted Code | 14 | `import numpy as np` |
| 404 | code |  | Exploratory Data Analysis, UMAP & Embeddings | Module 01 / Module 12 (EDA) | Unexecuted Code | 2 | `import os` |
| 405 | code |  | Exploratory Data Analysis, UMAP & Embeddings | Module 01 / Module 12 (EDA) | Unexecuted Code | 36 | `import pandas as pd` |
| 406 | code |  | Exploratory Data Analysis, UMAP & Embeddings | Module 01 / Module 12 (EDA) | Unexecuted Code | 42 | `import pandas as pd` |
| 407 | code |  | Exploratory Data Analysis, UMAP & Embeddings | Module 01 / Module 12 (EDA) | Unexecuted Code | 48 | `from sklearn.decomposition import PCA` |
| 408 | code |  | Exploratory Data Analysis, UMAP & Embeddings | Module 01 / Module 12 (EDA) | Unexecuted Code | 51 | `import numpy as np` |
| 409 | code |  | Exploratory Data Analysis, UMAP & Embeddings | Module 01 / Module 12 (EDA) | Unexecuted Code | 33 | `import numpy as np` |
| 410 | code | 3 | Exploratory Data Analysis, UMAP & Embeddings | Module 01 / Module 12 (EDA) | Executed (Run #3) | 6 | `import pandas as pd` |
| 411 | code |  | Exploratory Data Analysis, UMAP & Embeddings | Module 01 / Module 12 (EDA) | Unexecuted Code | 1 | `print(rad_df.columns.tolist())` |
| 412 | code | 9 | Exploratory Data Analysis, UMAP & Embeddings | Module 01 / Module 12 (EDA) | Executed (Run #9) | 16 | `# Radiomics` |
| 413 | code |  | Exploratory Data Analysis, UMAP & Embeddings | Module 01 / Module 12 (EDA) | Unexecuted Code | 0 | `<empty>` |
| 414 | code |  | Exploratory Data Analysis, UMAP & Embeddings | Module 01 / Module 12 (EDA) | Unexecuted Code | 20 | `from sklearn.preprocessing import StandardScaler` |
| 415 | code |  | Exploratory Data Analysis, UMAP & Embeddings | Module 01 / Module 12 (EDA) | Unexecuted Code | 22 | `# --- Radiomics plot ---` |
| 416 | code |  | Exploratory Data Analysis, UMAP & Embeddings | Module 01 / Module 12 (EDA) | Unexecuted Code | 2 | `import torch` |
| 417 | code |  | Exploratory Data Analysis, UMAP & Embeddings | Module 01 / Module 12 (EDA) | Unexecuted Code | 0 | `<empty>` |
| 418 | code |  | Exploratory Data Analysis, UMAP & Embeddings | Module 01 / Module 12 (EDA) | Unexecuted Code | 0 | `<empty>` |
| 419 | code | 39 | Backbone Comparisons & CORAL Evaluation | Module 05 (Selection) / Module 09 (CORAL) | Failed (NameError: name 'train_loader' is not defined) | 61 | `import torch` |
| 420 | code |  | Backbone Comparisons & CORAL Evaluation | Module 05 (Selection) / Module 09 (CORAL) | Unexecuted Code | 28 | `from sklearn.linear_model import LogisticRegression` |
| 421 | code |  | Backbone Comparisons & CORAL Evaluation | Module 05 (Selection) / Module 09 (CORAL) | Unexecuted Code | 49 | `# Categorize outcomes` |
| 422 | code | 23 | Backbone Comparisons & CORAL Evaluation | Module 05 (Selection) / Module 09 (CORAL) | Failed (NameError: name 'train_df' is not defined) | 20 | `from sklearn.linear_model import LogisticRegression` |
| 423 | code |  | Backbone Comparisons & CORAL Evaluation | Module 05 (Selection) / Module 09 (CORAL) | Unexecuted Code | 16 | `def outcome_map(cnn_pred, fusion_pred, y_true):` |
| 424 | code |  | Backbone Comparisons & CORAL Evaluation | Module 05 (Selection) / Module 09 (CORAL) | Unexecuted Code | 10 | `from umap import UMAP` |
| 425 | code |  | Backbone Comparisons & CORAL Evaluation | Module 05 (Selection) / Module 09 (CORAL) | Unexecuted Code | 28 | `for name, outcome in zip(` |
| 426 | code |  | Backbone Comparisons & CORAL Evaluation | Module 05 (Selection) / Module 09 (CORAL) | Unexecuted Code | 12 | `conf_df = pd.DataFrame({` |
| 427 | code |  | Backbone Comparisons & CORAL Evaluation | Module 05 (Selection) / Module 09 (CORAL) | Unexecuted Code | 16 | `from sklearn.metrics import confusion_matrix` |
| 428 | code |  | Backbone Comparisons & CORAL Evaluation | Module 05 (Selection) / Module 09 (CORAL) | Unexecuted Code | 26 | `import pandas as pd` |
| 429 | code |  | Backbone Comparisons & CORAL Evaluation | Module 05 (Selection) / Module 09 (CORAL) | Unexecuted Code | 43 | `import networkx as nx` |
| 430 | markdown |  | Backbone Comparisons & CORAL Evaluation | Module 05 (Selection) / Module 09 (CORAL) | Markdown Documentation | 0 | `<empty>` |
| 431 | code | 13 | Backbone Comparisons & CORAL Evaluation | Module 05 (Selection) / Module 09 (CORAL) | Executed (Run #13) | 8 | `import numpy as np` |
| 432 | code |  | Backbone Comparisons & CORAL Evaluation | Module 05 (Selection) / Module 09 (CORAL) | Unexecuted Code | 28 | `# Absolute correlation matrix` |
| 433 | code |  | Backbone Comparisons & CORAL Evaluation | Module 05 (Selection) / Module 09 (CORAL) | Unexecuted Code | 14 | `# import shutil` |
| 434 | code |  | Backbone Comparisons & CORAL Evaluation | Module 05 (Selection) / Module 09 (CORAL) | Unexecuted Code | 4 | `import numpy as np` |
| 435 | code |  | Backbone Comparisons & CORAL Evaluation | Module 05 (Selection) / Module 09 (CORAL) | Unexecuted Code | 0 | `<empty>` |
| 436 | code | 10 | Backbone Comparisons & CORAL Evaluation | Module 05 (Selection) / Module 09 (CORAL) | Executed (Run #10) | 3 | `from sklearn.preprocessing import StandardScaler` |
| 437 | code | 17 | Backbone Comparisons & CORAL Evaluation | Module 05 (Selection) / Module 09 (CORAL) | Executed (Run #17) | 9 | `# Merge labels from manifest` |
| 438 | code | 18 | Backbone Comparisons & CORAL Evaluation | Module 05 (Selection) / Module 09 (CORAL) | Executed (Run #18) | 8 | `X_rad = rad_df_labeled[feature_cols].values` |
| 439 | code | 19 | Backbone Comparisons & CORAL Evaluation | Module 05 (Selection) / Module 09 (CORAL) | Executed (Run #19) | 6 | `order = np.argsort(severity_score)` |
| 440 | code | 20 | Backbone Comparisons & CORAL Evaluation | Module 05 (Selection) / Module 09 (CORAL) | Executed (Run #20) | 30 | `import pandas as pd` |
| 441 | code | 24 | Backbone Comparisons & CORAL Evaluation | Module 05 (Selection) / Module 09 (CORAL) | Executed (Run #24) | 16 | `from sklearn.metrics import cohen_kappa_score` |
| 442 | code | 35 | Backbone Comparisons & CORAL Evaluation | Module 05 (Selection) / Module 09 (CORAL) | Executed (Run #35) | 12 | `# Paths` |
| 443 | code | 36 | Backbone Comparisons & CORAL Evaluation | Module 05 (Selection) / Module 09 (CORAL) | Executed (Run #36) | 8 | `# Load embeddings (already in manifest order)` |
| 444 | code | 28 | Backbone Comparisons & CORAL Evaluation | Module 05 (Selection) / Module 09 (CORAL) | Executed (Run #28) | 9 | `from sklearn.model_selection import train_test_split` |
| 445 | code | 32 | Backbone Comparisons & CORAL Evaluation | Module 05 (Selection) / Module 09 (CORAL) | Failed (NameError: name 'test_df' is not defined) | 2 | `test_idx = test_df.index.values` |
| 446 | code | 33 | Backbone Comparisons & CORAL Evaluation | Module 05 (Selection) / Module 09 (CORAL) | Executed (Run #33) | 8 | `def backbone_qwk(X, train_idx, test_idx, y):` |
| 447 | code | 38 | Backbone Comparisons & CORAL Evaluation | Module 05 (Selection) / Module 09 (CORAL) | Executed (Run #38) | 6 | `qwk_eff,  pred_eff  = backbone_qwk(X_eff,  train_idx, test_idx, y)` |
| 448 | code | 44 | Backbone Comparisons & CORAL Evaluation | Module 05 (Selection) / Module 09 (CORAL) | Executed (Run #44) | 106 | `import torch` |
| 449 | code | 55 | Backbone Comparisons & CORAL Evaluation | Module 05 (Selection) / Module 09 (CORAL) | Executed (Run #55) | 19 | `# from torch.utils.data import DataLoader` |
| 450 | code | 53 | Backbone Comparisons & CORAL Evaluation | Module 05 (Selection) / Module 09 (CORAL) | Executed (Run #53) | 65 | `import torch` |
| 451 | code | 56 | Backbone Comparisons & CORAL Evaluation | Module 05 (Selection) / Module 09 (CORAL) | Executed (Run #56) | 1 | `qwk_fusion = test_qwk` |
| 452 | code | 57 | Backbone Comparisons & CORAL Evaluation | Module 05 (Selection) / Module 09 (CORAL) | Executed (Run #57) | 37 | `import plotly.express as px` |
| 453 | code | 58 | Backbone Comparisons & CORAL Evaluation | Module 05 (Selection) / Module 09 (CORAL) | Executed (Run #58) | 9 | `abs_err = np.abs(y_pred - y_true)` |
| 454 | code | 59 | Backbone Comparisons & CORAL Evaluation | Module 05 (Selection) / Module 09 (CORAL) | Executed (Run #59) | 38 | `from sklearn.calibration import calibration_curve` |
| 455 | code | 62 | Backbone Comparisons & CORAL Evaluation | Module 05 (Selection) / Module 09 (CORAL) | Executed (Run #62) | 9 | `# abs_err = np.abs(y_pred - y_true)` |
| 456 | markdown |  | Clinical Decision Rules, Error Analysis & Visuals | Module 12 (Evaluation) / Module 14 (Demo Logic) | Markdown Documentation | 1 | `# Test` |
| 457 | code |  | Clinical Decision Rules, Error Analysis & Visuals | Module 12 (Evaluation) / Module 14 (Demo Logic) | Unexecuted Code | 51 | `import os` |
| 458 | code |  | Clinical Decision Rules, Error Analysis & Visuals | Module 12 (Evaluation) / Module 14 (Demo Logic) | Unexecuted Code | 106 | `import torch` |
| 459 | code |  | Clinical Decision Rules, Error Analysis & Visuals | Module 12 (Evaluation) / Module 14 (Demo Logic) | Unexecuted Code | 11 | `# # Load checkpoint first` |
| 460 | code | 64 | Clinical Decision Rules, Error Analysis & Visuals | Module 12 (Evaluation) / Module 14 (Demo Logic) | Executed (Run #64) | 11 | `# Load checkpoint first` |
| 461 | code |  | Clinical Decision Rules, Error Analysis & Visuals | Module 12 (Evaluation) / Module 14 (Demo Logic) | Unexecuted Code | 65 | `import numpy as np` |
| 462 | code |  | Clinical Decision Rules, Error Analysis & Visuals | Module 12 (Evaluation) / Module 14 (Demo Logic) | Unexecuted Code | 7 | `train_ds = FusionDataset(train_df, eff_emb, conv_emb, swin_emb, rad_df)` |
| 463 | code | 65 | Clinical Decision Rules, Error Analysis & Visuals | Module 12 (Evaluation) / Module 14 (Demo Logic) | Executed (Run #65) | 4 | `test_acc, test_qwk, y_pred, y_true, conf = evaluate(fusion_model, test_loader)` |
| 464 | code |  | Clinical Decision Rules, Error Analysis & Visuals | Module 12 (Evaluation) / Module 14 (Demo Logic) | Unexecuted Code | 0 | `<empty>` |
| 465 | code | 66 | Clinical Decision Rules, Error Analysis & Visuals | Module 12 (Evaluation) / Module 14 (Demo Logic) | Executed (Run #66) | 7 | `def clinical_decision(conf):` |
| 466 | code | 68 | Clinical Decision Rules, Error Analysis & Visuals | Module 12 (Evaluation) / Module 14 (Demo Logic) | Executed (Run #68) | 35 | `import torch` |
| 467 | code | 86 | Clinical Decision Rules, Error Analysis & Visuals | Module 12 (Evaluation) / Module 14 (Demo Logic) | Executed (Run #86) | 30 | `import pandas as pd` |
| 468 | code | 87 | Clinical Decision Rules, Error Analysis & Visuals | Module 12 (Evaluation) / Module 14 (Demo Logic) | Executed (Run #87) | 4 | `import numpy as np` |
| 469 | code | 88 | Clinical Decision Rules, Error Analysis & Visuals | Module 12 (Evaluation) / Module 14 (Demo Logic) | Executed (Run #88) | 5 | `def visual_category(row):` |
| 470 | code | 89 | Clinical Decision Rules, Error Analysis & Visuals | Module 12 (Evaluation) / Module 14 (Demo Logic) | Executed (Run #89) | 35 | `import plotly.express as px` |
| 471 | code | 90 | Clinical Decision Rules, Error Analysis & Visuals | Module 12 (Evaluation) / Module 14 (Demo Logic) | Executed (Run #90) | 13 | `from sklearn.metrics import accuracy_score, cohen_kappa_score` |
| 472 | code | 91 | Clinical Decision Rules, Error Analysis & Visuals | Module 12 (Evaluation) / Module 14 (Demo Logic) | Executed (Run #91) | 6 | `import numpy as np` |
| 473 | code | 92 | Clinical Decision Rules, Error Analysis & Visuals | Module 12 (Evaluation) / Module 14 (Demo Logic) | Executed (Run #92) | 50 | `import plotly.graph_objects as go` |
