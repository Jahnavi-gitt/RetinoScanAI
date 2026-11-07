# Module 00: Comprehensive Project and Notebook Audit Report

**Project Title**: Deep Learning for Diabetic Retinopathy Grading  
**Audited Notebook**: `dr-ansh (11).ipynb` (backed up to `original_files/dr-ansh (11).ipynb`)  
**Auditor**: Senior Deep Learning Engineer & CV Researcher  
**Execution Environment**: Apple M5 (ARM64), macOS 25.6.0, 16 GB Unified Memory  
**Audit Date**: October 25, 2026  

---

## 1. Executive Summary

A comprehensive, cell-by-cell inspection of `dr-ansh (11).ipynb` (474 cells: 26 Markdown cells, 448 Code cells) was performed. The notebook represents an evolving, multi-week research project originally developed on the Kaggle platform with an attached GPU runtime.

### Key Audit Findings:
1. **Critical Dataset Realignment Identified (Cell 206)**:
   In earlier cells (0–205), experiments attempted to fine-tune models on the **EyePACS** dataset while extracting radiomics features on the **DDR (Dataset for Diabetic Retinopathy)** dataset. In cell 206, the authors documented this discrepancy:
   > `### misaligned as the models(eyepac) and radiomics(DDR)`  
   From Cell 211 onwards, the codebase correctly transitioned to using the **DDR dataset** across all fine-tuning loops, embedding extractions, and radiomics models. This ensures features and image representations correspond to the same retinal fundus domain.
2. **Baseline Architectures Implemented in Notebook**:
   The notebook contains explicit fine-tuning code for all three required candidate baselines on DDR:
   - **EfficientNet-B3** (Cells 222–229) using `timm.create_model("efficientnet_b3", pretrained=False/True)`
   - **ConvNeXt-Small** (Cells 230–235) using `timm.create_model("convnext_small", pretrained=False/True)`
   - **Swin-Tiny** (Cells 236–242) using `timm.create_model("swin_tiny_patch4_window7_224", pretrained=False/True)`
3. **Existing Novelty Implementations vs. Proposed Enhancements**:
   - *Existing in Notebook*: Global radiomics extraction (107 features), multi-backbone concatenation via MLP, Grad-CAM++ lesion bounding-box extraction, a Graph Attention Network (GATv2 with Torch-Geometric) on lesion graphs, and CORAL ordinal heads.
   - *Proposed Refinements*: The notebook's multi-backbone concatenation and heavy GAT graph modules are computationally cumbersome, fragile, and lack unified spatial association. Our master architecture refines this into:
     1. Selecting the single **best baseline backbone** via validation metrics (Module 05).
     2. **Spatially Localized Radiomics** over a structured 3×3 retinal grid (Module 06).
     3. **Spatial Visual–Radiomic Cross-Attention** fusing backbone feature tokens directly with regional radiomic tokens (Module 07).
     4. **Minority-aware Class Imbalance Strategies** (Module 08).
     5. **CORAL Ordinal Grading** integrated onto the fused representation (Module 09).
4. **Notebook Health & Execution State**:
   - 44 code cells have execution outputs; 397 code cells are unexecuted; 3 cells have NameErrors (cells 419, 422, 445) caused by missing in-memory variables in an interactive session.
   - All paths are hard-coded to Kaggle paths (`/kaggle/input/...` and `/kaggle/working/...`).
   - The original notebook is preserved untouched, with a safe backup in `original_files/dr-ansh (11).ipynb`.

---

## 2. Dataset and Annotation Audit

### 2.1 Target Dataset: DDR (Dataset for Diabetic Retinopathy)
- **Primary Image Root**: Referenced as `/kaggle/input/ddr-final/final` or `/kaggle/input/ddr-final/images`.
- **Annotation Manifests**:
  - `updated_ddr_manifest.csv` (`/kaggle/input/ddr-manifest-updated/updated_ddr_manifest.csv`)
  - `manifest_ddr_aligned.csv` (`/kaggle/input/manifest-ddr-aligned/manifest_ddr_aligned.csv`)
- **DR Severity Scale (5-Grade Ordinal Mapping)**:
  - `0`: No DR (Normal)
  - `1`: Mild NPDR (Microaneurysms only)
  - `2`: Moderate NPDR (More than microaneurysms, but less than severe NPDR)
  - `3`: Severe NPDR (20+ intraretinal hemorrhages in each of 4 quadrants, or definite venous beading in 2+ quadrants, or prominent IRMA in 1+ quadrant)
  - `4`: PDR (Proliferative DR: Neovascularization and/or vitreous hemorrhage)
- **Secondary / Pretraining Dataset**: EyePACS (`/kaggle/input/eyepack-final/eypac_final`), used in legacy cells for initial weights before DDR fine-tuning.

### 2.2 Data Loading & Preprocessing
- **Image Operations**:
  - Circular retinal masking: `closing(th > 0, disk(10))` (Cell 100).
  - Contrast enhancement: CLAHE in LAB color space (`cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))` on L channel).
  - Resizing: DDR images resized to (384, 384) for radiomics, and (224, 224) or (300, 300) for backbones.
  - Normalization: Standard ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.
- **Known Issues**:
  - Image paths in manifests have inconsistent filename extensions (e.g. some entries include `.jpg`, others omit it, requiring double-check logic like `resolve_image_path`).
  - Augmentation is haphazardly placed across notebook cells; it must be standardized and restricted strictly to training loaders.

---

## 3. Baseline Backbones Audit

| Backbone | Notebook Implementation | Input Size | Head Adaptation | Initialization |
|---|---|---|---|---|
| **ConvNeXt-Small** | Cells 180, 230–235 | 224×224 | `m.head.fc = nn.Linear(768, num_classes)` | `timm.create_model("convnext_small", pretrained=...)` |
| **Swin-Tiny** | Cells 236–242 | 224×224 | `m.head.fc = nn.Linear(768, num_classes)` | `timm.create_model("swin_tiny_patch4_window7_224", pretrained=...)` |
| **EfficientNet-B3** | Cells 28, 222–229 | 300×300 (or 224) | `m.classifier = nn.Linear(1536, num_classes)` | `timm.create_model("efficientnet_b3", pretrained=...)` |

**Baseline Training Loop Observations**:
- Optimizers: AdamW with `lr=1e-4` to `3e-4`, `weight_decay=1e-2`.
- Schedulers: CosineAnnealingLR or custom `cosine_warmup_scheduler` with 1–2 warmup epochs.
- Gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)`.
- Primary Metric: Quadratic Weighted Kappa (`cohen_kappa_score(..., weights="quadratic")`).

---

## 4. Radiomics and Feature Extraction Audit

### 4.1 Global Radiomics (Cells 98–106)
- **Extracted Feature Count**: 107 numerical features per fundus image.
- **Feature Breakdown**:
  1. **PyRadiomics Features** (~93 features): First-order intensity statistics, GLCM (Gray-Level Co-occurrence Matrix), GLRLM (Run-Length Matrix), GLSZM (Size-Zone Matrix), NGTDM (Neighbouring Gray Tone Difference Matrix), GLDM (Dependence Matrix).
  2. **Local Binary Patterns (LBP)**: 10 uniform histogram bins (`skimage.feature.local_binary_pattern(gray, P=8, R=1, method='uniform')`).
  3. **Vessel Morphometry** (4 features): Vessel length (skeleton pixel count), branch point count via 3×3 neighbor convolution, total vessel area, and vessel density ratio.
- **Limitations in Existing Code**:
  - Global extraction averages out focal retinal micro-lesions (microaneurysms, small hard exudates).
  - Pre-extracted CSV was generated on Kaggle; extraction requires `pyradiomics` and `SimpleITK`.

### 4.2 Patch-Level Radiomics (Cells 337–358)
- Notebook explored extracting patches around CAM high-activation areas. However, this relied on dynamic thresholding and lacked consistent spatial coordinates across samples.
- **Master Plan Improvement**: Adopt a clean, deterministic 3×3 spatial grid over the fundus image, extracting localized texture and intensity descriptors per grid cell to align with visual feature tokens.

---

## 5. Attention and Fusion Audit

### 5.1 Existing Fusion Approaches in Notebook
1. **Late Fusion MLP (Cells 254–259)**:
   Concatenates pre-extracted embeddings from all 3 backbones (1536 + 768 + 768 = 3072 dims) with 107 global radiomic features (total 3179 dims), passing through a 3-layer MLP.
   *Critique*: Computationally wasteful to require inference across 3 heavy backbones for every image. Lacks spatial grounding.
2. **Lesion Graph GATv2 (Cells 359–390)**:
   Extracts patches, connects them as graph nodes, and runs a Graph Attention Network using `torch_geometric`.
   *Critique*: Adds heavy dependency on `torch_geometric`, suffers from variable patch counts, and is prone to instability (noted in notebook comments: `StableGAT`, `TinyCheck`).

### 5.2 Master Plan Design: Spatial Visual–Radiomic Cross-Attention (Module 07)
- Select the single best backbone from Module 05.
- Extract its spatial convolutional / transformer feature map (e.g. 7×7 grid of visual tokens).
- Project regional radiomic tokens (e.g. 3×3 grid) into a matching embedding dimension.
- Multi-Head Cross-Attention fuses visual queries with radiomic keys/values (or bidirectional).
- Fused tokens are globally pooled and passed to the classification / ordinal head.
- This provides an elegant, scientifically rigorous mechanism without requiring graph neural networks.

---

## 6. Loss Functions and Ordinal Grading Audit

### 6.1 Loss Functions
- **Standard Cross-Entropy**: Used as baseline.
- **Focal Loss** (Cells 29, 73, 177, 212): Implemented with $\gamma = 2.0$, $\alpha = 1.0$, and optional label smoothing.
- **Weighted Cross-Entropy** (Cell 329): Class weights inversely proportional to frequency.

### 6.2 CORAL Ordinal Grading (Cells 256, 317, 332, 448)
- **Target Formulation**: For $K=5$ classes, generates $K-1 = 4$ binary targets:
  $y^{(k)} = \mathbb{I}(y > k)$ for $k \in \{0, 1, 2, 3\}$.
- **Loss**: Binary Cross-Entropy with Logits averaged across the 4 thresholds.
- **Prediction Reconstruction**: Sum of sigmoids thresholded at 0.5:
  $\hat{y} = \sum_{k=0}^{K-2} \mathbb{I}(\sigma(s_k) > 0.5)$.
- **Status in Notebook**: Correctly formulated and used in fusion experiments. Reusable directly for Module 09.

---

## 7. Hard-Coded Paths and Environmental Blockers

All dataset and checkpoint references in `dr-ansh (11).ipynb` point to `/kaggle/input/` and `/kaggle/working/`.  
When running locally:
1. Manifests and dataset paths must be decoupled into `configs/dataset_config.yaml`.
2. PyTorch MPS acceleration (`torch.device("mps")`) will be configured for Apple Silicon.
3. Checkpoints and feature caches must be stored in standardized directories under `checkpoints/` and `data_manifests/`.

---

## 8. Summary of Notebook Cells Mapping to Project Modules

| Module | Module Name | Relevant Notebook Cells | Usability Assessment |
|---|---|---|---|
| **Module 00** | Project Audit | All cells (0–474) | Complete audit report, cell breakdown, and hardware matrix generated. |
| **Module 01** | Dataset & Preprocessing | 28, 94–106, 220–224 | Reusable CLAHE, circular mask, and manifest schema; paths need localization. |
| **Module 02** | ConvNeXt-Small Baseline | 180, 230–235 | Reusable architecture and fine-tuning loop; needs modularization. |
| **Module 03** | Swin-Tiny Baseline | 236–242 | Reusable architecture and fine-tuning loop; needs modularization. |
| **Module 04** | EfficientNet-B3 Baseline | 28, 222–229 | Reusable architecture and fine-tuning loop; needs modularization. |
| **Module 05** | Model Selection | 419–455 | Needs standardized QWK/F1/MAE evaluation across all 3 trained backbones. |
| **Module 06** | Local Radiomics | 98–106, 337–358 | PyRadiomics + LBP + skeleton logic reusable; upgrade to 3×3 spatial grid. |
| **Module 07** | Spatial Cross-Attention | 359–390 (GAT reference) | Replace GAT with clean PyTorch MultiheadAttention between visual & radiomic tokens. |
| **Module 08** | Class Imbalance | 29, 73, 128, 329 | FocalLoss and class-weighting formulas verified; needs controlled benchmark. |
| **Module 09** | CORAL Ordinal Grading | 256, 317, 332, 448 | Mathematical formulation verified; port to `src/models/coral.py`. |
| **Module 10** | Integrated Proposed Model | 254–259, 314–318 | Refactor into end-to-end model (Best Backbone + Local Rad + Cross-Attn + CORAL). |
| **Module 11** | Ablation Studies | 311–336, 446 | Structure formal A0–A5 ablation suite. |
| **Module 12** | Final Evaluation | 419–455, 463 | Port QWK, confusion matrix, and classification report functions. |
| **Module 13** | Explainability & Grad-CAM | 42–60, 260–310 | Reusable Grad-CAM implementation; target selected best backbone. |
| **Module 14** | Demo Application | 465–473 | Reusable clinical decision thresholds; wrap in modern Streamlit UI. |
| **Module 15** | Reproducibility & Docs | N/A | Build unified documentation, viva guide, and reproducible runner. |

---

## 9. Conclusion and Next Steps

Module 00 is successfully completed. The project structure is established, the original notebook is backed up in `original_files/`, and all cells and algorithms have been thoroughly analyzed.  
We are ready to proceed to **Module 01: Dataset Validation and Reproducible Preprocessing**.
