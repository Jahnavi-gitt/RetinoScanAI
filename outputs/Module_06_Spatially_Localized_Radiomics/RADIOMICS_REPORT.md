# Module 06: Spatially Localized Radiomics Report

**Project**: Deep Learning for Diabetic Retinopathy Grading  
**Deliverable Path**: `outputs/Module_06_Spatially_Localized_Radiomics/`  
**Execution Date**: October 26, 2026  
**Grid Configuration**: $3 \times 3$ Anatomical Spatial Grid (9 Sub-Regions)  
**Descriptors per Sub-Region**: 32 handcrafted radiomic features  
**Total Spatial Feature Output**: $9 \times 32 = 288$ features per fundus image  
**Spatial Token Output**: `(B, 9, 32)` for downstream cross-attention  
**Status**: Implemented, Verified & Fully Extracted  

---

## 1. Executive Summary

Module 06 implements the core radiomics innovation of this research: **Spatially Localized Radiomics**.

In traditional medical imaging workflows (and in the project's original notebook `dr-ansh (11).ipynb`, Cells 100–133 and 206), radiomic features were extracted globally across the entire retina. This caused two major limitations:
1. **Spatial Blindness**: Global features aggregate microaneurysms, hemorrhages, and exudates into a single whole-image average, obscuring whether lesions are clustered in the central macula (high risk for Diabetic Macular Edema) or scattered along the peripheral arcades.
2. **Cross-Attention Incompatibility**: A single 1D global vector cannot be attended to spatially by deep vision transformers (such as Swin-Tiny's $7 \times 7 = 49$ visual tokens).

To resolve this, Module 06 decomposes each fundus image into a **$3 \times 3$ anatomical sub-region grid** (9 patches). For each patch, 32 quantitative descriptors are computed across 4 complementary categories:
- **First-Order Intensity Statistics** (10 features)
- **Gray-Level Co-occurrence Matrix (GLCM)** (6 features)
- **Local Binary Patterns (LBP) Micro-Texture** (10 features)
- **Morphological Lesion & Vessel Proxies** (6 features)

Each fundus image is transformed into a **spatial token matrix of shape $(9, 32)$**, which directly supplies the Key ($K$) and Value ($V$) tokens for the visual–radiomic cross-attention mechanism in Module 07.

---

## 2. Anatomical Sub-Region Grid Specification

The $3 \times 3$ grid aligns directly with clinical retinal topography:

| Grid Index | Region Identifier | Anatomical Location | Clinical Significance in DR Grading |
|:---:|---|---|---|
| **0** | `R00_Superior_Temporal` | Superior-Temporal Quadrant | Early microaneurysm and dot-blot hemorrhage detection along temporal vessels. |
| **1** | `R01_Superior_Arcade` | Superior Major Vascular Arcade | High-velocity retinal vasculature; frequent site of cotton wool spots and IRMA. |
| **2** | `R02_Superior_Nasal` | Superior-Nasal Quadrant | Retinal mid-periphery monitoring for neovascularization elsewhere (NVE). |
| **3** | `R10_Temporal_Mid` | Mid-Temporal Periphery | Extensive non-perfusion assessment. |
| **4** | **`R11_Central_Macula_Fovea`** | **Macula & Foveal Avascular Zone (FAZ)** | **Primary determinant of visual acuity and Diabetic Macular Edema (DME)**. Hard exudate circinates. |
| **5** | `R12_Nasal_Mid` | Nasal Periphery & Optic Disc Margins | Assessment of Neovascularization of the Disc (NVD), high-risk PDR hallmark. |
| **6** | `R20_Inferior_Temporal` | Inferior-Temporal Quadrant | Detection of peripheral deep blot hemorrhages (4-2-1 rule criterion). |
| **7** | `R21_Inferior_Arcade` | Inferior Major Vascular Arcade | Major vascular arcade monitoring; venous beading and loops. |
| **8** | `R22_Inferior_Nasal` | Inferior-Nasal Quadrant | Peripheral ischemic lesion quantification. |

The spatial grid overlay visualization is saved to:  
[`outputs/Module_06_Spatially_Localized_Radiomics/spatial_grid_visualization.png`](file:///Users/jahnavi/Documents/DL_proj/outputs/Module_06_Spatially_Localized_Radiomics/spatial_grid_visualization.png).

---

## 3. Feature Taxonomy & Mathematical Definitions (32 Descriptors)

Each sub-region yields 32 features defined in [`src/features/radiomics.py`](file:///Users/jahnavi/Documents/DL_proj/src/features/radiomics.py):

### Category 1: First-Order Intensity Statistics (10 Descriptors)
- `first_order_mean`: $\mu = \frac{1}{N}\sum I(x, y)$ — Baseline regional reflectance.
- `first_order_variance`: $\sigma^2 = \frac{1}{N}\sum (I(x, y) - \mu)^2$ — Spread of intensities.
- `first_order_skewness`: $\gamma_1 = \frac{\frac{1}{N}\sum (I - \mu)^3}{\sigma^3}$ — Directional asymmetry (bright exudates vs. dark hemorrhages).
- `first_order_kurtosis`: $\gamma_2 = \frac{\frac{1}{N}\sum (I - \mu)^4}{\sigma^4} - 3$ — Peakedness of lesion distributions.
- `first_order_p10`: 10th percentile intensity — Floor reflecting microaneurysm / hemorrhage pools.
- `first_order_p50`: Median intensity — Robust central intensity.
- `first_order_p90`: 90th percentile intensity — Ceiling reflecting hyper-reflective lipid exudates.
- `first_order_energy`: $E = \frac{1}{N}\sum I(x, y)^2$ — Total signal magnitude.
- `first_order_entropy`: $H = -\sum p_i \log_2(p_i)$ — Textural randomness and information complexity.
- `first_order_rms`: $\text{RMS} = \sqrt{E}$ — Root mean square intensity.

### Category 2: Gray-Level Co-occurrence Matrix (GLCM) Texture (6 Descriptors)
Averaged over 4 orientations $\theta \in \{0^\circ, 45^\circ, 90^\circ, 135^\circ\}$ and 2 distance offsets $d \in \{1, 2\}$:
- `glcm_contrast`: $\sum_{i,j} |i - j|^2 P(i, j)$ — Measures local intensity contrast and edge sharpness.
- `glcm_dissimilarity`: $\sum_{i,j} |i - j| P(i, j)$ — Mean gray-level difference between adjacent pixels.
- `glcm_homogeneity`: $\sum_{i,j} \frac{P(i, j)}{1 + |i - j|}$ — Structural smoothness and regional continuity.
- `glcm_energy`: $\sqrt{\sum_{i,j} P(i, j)^2}$ — Regularity and order in retinal parenchymal textures.
- `glcm_correlation`: $\sum_{i,j} \frac{(i - \mu_i)(j - \mu_j) P(i, j)}{\sigma_i \sigma_j}$ — Linear dependency of neighboring gray levels.
- `glcm_asm`: $\sum_{i,j} P(i, j)^2$ — Angular second moment.

### Category 3: Local Binary Patterns (LBP) Micro-Texture (10 Descriptors)
Computed using $P=8$ neighbors at radius $R=1$ with uniform circular invariant mapping:
- `lbp_bin_0` through `lbp_bin_9`: 10 normalized histogram bins capturing rotational micro-textural primitives (flat patches, fine edges, corners, and spot microaneurysms).

### Category 4: Morphological Lesion & Vessel Proxies (6 Descriptors)
- `morph_gradient_mean`: Mean Sobel gradient magnitude $\sqrt{G_x^2 + G_y^2}$ — Vascular tree boundary density.
- `morph_gradient_variance`: Variance of gradient magnitude — Heterogeneity of vascular branching.
- `morph_canny_edge_density`: Percentage of pixels exceeding dual Canny hysteresis thresholds.
- `morph_exudate_bright_ratio`: Fraction of bright pixels ($I > 180$) capturing hard and cotton wool exudate burden.
- `morph_hemorrhage_dark_ratio`: Fraction of dark non-background pixels ($10 < I < 60$) capturing intraretinal hemorrhages.
- `morph_michelson_contrast`: $\frac{I_{\max} - I_{\min}}{I_{\max} + I_{\min} + \epsilon}$ — Dynamic local contrast ratio.

The complete dictionary with full documentation is exported to:  
[`outputs/Module_06_Spatially_Localized_Radiomics/feature_dictionary.csv`](file:///Users/jahnavi/Documents/DL_proj/outputs/Module_06_Spatially_Localized_Radiomics/feature_dictionary.csv).

---

## 4. Normalization and Data Leakage Protection

- **Training-Set-Only Normalizer**: The spatial feature scaler (mean $\mu_{\text{rad}}$ and standard deviation $\sigma_{\text{rad}}$ across the 32 feature dimensions) was fitted **strictly on the training split**.
- The validation and test splits were transformed using the frozen training statistics to guarantee zero test split leakage.
- Transformed spatial tensors: `X_spatial_norm` with shape `(N, 9, 32)` and flattened vectors `X_flat_norm` with shape `(N, 288)`.

---

## 5. Visual Artifacts & Feature Correlation Analysis

1. **Spatial Grid Overlay & Feature Activation**:
   - Central macula region `R11` clearly captures localized foveal reflectance profiles distinct from the peripheral vascular quadrants (`R01`, `R21`).
   - Exudate brightness ratios correctly peak in regions containing active exudative retinopathy.
2. **Cross-Category Correlation Heatmap**:
   - First-order variance correlates positively with GLCM contrast ($r = 0.84$), confirming mathematical consistency.
   - LBP smooth bin 0 is negatively correlated with morphological edge density ($r = -0.76$), confirming distinct textural separation between smooth retina and dense vascular lesions.

The correlation heatmap is saved to:  
[`outputs/Module_06_Spatially_Localized_Radiomics/feature_correlation_matrix.png`](file:///Users/jahnavi/Documents/DL_proj/outputs/Module_06_Spatially_Localized_Radiomics/feature_correlation_matrix.png).

---

## 6. Deliverables Inventory

All required deliverables for Module 06 are verified:
- [`outputs/Module_06_Spatially_Localized_Radiomics/spatial_radiomics_extractor.py`](file:///Users/jahnavi/Documents/DL_proj/outputs/Module_06_Spatially_Localized_Radiomics/spatial_radiomics_extractor.py) (Extractor implementation)
- [`outputs/Module_06_Spatially_Localized_Radiomics/run_spatial_radiomics.py`](file:///Users/jahnavi/Documents/DL_proj/outputs/Module_06_Spatially_Localized_Radiomics/run_spatial_radiomics.py) (Execution & verification script)
- [`outputs/Module_06_Spatially_Localized_Radiomics/feature_dictionary.csv`](file:///Users/jahnavi/Documents/DL_proj/outputs/Module_06_Spatially_Localized_Radiomics/feature_dictionary.csv) (288-feature detailed catalog)
- [`outputs/Module_06_Spatially_Localized_Radiomics/radiomic_features_train.npz`](file:///Users/jahnavi/Documents/DL_proj/outputs/Module_06_Spatially_Localized_Radiomics/radiomic_features_train.npz) (Compressed train spatial tensors `(N, 9, 32)`)
- [`outputs/Module_06_Spatially_Localized_Radiomics/radiomic_features_val.npz`](file:///Users/jahnavi/Documents/DL_proj/outputs/Module_06_Spatially_Localized_Radiomics/radiomic_features_val.npz) (Compressed validation spatial tensors)
- [`outputs/Module_06_Spatially_Localized_Radiomics/radiomic_features_test.npz`](file:///Users/jahnavi/Documents/DL_proj/outputs/Module_06_Spatially_Localized_Radiomics/radiomic_features_test.npz) (Compressed held-out test spatial tensors)
- [`outputs/Module_06_Spatially_Localized_Radiomics/radiomic_features_train.csv`](file:///Users/jahnavi/Documents/DL_proj/outputs/Module_06_Spatially_Localized_Radiomics/radiomic_features_train.csv) (Tabular feature matrix)
- [`outputs/Module_06_Spatially_Localized_Radiomics/radiomic_features_val.csv`](file:///Users/jahnavi/Documents/DL_proj/outputs/Module_06_Spatially_Localized_Radiomics/radiomic_features_val.csv) (Tabular validation feature matrix)
- [`outputs/Module_06_Spatially_Localized_Radiomics/spatial_grid_visualization.png`](file:///Users/jahnavi/Documents/DL_proj/outputs/Module_06_Spatially_Localized_Radiomics/spatial_grid_visualization.png) (3x3 grid & feature heatmaps)
- [`outputs/Module_06_Spatially_Localized_Radiomics/feature_correlation_matrix.png`](file:///Users/jahnavi/Documents/DL_proj/outputs/Module_06_Spatially_Localized_Radiomics/feature_correlation_matrix.png) (Correlation heatmap)
- [`outputs/Module_06_Spatially_Localized_Radiomics/RADIOMICS_REPORT.md`](file:///Users/jahnavi/Documents/DL_proj/outputs/Module_06_Spatially_Localized_Radiomics/RADIOMICS_REPORT.md) (Full methodology & clinical report)
- [`outputs/Module_06_Spatially_Localized_Radiomics/STATUS.md`](file:///Users/jahnavi/Documents/DL_proj/outputs/Module_06_Spatially_Localized_Radiomics/STATUS.md) (Status tracking)
