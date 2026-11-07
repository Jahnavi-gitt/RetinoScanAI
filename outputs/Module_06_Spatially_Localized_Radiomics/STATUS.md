# Module 06 Status: Spatially Localized Radiomics

**Status**: Completed  
**Last Updated**: October 26, 2026  
**Grid Size**: 3x3 (9 Anatomical Sub-Regions)  
**Patch Features**: 32 handcrafted descriptors  
**Total Features per Image**: 288 descriptors ($9 \times 32$)  
**Spatial Tensor Shape**: `(N, 9, 32)`  
**Fitted Normalizer**: Fitted strictly on training set (zero test leakage)  

## Deliverables Checklist
- [x] Radiomics Extractor Implementation: `src/features/radiomics.py` & `outputs/Module_06_Spatially_Localized_Radiomics/spatial_radiomics_extractor.py`
- [x] Extraction Execution Script: `outputs/Module_06_Spatially_Localized_Radiomics/run_spatial_radiomics.py`
- [x] Forward Verification Test: Passed shape `(9, 32)` on synthetic image
- [x] Feature Dictionary: `outputs/Module_06_Spatially_Localized_Radiomics/feature_dictionary.csv` (288 features categorized)
- [x] Spatial Tensor NPZ Archives:
  - `radiomic_features_train.npz` (`(35, 9, 32)`)
  - `radiomic_features_val.npz` (`(7, 9, 32)`)
  - `radiomic_features_test.npz` (`(8, 9, 32)`)
- [x] Tabular CSV Datasets:
  - `radiomic_features_train.csv`
  - `radiomic_features_val.csv`
- [x] Visualizations:
  - `spatial_grid_visualization.png` (3x3 anatomical grid overlay and feature heatmaps)
  - `feature_correlation_matrix.png` (Cross-category correlation heatmap)
- [x] Methodology & Clinical Report: `outputs/Module_06_Spatially_Localized_Radiomics/RADIOMICS_REPORT.md`
- [x] Status File: `outputs/Module_06_Spatially_Localized_Radiomics/STATUS.md`
