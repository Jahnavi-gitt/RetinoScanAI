# Proposed Project Architecture and Repository Structure

## 1. Design Principles
1. **Preserve Original Artifacts**: Keep `dr-ansh (11).ipynb` untouched. A full backup is maintained in `original_files/`.
2. **Modular Python Codebase (`src/`)**: Decouple data handling, model definitions, radiomics, training, evaluation, and explainability into reusable, clean modules rather than monolithic notebook scripts.
3. **Module-Scoped Outputs (`outputs/Module_XX_...`)**: Every phase and milestone has an isolated, self-contained output directory containing its configurations, logs, metrics, figures, and status reports.
4. **Configuration-Driven**: All hyperparameters, paths, and random seeds live in YAML configs under `configs/`, avoiding hard-coded machine paths.
5. **Downstream Dependency Chain**: Downstream novelty modules consume the output of Module 05 via `model_selection.json`.

---

## 2. Directory Tree Layout

```
DL_proj/
├── original_files/                                 # Intact backup of user's original work
│   └── dr-ansh (11).ipynb
├── dr-ansh (11).ipynb                              # Original working notebook (preserved)
├── configs/                                        # Central experiment configurations
│   ├── dataset_config.yaml
│   ├── baseline_convnext.yaml
│   ├── baseline_swin.yaml
│   ├── baseline_effnet.yaml
│   ├── radiomics_config.yaml
│   └── integrated_model.yaml
├── data_manifests/                                 # Reproducible data splits & audit CSVs
│   ├── train_manifest.csv
│   ├── val_manifest.csv
│   └── test_manifest.csv
├── checkpoints/                                    # Saved model weights
│   ├── convnext_small_best.pth
│   ├── swin_tiny_best.pth
│   ├── effnet_b3_best.pth
│   └── proposed_integrated_best.pth
├── logs/                                           # Runtime and training logs
├── src/                                            # Reusable core library
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py                             # PyTorch Dataset implementation for DDR/EyePACS
│   │   ├── preprocessing.py                       # CLAHE, resizing, circular masking, transforms
│   │   └── split_utils.py                         # Stratified / patient-level splitting
│   ├── models/
│   │   ├── __init__.py
│   │   ├── backbones.py                           # ConvNeXt-Small, Swin-Tiny, EfficientNet-B3
│   │   ├── coral.py                               # CORAL ordinal head, loss & rank decoding
│   │   ├── cross_attention.py                     # Spatial visual-radiomic cross-attention module
│   │   └── integrated_model.py                    # End-to-end proposed architecture
│   ├── radiomics/
│   │   ├── __init__.py
│   │   ├── extractor.py                           # 3x3 spatial grid regional descriptor extractor
│   │   └── cache_manager.py                       # Deterministic feature caching and hashing
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py                             # Standardized training/validation loop
│   │   ├── losses.py                              # FocalLoss, weighted CrossEntropy, CORAL loss
│   │   └── scheduler.py                           # Cosine annealing with warmup
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py                             # QWK, Macro-F1, balanced accuracy, MAE
│   │   └── report_generator.py                    # Confusion matrices, classification reports
│   └── explainability/
│       ├── __init__.py
│       └── gradcam.py                             # Grad-CAM implementation for backbones
├── tests/                                          # Unit tests and verification tests
│   ├── test_data_pipeline.py
│   ├── test_backbones.py
│   ├── test_coral.py
│   └── test_cross_attention.py
├── outputs/                                        # Deliverables for each module
│   ├── Module_00_Project_Audit/
│   ├── Module_01_Dataset_and_Preprocessing/
│   ├── Module_02_Baseline_ConvNeXt_Small/
│   ├── Module_03_Baseline_Swin_Tiny/
│   ├── Module_04_Baseline_EfficientNet_B3/
│   ├── Module_05_Baseline_Comparison_and_Model_Selection/
│   ├── Module_06_Local_Radiomics/
│   ├── Module_07_Spatial_Cross_Attention/
│   ├── Module_08_Class_Imbalance_Strategies/
│   ├── Module_09_CORAL_Ordinal_Grading/
│   ├── Module_10_Integrated_Proposed_Model/
│   ├── Module_11_Ablation_Studies/
│   ├── Module_12_Final_Evaluation/
│   ├── Module_13_Explainability/
│   ├── Module_14_Demo_Application/
│   └── Module_15_Reproducibility_and_Final_Documentation/
└── requirements.txt                                # Environment dependencies
```

---

## 3. Data Flow and Module Coupling

```
[Module 00: Audit]
       │
[Module 01: Dataset Validation & Preprocessing]
       │
┌──────┴──────────────────────────┐
│   Baseline Competition          │
│   ├── Module 02: ConvNeXt-Small │
│   ├── Module 03: Swin-Tiny      │
│   └── Module 04: EfficientNet-B3│
└──────┬──────────────────────────┘
       │
[Module 05: Baseline Selection] ──> generates model_selection.json
       │
       ├────────────────────────────────────────┐
       ▼                                        ▼
[Module 06: Local Radiomics]           [Selected Best Backbone]
       │                                        │
       └───────────────────┬────────────────────┘
                           ▼
              [Module 07: Spatial Cross-Attention]
                           │
              [Module 08: Class Imbalance] + [Module 09: CORAL Head]
                           │
                           ▼
              [Module 10: Integrated Proposed Model]
                           │
              [Module 11: Ablation Studies]
                           │
              [Module 12: Final Evaluation]
                           │
              [Module 13: Explainability & CAM]
                           │
              [Module 14: Demo Application (Streamlit)]
                           │
              [Module 15: Final Documentation & Delivery]
```
