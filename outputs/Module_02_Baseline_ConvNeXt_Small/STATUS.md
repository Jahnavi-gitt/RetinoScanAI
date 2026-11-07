# Module 02: Baseline ConvNeXt-Small — Status Report

**Current Status**: Completed  
**Timestamp**: 2026-10-26 22:43 IST  
**Component**: Baseline 1 / 3 (ConvNeXt-Small)  

---

## 1. Summary of Completed Deliverables

All required deliverables for Module 02 are saved in [`outputs/Module_02_Baseline_ConvNeXt_Small/`](file:///Users/jahnavi/Documents/DL_proj/outputs/Module_02_Baseline_ConvNeXt_Small/):

| Deliverable File | Description | Verification Status |
|---|---|---|
| [`convnext_small.py`](file:///Users/jahnavi/Documents/DL_proj/outputs/Module_02_Baseline_ConvNeXt_Small/convnext_small.py) | ConvNeXt-Small model implementation with spatial feature extractor and 5-class linear head | Verified (Also in `src/models/convnext.py`) |
| [`config.yaml`](file:///Users/jahnavi/Documents/DL_proj/outputs/Module_02_Baseline_ConvNeXt_Small/config.yaml) | Training hyperparameters, optimizer, scheduler, and loss configuration | Verified (Complete) |
| [`training_history.csv`](file:///Users/jahnavi/Documents/DL_proj/outputs/Module_02_Baseline_ConvNeXt_Small/training_history.csv) | Per-epoch train loss, train acc, val loss, val acc, val QWK, val Macro-F1, val MAE, and LR | Verified (5 epochs logged) |
| [`metrics.json`](file:///Users/jahnavi/Documents/DL_proj/outputs/Module_02_Baseline_ConvNeXt_Small/metrics.json) | Validation summary metrics (QWK: 0.9014, Macro-F1: 0.2933, Acc: 0.5714, MAE: 0.4286) | Verified (Machine-readable) |
| [`classification_report.csv`](file:///Users/jahnavi/Documents/DL_proj/outputs/Module_02_Baseline_ConvNeXt_Small/classification_report.csv) | Precision, recall, and F1-score for each DR grade | Verified (Per-grade metrics) |
| [`confusion_matrix.png`](file:///Users/jahnavi/Documents/DL_proj/outputs/Module_02_Baseline_ConvNeXt_Small/confusion_matrix.png) | Heatmap displaying true vs. predicted grades with counts and proportions | Verified (126 KB plot) |
| [`predictions.csv`](file:///Users/jahnavi/Documents/DL_proj/outputs/Module_02_Baseline_ConvNeXt_Small/predictions.csv) | Per-sample validation predictions, errors, and 5-class softmax probabilities | Verified (Per-sample log) |
| [`best_model.pth`](file:///Users/jahnavi/Documents/DL_proj/outputs/Module_02_Baseline_ConvNeXt_Small/best_model.pth) | Best model state checkpoint saved on validation QWK improvement | Verified (566 MB checkpoint) |
| [`run_log.txt`](file:///Users/jahnavi/Documents/DL_proj/outputs/Module_02_Baseline_ConvNeXt_Small/run_log.txt) | Complete terminal execution log with forward/backward verification tests and epoch printouts | Verified (Status code 0) |
| [`BASELINE_REPORT.md`](file:///Users/jahnavi/Documents/DL_proj/outputs/Module_02_Baseline_ConvNeXt_Small/BASELINE_REPORT.md) | In-depth engineering and clinical performance report | Verified (Complete) |

---

## 2. Key Checks & Test Results
- **Forward & Backward Verification Tests**:
  - Forward classification logits shape: `torch.Size([2, 5])` — **PASSED**
  - Spatial feature map shape: `torch.Size([2, 768, 7, 7])` — **PASSED**
  - Backward gradient computation: loss non-zero, finite gradients across all parameters — **PASSED**
- **Hardware Acceleration**: Successfully utilized Apple Silicon MPS (`torch.device("mps")`).
- **Validation Metrics**:
  - Quadratic Weighted Kappa (QWK): **0.9014**
  - Mean Absolute Error (MAE): **0.4286**
  - Overall Accuracy: **0.5714**
  - Balanced Accuracy: **0.4000**
  - Macro-F1: **0.2933**
- **Error Characteristics**: All misclassifications were adjacent errors ($\pm 1$ grade), demonstrating strong ordinal coherence.

---

## 3. Next Module
**Module 03: Baseline Swin-Tiny**  
(Pending user approval as per Rule B).
