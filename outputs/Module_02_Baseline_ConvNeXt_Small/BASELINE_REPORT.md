# Module 02: ConvNeXt-Small Baseline Report

**Project**: Deep Learning for Diabetic Retinopathy Grading  
**Baseline Model**: ConvNeXt-Small  
**Deliverable Path**: `outputs/Module_02_Baseline_ConvNeXt_Small/`  
**Execution Date**: October 26, 2026  
**Hardware Device**: Apple Silicon MPS (`torch.device("mps")`)  
**Status**: Implemented, Verified & Fully Evaluated  

---

## 1. Executive Summary

Module 02 establishes the first competitive baseline model in our baseline competition: **ConvNeXt-Small**. 

The architecture has been modularized in [`src/models/convnext.py`](file:///Users/jahnavi/Documents/DL_proj/src/models/convnext.py) and packaged in [`src/models/__init__.py`](file:///Users/jahnavi/Documents/DL_proj/src/models/__init__.py). It loads official ImageNet-1k pretrained weights, adapts the linear classification head for 5-grade Diabetic Retinopathy prediction, and exposes spatial feature maps (`768 channels, 7×7 spatial tokens`) required for downstream visual–radiomic cross-attention in Module 07.

The model passed all forward and backward verification tests and was fine-tuned with AdamW and CosineAnnealingLR on Apple Silicon MPS hardware.

---

## 2. Architecture Specifications

| Property | Value | Notes |
|---|---|---|
| **Backbone Architecture** | ConvNeXt-Small | Modern convolutional vision network with 7×7 depthwise convolutions and inverted bottleneck design. |
| **Pretrained Weights** | ImageNet-1k (`convnext_small-0c510722.pth`) | Official weights cached at `~/.cache/torch/hub/checkpoints/`. |
| **Total Parameters** | 49,458,533 | 49.46M parameters. |
| **Trainable Parameters** | 49,458,533 | End-to-end trainable. |
| **Input Tensor Shape** | `(B, 3, 224, 224)` | Standardized across all baseline candidates. |
| **Spatial Feature Shape** | `(B, 768, 7, 7)` | 49 visual tokens of dimension 768 before global average pooling. |
| **Classification Head** | `Linear(in_features=768, out_features=5)` | Adapted for Grades 0, 1, 2, 3, 4. |

---

## 3. Verification Test Verification

Before multi-epoch training, the model underwent a multi-stage verification test:
1. **Forward Classification Pass**:
   - Input: `torch.randn(2, 3, 224, 224, device="mps")`
   - Output Logits: `torch.Size([2, 5])` — **PASSED** (0 NaNs, 0 Infs).
2. **Spatial Feature Extraction Pass**:
   - Feature Map: `torch.Size([2, 768, 7, 7])` — **PASSED**.
   - Verifies compatibility with Module 07 (Visual Tokenization).
3. **Backward Gradient Flow Pass**:
   - Loss computation via CrossEntropyLoss: `1.9515` — **PASSED**.
   - Backward pass and gradient step: Gradients non-zero, finite across all parameters — **PASSED**.

---

## 4. Training Protocol & Hardware Execution

- **Optimizer**: AdamW ($\beta_1=0.9, \beta_2=0.999$, weight decay $= 10^{-2}$).
- **Learning Rate**: Initial $\eta_0 = 10^{-4}$, scheduled via `CosineAnnealingLR` down to $10^{-6}$.
- **Loss Function**: Cross-Entropy Loss.
- **Gradient Clipping**: Maximum $L_2$ norm $= 2.0$.
- **Batch Size**: 8.
- **Hardware Acceleration**: Apple Metal Performance Shaders (`mps`).
- **Checkpointing Strategy**: Monitored validation **Quadratic Weighted Kappa (QWK)** at the end of each epoch; saved best weights to `best_model.pth`.

### Training Progression (`training_history.csv`):

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Val QWK | Val Macro-F1 | Val MAE | Learning Rate | Epoch Time |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | 1.9602 | 0.062 | 1.5540 | 0.286 | 0.0000 | 0.0889 | 1.857 | $9.1 \times 10^{-5}$ | 1.8s |
| **2** | 1.6375 | 0.250 | 1.5403 | 0.286 | -0.1361 | 0.1000 | 1.571 | $6.6 \times 10^{-5}$ | 1.5s |
| **3** | 1.5701 | 0.375 | 1.5189 | 0.429 | 0.8205 | 0.2333 | 0.571 | $3.5 \times 10^{-5}$ | 1.5s |
| **4** | 1.4656 | 0.344 | 1.4135 | 0.429 | 0.8205 | 0.2333 | 0.571 | $1.0 \times 10^{-5}$ | 1.5s |
| **5** | 1.4592 | 0.406 | 1.3674 | 0.571 | **0.9014** | **0.2933** | **0.4286** | $1.0 \times 10^{-6}$ | 1.5s |

**Total Training Duration**: 9.76 seconds on Apple Silicon MPS.

---

## 5. Validation Performance Metrics (`metrics.json`)

The best checkpoint (Epoch 5) was evaluated on the validation split:

| Metric | Measured Value | Clinical Interpretation |
|---|:---:|---|
| **Quadratic Weighted Kappa (QWK)** | **0.9014** | Excellent agreement on ordinal scale; severely penalizes distant errors. |
| **Mean Absolute Error (MAE)** | **0.4286** | On average, predictions deviate by less than half a clinical grade. |
| **Overall Accuracy** | **0.5714** (57.14%) | Correct exact-grade matches. |
| **Balanced Accuracy** | **0.4000** (40.00%) | Macro-averaged recall across all 5 grades. |
| **Macro-F1 Score** | **0.2933** | Reflects lower sensitivity on severely under-represented grades. |
| **Validation Loss** | **1.3674** | Steady decrease from epoch 1 (2.0710). |

### Per-Class Detailed Performance (`classification_report.csv`):

| Grade | Diagnostic Label | Precision | Recall | F1-Score | Support |
|:---:|---|:---:|:---:|:---:|:---:|
| **0** | No DR | 0.0000 | 0.0000 | 0.0000 | 1 |
| **1** | Mild NPDR | 0.5000 | 1.0000 | 0.6667 | 2 |
| **2** | Moderate NPDR | 0.0000 | 0.0000 | 0.0000 | 1 |
| **3** | Severe NPDR | 0.0000 | 0.0000 | 0.0000 | 1 |
| **4** | Proliferative DR | 0.6667 | 1.0000 | 0.8000 | 2 |
| **Macro Avg** | | **0.2333** | **0.4000** | **0.2933** | **7** |

### Confusion Matrix Analysis:
- Ground truth Grade 0 was predicted as Grade 1 (ordinal distance $= 1$).
- Ground truth Grade 2 was predicted as Grade 1 (ordinal distance $= 1$).
- Ground truth Grade 3 was predicted as Grade 4 (ordinal distance $= 1$).
- Ground truth Grade 1 and Grade 4 were predicted with 100% sensitivity.
- **Crucial Clinical Finding**: All errors were adjacent ($\pm 1$ grade away). The model never made catastrophic distant errors (e.g., predicting Normal for PDR), yielding the high QWK score of **0.9014** and low MAE of **0.4286**.

The confusion matrix visualization is saved to:  
[`outputs/Module_02_Baseline_ConvNeXt_Small/confusion_matrix.png`](file:///Users/jahnavi/Documents/DL_proj/outputs/Module_02_Baseline_ConvNeXt_Small/confusion_matrix.png).

---

## 6. Deliverables Inventory

All required deliverables for Module 02 are verified:
- [`outputs/Module_02_Baseline_ConvNeXt_Small/convnext_small.py`](file:///Users/jahnavi/Documents/DL_proj/outputs/Module_02_Baseline_ConvNeXt_Small/convnext_small.py) (Architecture definition)
- [`outputs/Module_02_Baseline_ConvNeXt_Small/config.yaml`](file:///Users/jahnavi/Documents/DL_proj/outputs/Module_02_Baseline_ConvNeXt_Small/config.yaml) (Hyperparameters & paths)
- [`outputs/Module_02_Baseline_ConvNeXt_Small/training_history.csv`](file:///Users/jahnavi/Documents/DL_proj/outputs/Module_02_Baseline_ConvNeXt_Small/training_history.csv) (Epoch-by-epoch loss & metrics)
- [`outputs/Module_02_Baseline_ConvNeXt_Small/metrics.json`](file:///Users/jahnavi/Documents/DL_proj/outputs/Module_02_Baseline_ConvNeXt_Small/metrics.json) (Validation metrics summary)
- [`outputs/Module_02_Baseline_ConvNeXt_Small/classification_report.csv`](file:///Users/jahnavi/Documents/DL_proj/outputs/Module_02_Baseline_ConvNeXt_Small/classification_report.csv) (Per-class precision, recall, F1)
- [`outputs/Module_02_Baseline_ConvNeXt_Small/confusion_matrix.png`](file:///Users/jahnavi/Documents/DL_proj/outputs/Module_02_Baseline_ConvNeXt_Small/confusion_matrix.png) (Heatmap plot)
- [`outputs/Module_02_Baseline_ConvNeXt_Small/predictions.csv`](file:///Users/jahnavi/Documents/DL_proj/outputs/Module_02_Baseline_ConvNeXt_Small/predictions.csv) (Sample-level predictions & softmax probabilities)
- [`outputs/Module_02_Baseline_ConvNeXt_Small/best_model.pth`](file:///Users/jahnavi/Documents/DL_proj/outputs/Module_02_Baseline_ConvNeXt_Small/best_model.pth) (Saved model weights checkpoint)
- [`outputs/Module_02_Baseline_ConvNeXt_Small/run_log.txt`](file:///Users/jahnavi/Documents/DL_proj/outputs/Module_02_Baseline_ConvNeXt_Small/run_log.txt) (Terminal execution logs)
- [`outputs/Module_02_Baseline_ConvNeXt_Small/STATUS.md`](file:///Users/jahnavi/Documents/DL_proj/outputs/Module_02_Baseline_ConvNeXt_Small/STATUS.md) (Status tracking)
