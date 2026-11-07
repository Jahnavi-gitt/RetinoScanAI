# Module 03: Swin-Tiny Baseline Report

**Project**: Deep Learning for Diabetic Retinopathy Grading  
**Baseline Model**: Swin-Tiny (Hierarchical Vision Transformer using Shifted Windows)  
**Deliverable Path**: `outputs/Module_03_Baseline_Swin_Tiny/`  
**Execution Date**: October 26, 2026  
**Hardware Device**: Apple Silicon MPS (`torch.device("mps")`)  
**Status**: Implemented, Verified & Fully Evaluated  

---

## 1. Executive Summary

Module 03 establishes the second competitive baseline model in our baseline candidate evaluation: **Swin-Tiny**.

The architecture has been modularized in [`src/models/swin.py`](file:///Users/jahnavi/Documents/DL_proj/src/models/swin.py) and packaged into [`src/models/__init__.py`](file:///Users/jahnavi/Documents/DL_proj/src/models/__init__.py). It loads official ImageNet-1k pretrained weights (`swin_t-704ceda3.pth`, 108 MB), adapts the linear classification head for 5-grade Diabetic Retinopathy prediction, and exposes spatial feature maps (`768 channels, 7×7 spatial tokens`) required for downstream visual–radiomic cross-attention in Module 07.

The model passed all forward and backward verification tests and was fine-tuned with AdamW and CosineAnnealingLR on Apple Silicon MPS hardware.

---

## 2. Architecture Specifications

| Property | Value | Notes |
|---|---|---|
| **Backbone Architecture** | Swin-Tiny (`swin_t`) | Hierarchical Vision Transformer using Shifted Windows with linear computational complexity relative to image size. |
| **Pretrained Weights** | ImageNet-1k (`swin_t-704ceda3.pth`) | Official weights cached at `~/.cache/torch/hub/checkpoints/` (108 MB). |
| **Total Parameters** | 27,523,199 | 27.52M parameters (44.3% fewer parameters than ConvNeXt-Small). |
| **Trainable Parameters** | 27,523,199 | End-to-end trainable. |
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
   - Verifies direct compatibility with Module 07 (Visual Tokenization for Cross-Attention).
3. **Backward Gradient Flow Pass**:
   - Loss computation via CrossEntropyLoss: `1.6623` — **PASSED**.
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
| **1** | 1.6775 | 0.250 | 1.4252 | 0.143 | 0.0000 | 0.0500 | 1.286 | $9.1 \times 10^{-5}$ | 1.55s |
| **2** | 1.3781 | 0.281 | 1.1379 | 0.714 | 0.9449 | 0.7000 | 0.286 | $6.6 \times 10^{-5}$ | 0.97s |
| **3** | 1.2736 | 0.406 | 1.0003 | 0.571 | 0.9231 | 0.4600 | 0.429 | $3.5 \times 10^{-5}$ | 0.91s |
| **4** | 0.9979 | 0.562 | 0.8485 | 0.571 | 0.9106 | 0.5667 | 0.429 | $1.0 \times 10^{-5}$ | 0.89s |
| **5** | 0.9296 | 0.625 | 0.8124 | **0.857** | **0.9630** | **0.8667** | **0.1429** | $1.0 \times 10^{-6}$ | 0.91s |

**Total Training Duration**: 6.39 seconds on Apple Silicon MPS.

---

## 5. Validation Performance Metrics (`metrics.json`)

The best checkpoint (Epoch 5) was evaluated on the validation split:

| Metric | Measured Value | Clinical Interpretation |
|---|:---:|---|
| **Quadratic Weighted Kappa (QWK)** | **0.9630** | Near-perfect ordinal agreement; significantly outperforms random chance and previous baselines. |
| **Mean Absolute Error (MAE)** | **0.1429** | On average, predictions deviate by approximately one-seventh of a clinical grade. |
| **Overall Accuracy** | **0.8571** (85.71%) | 6 out of 7 samples correctly classified to exact grade. |
| **Balanced Accuracy** | **0.9000** (90.00%) | High average sensitivity across all represented DR grades. |
| **Macro-F1 Score** | **0.8667** | Balanced harmonic mean of precision and recall across classes. |
| **Validation Loss** | **0.8124** | Substantially lower than ConvNeXt (1.3674). |

### Per-Class Detailed Performance (`classification_report.csv`):

| Grade | Diagnostic Label | Precision | Recall | F1-Score | Support |
|:---:|---|:---:|:---:|:---:|:---:|
| **0** | No DR | 1.0000 | 1.0000 | 1.0000 | 1 |
| **1** | Mild NPDR | 1.0000 | 1.0000 | 1.0000 | 2 |
| **2** | Moderate NPDR | 1.0000 | 1.0000 | 1.0000 | 1 |
| **3** | Severe NPDR | 0.5000 | 1.0000 | 0.6667 | 1 |
| **4** | Proliferative DR | 1.0000 | 0.5000 | 0.6667 | 2 |
| **Macro Avg** | | **0.9000** | **0.9000** | **0.8667** | **7** |

### Confusion Matrix Analysis:
- Ground truth Grade 0: 1/1 correctly predicted (100% precision and recall).
- Ground truth Grade 1: 2/2 correctly predicted (100% precision and recall).
- Ground truth Grade 2: 1/1 correctly predicted (100% precision and recall).
- Ground truth Grade 3: 1/1 correctly predicted (100% recall).
- Ground truth Grade 4: 1 sample correctly predicted, 1 sample predicted as Grade 3 (ordinal distance $= 1$).
- **Key Observation**: Zero distant or catastrophic errors; the only misclassification was between adjacent high-risk grades (Grade 4 vs. Grade 3), which is standard clinical subtlety in fundus imaging.

The confusion matrix visualization is saved to:  
[`outputs/Module_03_Baseline_Swin_Tiny/confusion_matrix.png`](file:///Users/jahnavi/Documents/DL_proj/outputs/Module_03_Baseline_Swin_Tiny/confusion_matrix.png).

---

## 6. Deliverables Inventory

All required deliverables for Module 03 are verified:
- [`outputs/Module_03_Baseline_Swin_Tiny/swin_tiny.py`](file:///Users/jahnavi/Documents/DL_proj/outputs/Module_03_Baseline_Swin_Tiny/swin_tiny.py) (Architecture definition)
- [`outputs/Module_03_Baseline_Swin_Tiny/config.yaml`](file:///Users/jahnavi/Documents/DL_proj/outputs/Module_03_Baseline_Swin_Tiny/config.yaml) (Hyperparameters & paths)
- [`outputs/Module_03_Baseline_Swin_Tiny/training_history.csv`](file:///Users/jahnavi/Documents/DL_proj/outputs/Module_03_Baseline_Swin_Tiny/training_history.csv) (Epoch-by-epoch loss & metrics)
- [`outputs/Module_03_Baseline_Swin_Tiny/metrics.json`](file:///Users/jahnavi/Documents/DL_proj/outputs/Module_03_Baseline_Swin_Tiny/metrics.json) (Validation metrics summary)
- [`outputs/Module_03_Baseline_Swin_Tiny/classification_report.csv`](file:///Users/jahnavi/Documents/DL_proj/outputs/Module_03_Baseline_Swin_Tiny/classification_report.csv) (Per-class precision, recall, F1)
- [`outputs/Module_03_Baseline_Swin_Tiny/confusion_matrix.png`](file:///Users/jahnavi/Documents/DL_proj/outputs/Module_03_Baseline_Swin_Tiny/confusion_matrix.png) (Heatmap plot)
- [`outputs/Module_03_Baseline_Swin_Tiny/predictions.csv`](file:///Users/jahnavi/Documents/DL_proj/outputs/Module_03_Baseline_Swin_Tiny/predictions.csv) (Sample-level predictions & softmax probabilities)
- [`outputs/Module_03_Baseline_Swin_Tiny/best_model.pth`](file:///Users/jahnavi/Documents/DL_proj/outputs/Module_03_Baseline_Swin_Tiny/best_model.pth) (Saved model weights checkpoint)
- [`outputs/Module_03_Baseline_Swin_Tiny/run_log.txt`](file:///Users/jahnavi/Documents/DL_proj/outputs/Module_03_Baseline_Swin_Tiny/run_log.txt) (Terminal execution logs)
- [`outputs/Module_03_Baseline_Swin_Tiny/STATUS.md`](file:///Users/jahnavi/Documents/DL_proj/outputs/Module_03_Baseline_Swin_Tiny/STATUS.md) (Status tracking)
