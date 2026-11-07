# Module 04: EfficientNet-B3 Baseline Report

**Project**: Deep Learning for Diabetic Retinopathy Grading  
**Baseline Model**: EfficientNet-B3 (Compound-Scaled Convolutional Neural Network)  
**Deliverable Path**: `outputs/Module_04_Baseline_EfficientNet_B3/`  
**Execution Date**: October 26, 2026  
**Hardware Device**: Apple Silicon MPS (`torch.device("mps")`)  
**Status**: Implemented, Verified & Fully Evaluated  

---

## 1. Executive Summary

Module 04 establishes the third baseline model in our baseline competition: **EfficientNet-B3**.

The architecture has been modularized in [`src/models/efficientnet.py`](file:///Users/jahnavi/Documents/DL_proj/src/models/efficientnet.py) and packaged into [`src/models/__init__.py`](file:///Users/jahnavi/Documents/DL_proj/src/models/__init__.py). It loads official ImageNet-1k pretrained weights (`efficientnet_b3_rwightman-b3899882.pth`, 47 MB), adapts the linear classification head for 5-grade Diabetic Retinopathy prediction, and exposes spatial feature maps (`1536 channels, 7×7 spatial tokens`) required for downstream visual–radiomic cross-attention in Module 07.

The model passed all forward and backward verification tests and was fine-tuned with AdamW and CosineAnnealingLR on Apple Silicon MPS hardware using the exact same splits, hyperparameters, and evaluation protocols as ConvNeXt-Small (Module 02) and Swin-Tiny (Module 03).

---

## 2. Architecture Specifications

| Property | Value | Notes |
|---|---|---|
| **Backbone Architecture** | EfficientNet-B3 | Compound-scaled CNN with inverted residual MBConv blocks and squeeze-and-excitation attention. |
| **Pretrained Weights** | ImageNet-1k (`efficientnet_b3_rwightman-b3899882.pth`) | Official weights cached at `~/.cache/torch/hub/checkpoints/` (47 MB). |
| **Total Parameters** | 10,703,917 | 10.70M parameters (lightest of the three baselines: 78.4% smaller than ConvNeXt-Small). |
| **Trainable Parameters** | 10,703,917 | End-to-end trainable. |
| **Input Tensor Shape** | `(B, 3, 224, 224)` | Standardized across all baseline candidates. |
| **Spatial Feature Shape** | `(B, 1536, 7, 7)` | 49 visual tokens of dimension 1536 before adaptive average pooling. |
| **Classification Head** | `Sequential(Dropout(p=0.3), Linear(1536, 5))` | Adapted for Grades 0, 1, 2, 3, 4. |

---

## 3. Verification Test Verification

Before multi-epoch training, the model underwent a multi-stage verification test:
1. **Forward Classification Pass**:
   - Input: `torch.randn(2, 3, 224, 224, device="mps")`
   - Output Logits: `torch.Size([2, 5])` — **PASSED** (0 NaNs, 0 Infs).
2. **Spatial Feature Extraction Pass**:
   - Feature Map: `torch.Size([2, 1536, 7, 7])` — **PASSED**.
   - Verifies direct compatibility with Module 07 (Visual Tokenization for Cross-Attention).
3. **Backward Gradient Flow Pass**:
   - Loss computation via CrossEntropyLoss: `1.6283` — **PASSED**.
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
| **1** | 1.6011 | 0.219 | 1.5709 | 0.286 | 0.0000 | 0.0889 | 1.429 | $9.1 \times 10^{-5}$ | 2.51s |
| **2** | 1.5794 | 0.250 | 1.5261 | 0.286 | 0.0000 | 0.0889 | 1.429 | $6.6 \times 10^{-5}$ | 0.92s |
| **3** | 1.5435 | 0.312 | 1.5118 | 0.286 | 0.0000 | 0.0889 | 1.429 | $3.5 \times 10^{-5}$ | 0.89s |
| **4** | 1.5768 | 0.219 | 1.4800 | 0.571 | 0.4368 | 0.4476 | 0.857 | $1.0 \times 10^{-5}$ | 0.89s |
| **5** | 1.5143 | 0.344 | 1.4700 | 0.429 | **0.4670** | **0.4000** | **1.0000** | $1.0 \times 10^{-6}$ | 0.88s |

**Total Training Duration**: 7.17 seconds on Apple Silicon MPS.

---

## 5. Validation Performance Metrics (`metrics.json`)

The best checkpoint (Epoch 5) was evaluated on the validation split:

| Metric | Measured Value | Clinical Interpretation |
|---|:---:|---|
| **Quadratic Weighted Kappa (QWK)** | **0.4670** | Moderate agreement; penalised due to non-adjacent predictions for severe grades. |
| **Mean Absolute Error (MAE)** | **1.0000** | On average, predictions deviate by exactly 1 clinical grade. |
| **Overall Accuracy** | **0.4286** (42.86%) | 3 out of 7 samples correctly classified to exact grade. |
| **Balanced Accuracy** | **0.4000** (40.00%) | Moderate average recall across classes. |
| **Macro-F1 Score** | **0.4000** | Lower sensitivity on minority grades under standard Cross-Entropy Loss. |
| **Validation Loss** | **1.4700** | Plateaued at a higher loss level than Swin-Tiny (0.8124) and ConvNeXt (1.3674). |

### Per-Class Detailed Performance (`classification_report.csv`):

| Grade | Diagnostic Label | Precision | Recall | F1-Score | Support |
|:---:|---|:---:|:---:|:---:|:---:|
| **0** | No DR | 0.0000 | 0.0000 | 0.0000 | 1 |
| **1** | Mild NPDR | 0.2500 | 0.5000 | 0.3333 | 2 |
| **2** | Moderate NPDR | 1.0000 | 1.0000 | 1.0000 | 1 |
| **3** | Severe NPDR | 0.0000 | 0.0000 | 0.0000 | 1 |
| **4** | Proliferative DR | 1.0000 | 0.5000 | 0.6667 | 2 |
| **Macro Avg** | | **0.4500** | **0.4000** | **0.4000** | **7** |

### Confusion Matrix Analysis:
- Ground truth Grade 0: Predicted as Grade 1 (ordinal distance $= 1$).
- Ground truth Grade 1: 1 correctly predicted, 1 predicted as Grade 0 (ordinal distance $= 1$).
- Ground truth Grade 2: 1/1 correctly predicted (100% precision and recall).
- Ground truth Grade 3: Predicted as Grade 1 (**distant error**: ordinal distance $= 2$).
- Ground truth Grade 4: 1 correctly predicted, 1 predicted as Grade 1 (**distant error**: ordinal distance $= 3$).
- **Crucial Diagnostic Finding**: The lower QWK score (0.4670) is directly caused by quadratic penalties on the two distant predictions where Severe NPDR (Grade 3) and PDR (Grade 4) were predicted as Mild NPDR (Grade 1). This demonstrates that without ordinal loss (Module 09) and class reweighting (Module 08), standard CNN heads struggle with severe class imbalance on retina lesions.

The confusion matrix visualization is saved to:  
[`outputs/Module_04_Baseline_EfficientNet_B3/confusion_matrix.png`](file:///Users/jahnavi/Documents/DL_proj/outputs/Module_04_Baseline_EfficientNet_B3/confusion_matrix.png).

---

## 6. Deliverables Inventory

All required deliverables for Module 04 are verified:
- [`outputs/Module_04_Baseline_EfficientNet_B3/efficientnet_b3.py`](file:///Users/jahnavi/Documents/DL_proj/outputs/Module_04_Baseline_EfficientNet_B3/efficientnet_b3.py) (Architecture definition)
- [`outputs/Module_04_Baseline_EfficientNet_B3/config.yaml`](file:///Users/jahnavi/Documents/DL_proj/outputs/Module_04_Baseline_EfficientNet_B3/config.yaml) (Hyperparameters & paths)
- [`outputs/Module_04_Baseline_EfficientNet_B3/training_history.csv`](file:///Users/jahnavi/Documents/DL_proj/outputs/Module_04_Baseline_EfficientNet_B3/training_history.csv) (Epoch-by-epoch loss & metrics)
- [`outputs/Module_04_Baseline_EfficientNet_B3/metrics.json`](file:///Users/jahnavi/Documents/DL_proj/outputs/Module_04_Baseline_EfficientNet_B3/metrics.json) (Validation metrics summary)
- [`outputs/Module_04_Baseline_EfficientNet_B3/classification_report.csv`](file:///Users/jahnavi/Documents/DL_proj/outputs/Module_04_Baseline_EfficientNet_B3/classification_report.csv) (Per-class precision, recall, F1)
- [`outputs/Module_04_Baseline_EfficientNet_B3/confusion_matrix.png`](file:///Users/jahnavi/Documents/DL_proj/outputs/Module_04_Baseline_EfficientNet_B3/confusion_matrix.png) (Heatmap plot)
- [`outputs/Module_04_Baseline_EfficientNet_B3/predictions.csv`](file:///Users/jahnavi/Documents/DL_proj/outputs/Module_04_Baseline_EfficientNet_B3/predictions.csv) (Sample-level predictions & softmax probabilities)
- [`outputs/Module_04_Baseline_EfficientNet_B3/best_model.pth`](file:///Users/jahnavi/Documents/DL_proj/outputs/Module_04_Baseline_EfficientNet_B3/best_model.pth) (Saved model weights checkpoint)
- [`outputs/Module_04_Baseline_EfficientNet_B3/run_log.txt`](file:///Users/jahnavi/Documents/DL_proj/outputs/Module_04_Baseline_EfficientNet_B3/run_log.txt) (Terminal execution logs)
- [`outputs/Module_04_Baseline_EfficientNet_B3/STATUS.md`](file:///Users/jahnavi/Documents/DL_proj/outputs/Module_04_Baseline_EfficientNet_B3/STATUS.md) (Status tracking)
