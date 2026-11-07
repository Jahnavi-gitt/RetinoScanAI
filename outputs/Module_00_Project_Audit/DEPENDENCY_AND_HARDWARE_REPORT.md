# Dependency and Hardware Execution Report

**Date & Time**: 2026-10-25 15:55 IST  
**Host Machine**: Jahnavi's MacBook Air (Apple Silicon)  
**Project**: Deep Learning for Diabetic Retinopathy (DR) Grading  

---

## 1. Hardware Specifications

| Component | Detected Specification | Evaluation & Constraints |
|---|---|---|
| **SoC / Processor** | Apple M5 (ARM64 architecture) | High single-thread and multi-core efficiency with unified memory architecture. |
| **System Memory (RAM)** | 16 GB Unified RAM (`hw.memsize: 17179869184`) | Sufficient for medium-scale vision models (batch size 8–16 at 224×224 resolution). |
| **GPU / Accelerator** | Apple Silicon integrated Metal Performance Shaders (MPS) | PyTorch supports `device = torch.device("mps")` on macOS for hardware acceleration. No NVIDIA CUDA available (`nvidia-smi` not found). |
| **Available Storage** | 123 GiB free (`/System/Volumes/Data`) | Ample local storage for code, cached radiomics, manifests, and model checkpoints. |

---

## 2. Software & Python Environment Audit

### 2.1 Current System State
- **Operating System**: macOS Darwin 25.6.0 (Apple ARM64)
- **System Python**: `/usr/bin/python3` (Python 3.9.6)
- **Active User Virtual Environment**: `~/.venv` (Python 3.9.6)
- **Installed Packages in `~/.venv`**:
  - `numpy 2.0.2`
  - `opencv-python 5.0.0.93`
  - `pillow 11.3.0`
  - `pytest 8.4.2`
  - `pytesseract 0.3.13`
  - `pip 21.2.4`

### 2.2 Critical Missing Packages
The original notebook ran in a Kaggle GPU environment with pre-installed PyTorch, TorchVision, and Timm. The following packages must be installed in the project environment:

| Package | Minimum Recommended Version | Purpose |
|---|---|---|
| `torch` | `>= 2.2.0` | Core deep learning framework with Apple Silicon MPS acceleration. |
| `torchvision` | `>= 0.17.0` | Vision transformations, dataset utilities, and image operations. |
| `timm` | `>= 0.9.16` | Model backbones (`convnext_small`, `swin_tiny_patch4_window7_224`, `efficientnet_b3`). |
| `scikit-learn` | `>= 1.3.0` | Metrics (Quadratic Weighted Kappa `cohen_kappa_score`, Macro-F1, confusion matrix, train/test split). |
| `pandas` | `>= 2.0.0` | Manifest handling, CSV logging, class distribution analysis. |
| `matplotlib` & `seaborn` | `>= 3.7.0` | Confusion matrices, loss curves, ablation visualizations. |
| `scikit-image` | `>= 0.21.0` | Morphological filtering, vessel skeletonization, LBP descriptors. |
| `pyradiomics` & `SimpleITK` | Latest compatible | Standardized radiomic texture and shape extraction. |
| `pyyaml` | `>= 6.0` | Configuration parsing (`config.yaml`). |
| `tqdm` | `>= 4.65.0` | Progress reporting for training and feature extraction loops. |
| `streamlit` | `>= 1.30.0` | Interactive web demonstration (Module 14). |

---

## 3. Computational Budget and Training Constraints

### 3.1 Memory & Batch Size Strategy
- On an Apple M5 with 16 GB unified memory, training deep backbones (ConvNeXt-Small, Swin-Tiny, EfficientNet-B3) alongside feature caching requires careful batch management.
- **Recommended Image Resolution**: 224×224 pixels (consistent across all baseline backbones to ensure fair competition).
- **Recommended Batch Size**: 8 to 16.
- **DataLoader Workers**: 2 (to prevent macOS MPS memory over-allocation or thread thrashing).
- **Precision**: PyTorch `torch.float32` (or `torch.amp.autocast(device_type="mps")` if supported in the installed PyTorch version).

### 3.2 Verification-Test First Protocol (Rule D / Technical Requirements)
Before executing any multi-epoch training:
1. Run a 1-batch synthetic forward pass to verify input-output tensor shapes.
2. Run a 1-batch backward pass to verify gradient propagation and loss computation.
3. Test feature caching on 5 sample images to verify deterministic numerical values.
4. Scale up to the actual train split only after verification tests pass.
