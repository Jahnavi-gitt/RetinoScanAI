# Module 03 Status: Baseline Swin-Tiny

**Status**: Completed  
**Last Updated**: October 26, 2026  
**Primary Metric (Val QWK)**: 0.9630  
**Secondary Metric (Val Macro-F1)**: 0.8667  
**Secondary Metric (Val Accuracy)**: 0.8571  
**Secondary Metric (Val Balanced Acc)**: 0.9000  
**Secondary Metric (Val MAE)**: 0.1429  

## Deliverables Checklist
- [x] Model Architecture Implementation: `outputs/Module_03_Baseline_Swin_Tiny/swin_tiny.py` (`src/models/swin.py`)
- [x] Pretrained Weights: Loaded from torchvision cache `swin_t-704ceda3.pth` (108 MB)
- [x] Forward and Backward Verification Tests: Passed (`(2, 5)` logits, `(2, 768, 7, 7)` spatial features)
- [x] Training Configuration: `outputs/Module_03_Baseline_Swin_Tiny/config.yaml`
- [x] Training History Log: `outputs/Module_03_Baseline_Swin_Tiny/training_history.csv`
- [x] Validation Metrics: `outputs/Module_03_Baseline_Swin_Tiny/metrics.json`
- [x] Per-Class Classification Report: `outputs/Module_03_Baseline_Swin_Tiny/classification_report.csv`
- [x] Confusion Matrix Heatmap: `outputs/Module_03_Baseline_Swin_Tiny/confusion_matrix.png`
- [x] Per-Sample Prediction Table: `outputs/Module_03_Baseline_Swin_Tiny/predictions.csv`
- [x] Best Model Weights Checkpoint: `outputs/Module_03_Baseline_Swin_Tiny/best_model.pth`
- [x] Execution Console Log: `outputs/Module_03_Baseline_Swin_Tiny/run_log.txt`
- [x] Baseline Report: `outputs/Module_03_Baseline_Swin_Tiny/BASELINE_REPORT.md`
- [x] Status File: `outputs/Module_03_Baseline_Swin_Tiny/STATUS.md`
