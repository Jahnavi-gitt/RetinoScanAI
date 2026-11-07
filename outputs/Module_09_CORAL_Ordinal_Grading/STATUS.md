# Module 09 Status: CORAL Ordinal Grading

**Status**: Completed  
**Last Updated**: October 26, 2026  
**Primary Metric (Val QWK)**: 0.8993  
**Secondary Metric (Val Macro-F1)**: 0.2600  
**Secondary Metric (Val Accuracy)**: 0.4286  
**Secondary Metric (Val Balanced Acc)**: 0.4000  
**Secondary Metric (Val MAE)**: 0.5714  
**Validation CORAL Loss**: 5.1952  

## Deliverables Checklist
- [x] CORAL Layer and Loss Implementation: `src/models/coral.py` & `outputs/Module_09_CORAL_Ordinal_Grading/coral_layer.py`
- [x] CORAL Execution Script: `outputs/Module_09_CORAL_Ordinal_Grading/run_coral_grading.py`
- [x] Forward & Backward Verification Tests: Passed on (B, 4) rank logits
- [x] Training Configuration: `outputs/Module_09_CORAL_Ordinal_Grading/config.yaml`
- [x] Training History Log: `outputs/Module_09_CORAL_Ordinal_Grading/training_history.csv`
- [x] Validation Metrics: `outputs/Module_09_CORAL_Ordinal_Grading/metrics.json`
- [x] Classification Report: `outputs/Module_09_CORAL_Ordinal_Grading/classification_report.csv`
- [x] Confusion Matrix Heatmap: `outputs/Module_09_CORAL_Ordinal_Grading/confusion_matrix.png`
- [x] Per-Sample Prediction Table: `outputs/Module_09_CORAL_Ordinal_Grading/predictions.csv`
- [x] Monotonic Ordinal Probability Curves: `outputs/Module_09_CORAL_Ordinal_Grading/coral_rank_curves.png`
- [x] Best Model Weights Checkpoint: `outputs/Module_09_CORAL_Ordinal_Grading/best_model.pth`
- [x] Execution Console Log: `outputs/Module_09_CORAL_Ordinal_Grading/run_log.txt`
- [x] Methodology & Clinical Report: `outputs/Module_09_CORAL_Ordinal_Grading/CORAL_REPORT.md`
- [x] Status File: `outputs/Module_09_CORAL_Ordinal_Grading/STATUS.md`
