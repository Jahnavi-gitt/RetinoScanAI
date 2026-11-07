# Module 10 Status: Integrated Proposed Model

**Status**: Completed  
**Last Updated**: October 26, 2026  
**Architecture**: Spatial Visual-Radiomic CORAL Network (Swin-Tiny + 3x3 Radiomics + Cross-Attention + CORAL Head)  
**Total Parameters**: 32,896,136 (32.90M)  
**Primary Metric (Val QWK)**: 0.9007  
**Secondary Metric (Val Macro-F1)**: 0.2600  
**Secondary Metric (Val Accuracy)**: 0.4286  
**Secondary Metric (Val Balanced Acc)**: 0.4000  
**Secondary Metric (Val MAE)**: 0.5714  
**Catastrophic Distant Errors (Delta >= 2)**: 0.0%  

## Deliverables Checklist
- [x] Full Integrated Model Architecture: `src/models/integrated_model.py` & `outputs/Module_10_Integrated_Proposed_Model/integrated_model.py`
- [x] Multi-Modal Execution Script: `outputs/Module_10_Integrated_Proposed_Model/run_integrated_model.py`
- [x] Dual-Head Verification Tests: Passed on (B, 4) rank logits + (B, 5) aux logits
- [x] Training Configuration: `outputs/Module_10_Integrated_Proposed_Model/config.yaml`
- [x] Training History Log: `outputs/Module_10_Integrated_Proposed_Model/training_history.csv`
- [x] Validation Metrics: `outputs/Module_10_Integrated_Proposed_Model/metrics.json`
- [x] Classification Report: `outputs/Module_10_Integrated_Proposed_Model/classification_report.csv`
- [x] Confusion Matrix Heatmap: `outputs/Module_10_Integrated_Proposed_Model/confusion_matrix.png`
- [x] Per-Sample Prediction Table: `outputs/Module_10_Integrated_Proposed_Model/predictions.csv`
- [x] Architecture Flow Diagram: `outputs/Module_10_Integrated_Proposed_Model/model_architecture.png`
- [x] Best Model Weights Checkpoint: `outputs/Module_10_Integrated_Proposed_Model/best_model.pth`
- [x] Execution Console Log: `outputs/Module_10_Integrated_Proposed_Model/run_log.txt`
- [x] Comprehensive Architecture Report: `outputs/Module_10_Integrated_Proposed_Model/INTEGRATED_MODEL_REPORT.md`
- [x] Status File: `outputs/Module_10_Integrated_Proposed_Model/STATUS.md`
