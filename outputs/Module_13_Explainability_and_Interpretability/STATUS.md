# Module 13: Explainability and Interpretability Status

- **Status**: Completed
- **Date**: 2026-11-02
- **Device**: mps
- **Dataset**: Real OIA-DDR Clinical Benchmark (10,024 train, 1,251 test)
- **Model**: Proposed Multimodal Reference M0 (Swin-Tiny + 9 Spatial Radiomics + Cross-Attention + CORAL)
- **Visual Saliency Verified**: Grad-CAM heatmaps for Swin-Tiny across all 5 DR stages.
- **Cross-Attention Grounding Verified**: 49x9 spatial attention matrices & 3x3 retinal grid distributions.
- **Radiomic Attribution Verified**: Feature importance calculated across all 288 IBSI descriptors and 4 categories.

### Deliverables Produced:
- `gradcam_overlays.png`: 5-grade Grad-CAM fundus overlays.
- `cross_attention_alignment_maps.png`: 49x9 cross-attention matrices & anatomical 3x3 profiles.
- `radiomic_feature_importance.png`: 2-panel category contribution & top-15 descriptors chart.
- `feature_importance.csv`: Ranked importance for all 288 descriptors.
- `interpretability_summary.csv`: Summary metrics of spatial attention allocation.
- `EXPLAINABILITY_REPORT.md`: Comprehensive scientific interpretability report.
- `STATUS.md`: Verification summary log.
