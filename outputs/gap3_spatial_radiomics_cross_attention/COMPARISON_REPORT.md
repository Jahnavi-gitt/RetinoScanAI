# Gap 3 Model Evaluation & Comparison Report
## Spatial Radiomics Cross-Attention vs. Original Swin-Tiny + CORAL

**Dataset**: DDR 80/10/10 Split (Held-Out Test Set: 1,251 Images)  
**Hardware**: Apple Silicon MPS  

### 1. Overall Test Metrics
| Metric | Original (Swin-Tiny + CORAL) | Proposed (Spatial Rad + Cross-Attn) | Delta |
| :--- | :---: | :---: | :---: |
| Exact-Grade Accuracy | 71.62% | 65.15% | -6.47% |
| Within-One-Grade Accuracy | 95.20% | 93.92% | -1.28% |
| Quadratic Weighted Kappa (QWK) | 0.8681 | 0.8366 | -0.0315 |
| Macro F1-Score | 0.5715 | 0.5145 | -0.0570 |
| Weighted F1-Score | 0.7464 | 0.6959 | -0.0505 |

### 2. Per-Grade Precision & Recall
| Grade | Support | Orig Prec | Prop Prec | Prec Delta | Orig Rec | Prop Rec | Rec Delta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Grade 0 | 628 | 87.40% | 87.41% | +0.01% | 90.61% | 81.85% | -8.76% |
| Grade 1 | 66 | 12.29% | 10.17% | -2.12% | 33.33% | 36.36% | +3.03% |
| Grade 2 | 444 | 92.76% | 86.73% | -6.03% | 46.17% | 41.22% | -4.95% |
| Grade 3 | 21 | 15.45% | 9.91% | -5.54% | 80.95% | 52.38% | -28.57% |
| Grade 4 | 92 | 92.22% | 79.05% | -13.17% | 90.22% | 90.22% | +0.00% |
