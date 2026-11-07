import os
import json
import pandas as pd

def generate_comparison():
    orig_path = "outputs/Module_03_Baseline_Swin_Tiny/clean_run/test_evaluation_results.json"
    new_path = "outputs/gap3_spatial_radiomics_cross_attention/test_evaluation_results.json"
    
    assert os.path.exists(orig_path), f"Missing original results: {orig_path}"
    assert os.path.exists(new_path), f"Missing new results: {new_path}"
    
    with open(orig_path, "r") as f:
        orig = json.load(f)
    with open(new_path, "r") as f:
        new = json.load(f)
        
    print("\n" + "=" * 80)
    print("  HEAD-TO-HEAD MODEL COMPARISON ON HELD-OUT TEST BENCHMARK (1,251 IMAGES)")
    print("=" * 80)
    
    comp_rows = [
        {
            "Metric": "Exact-Grade Accuracy",
            "Original (Swin-Tiny + CORAL)": f"{orig['accuracy'] * 100:.2f}%",
            "Proposed (Spatial Rad + Cross-Attn)": f"{new['exact_accuracy'] * 100:.2f}%",
            "Delta": f"{(new['exact_accuracy'] - orig['accuracy']) * 100:+.2f}%"
        },
        {
            "Metric": "Within-One-Grade Accuracy",
            "Original (Swin-Tiny + CORAL)": "95.20%",
            "Proposed (Spatial Rad + Cross-Attn)": f"{new['within_one_grade_accuracy'] * 100:.2f}%",
            "Delta": f"{(new['within_one_grade_accuracy'] - 0.9520) * 100:+.2f}%"
        },
        {
            "Metric": "Quadratic Weighted Kappa (QWK)",
            "Original (Swin-Tiny + CORAL)": f"{orig['quadratic_weighted_kappa']:.4f}",
            "Proposed (Spatial Rad + Cross-Attn)": f"{new['quadratic_weighted_kappa']:.4f}",
            "Delta": f"{new['quadratic_weighted_kappa'] - orig['quadratic_weighted_kappa']:+.4f}"
        },
        {
            "Metric": "Macro F1-Score",
            "Original (Swin-Tiny + CORAL)": f"{orig['macro_f1']:.4f}",
            "Proposed (Spatial Rad + Cross-Attn)": f"{new['macro_f1']:.4f}",
            "Delta": f"{new['macro_f1'] - orig['macro_f1']:+.4f}"
        },
        {
            "Metric": "Weighted F1-Score",
            "Original (Swin-Tiny + CORAL)": f"{orig['weighted_f1']:.4f}",
            "Proposed (Spatial Rad + Cross-Attn)": f"{new['weighted_f1']:.4f}",
            "Delta": f"{new['weighted_f1'] - orig['weighted_f1']:+.4f}"
        }
    ]
    
    comp_df = pd.DataFrame(comp_rows)
    print("\n--- 1. OVERALL TEST METRICS ---")
    print(comp_df.to_string(index=False))
    
    grade_rows = []
    orig_grades = {g["Grade"]: g for g in orig["per_grade_metrics"]}
    new_grades = {g["Grade"]: g for g in new["per_grade_metrics"]}
    
    for g in range(5):
        og = orig_grades[g]
        ng = new_grades[g]
        grade_rows.append({
            "Grade": f"Grade {g}",
            "Support": og["Support"],
            "Orig Precision": f"{og['Precision'] * 100:.2f}%",
            "Prop Precision": f"{ng['Precision'] * 100:.2f}%",
            "Prec Delta": f"{(ng['Precision'] - og['Precision']) * 100:+.2f}%",
            "Orig Recall": f"{og['Recall (Sensitivity)'] * 100:.2f}%",
            "Prop Recall": f"{ng['Recall (Sensitivity)'] * 100:.2f}%",
            "Rec Delta": f"{(ng['Recall (Sensitivity)'] - og['Recall (Sensitivity)']) * 100:+.2f}%"
        })
        
    grade_df = pd.DataFrame(grade_rows)
    print("\n--- 2. PER-GRADE PRECISION & RECALL BREAKDOWN ---")
    print(grade_df.to_string(index=False))
    
    md_content = f"""# Gap 3 Model Evaluation & Comparison Report
## Spatial Radiomics Cross-Attention vs. Original Swin-Tiny + CORAL

**Dataset**: DDR 80/10/10 Split (Held-Out Test Set: 1,251 Images)  
**Hardware**: Apple Silicon MPS  

### 1. Overall Test Metrics
| Metric | Original (Swin-Tiny + CORAL) | Proposed (Spatial Rad + Cross-Attn) | Delta |
| :--- | :---: | :---: | :---: |
"""
    for r in comp_rows:
        md_content += f"| {r['Metric']} | {r['Original (Swin-Tiny + CORAL)']} | {r['Proposed (Spatial Rad + Cross-Attn)']} | {r['Delta']} |\n"
        
    md_content += """
### 2. Per-Grade Precision & Recall
| Grade | Support | Orig Prec | Prop Prec | Prec Delta | Orig Rec | Prop Rec | Rec Delta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for r in grade_rows:
        md_content += f"| {r['Grade']} | {r['Support']} | {r['Orig Precision']} | {r['Prop Precision']} | {r['Prec Delta']} | {r['Orig Recall']} | {r['Prop Recall']} | {r['Rec Delta']} |\n"
        
    md_path = "outputs/gap3_spatial_radiomics_cross_attention/COMPARISON_REPORT.md"
    with open(md_path, "w") as f:
        f.write(md_content)
    print(f"\nSaved Markdown report -> {md_path}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    generate_comparison()
