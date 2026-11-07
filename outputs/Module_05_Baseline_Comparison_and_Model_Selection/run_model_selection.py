import os
import json
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = "outputs/Module_05_Baseline_Comparison_and_Model_Selection"
os.makedirs(OUTPUT_DIR, exist_ok=True)

baselines = {
    "ConvNeXt-Small": {
        "module": "Module_02_Baseline_ConvNeXt_Small",
        "metrics_path": "outputs/Module_02_Baseline_ConvNeXt_Small/metrics.json",
        "config_path": "outputs/Module_02_Baseline_ConvNeXt_Small/config.yaml",
        "checkpoint": "outputs/Module_02_Baseline_ConvNeXt_Small/best_model.pth",
        "params": 49_458_533,
        "feature_dim": 768,
        "spatial_shape": [768, 7, 7]
    },
    "Swin-Tiny": {
        "module": "Module_03_Baseline_Swin_Tiny",
        "metrics_path": "outputs/Module_03_Baseline_Swin_Tiny/metrics.json",
        "config_path": "outputs/Module_03_Baseline_Swin_Tiny/config.yaml",
        "checkpoint": "outputs/Module_03_Baseline_Swin_Tiny/best_model.pth",
        "params": 27_523_199,
        "feature_dim": 768,
        "spatial_shape": [768, 7, 7]
    },
    "EfficientNet-B3": {
        "module": "Module_04_Baseline_EfficientNet_B3",
        "metrics_path": "outputs/Module_04_Baseline_EfficientNet_B3/metrics.json",
        "config_path": "outputs/Module_04_Baseline_EfficientNet_B3/config.yaml",
        "checkpoint": "outputs/Module_04_Baseline_EfficientNet_B3/best_model.pth",
        "params": 10_703_917,
        "feature_dim": 1536,
        "spatial_shape": [1536, 7, 7]
    }
}

records = []
for name, data in baselines.items():
    with open(data["metrics_path"], "r") as f:
        m = json.load(f)
    
    records.append({
        "Model": name,
        "Validation QWK": round(m["quadratic_weighted_kappa"], 4),
        "Validation Macro-F1": round(m["macro_f1"], 4),
        "Validation Accuracy": round(m["accuracy"], 4),
        "Validation Balanced Acc": round(m["balanced_accuracy"], 4),
        "Validation MAE": round(m["mean_absolute_error"], 4),
        "Validation Loss": round(m["loss"], 4),
        "Parameters (M)": round(data["params"] / 1e6, 2),
        "Training Time (s)": round(m.get("total_training_time_seconds", 0), 2),
        "Feature Dim": data["feature_dim"],
        "Spatial Token Grid": f"{data['spatial_shape'][1]}x{data['spatial_shape'][2]}"
    })

comparison_df = pd.DataFrame(records)

comparison_df = comparison_df.sort_values(by="Validation QWK", ascending=False).reset_index(drop=True)
comparison_df.insert(0, "Rank", range(1, len(comparison_df) + 1))

csv_path = os.path.join(OUTPUT_DIR, "baseline_comparison.csv")
comparison_df.to_csv(csv_path, index=False)
print("Saved baseline comparison table to:", csv_path)
print(comparison_df.to_string())

winner_row = comparison_df.iloc[0]
winner_name = winner_row["Model"]
winner_info = baselines[winner_name]

print(f"\n==================================================================")
print(f"  WINNING BASELINE BACKBONE: {winner_name}")
print(f"  Primary Metric (Val QWK):  {winner_row['Validation QWK']}")
print(f"  Secondary Metric (Macro-F1): {winner_row['Validation Macro-F1']}")
print(f"  Secondary Metric (Accuracy): {winner_row['Validation Accuracy']}")
print(f"  Secondary Metric (MAE):      {winner_row['Validation MAE']}")
print(f"==================================================================")

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Diabetic Retinopathy Baseline Comparison (Validation Set)", fontsize=16, fontweight="bold", y=0.98)

colors = ["#2b5c8f", "#2ca02c", "#d62728"]

ax = axes[0, 0]
bars = ax.bar(comparison_df["Model"], comparison_df["Validation QWK"], color=["#2ca02c", "#2b5c8f", "#d62728"], width=0.55, edgecolor="black")
ax.set_title("Primary Metric: Quadratic Weighted Kappa (QWK)", fontsize=13, fontweight="bold")
ax.set_ylabel("QWK Score (Higher is better)", fontsize=11)
ax.set_ylim(0, 1.08)
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.4f}", ha='center', va='bottom', fontweight='bold', fontsize=11)

ax = axes[0, 1]
bars = ax.bar(comparison_df["Model"], comparison_df["Validation Macro-F1"], color=["#2ca02c", "#2b5c8f", "#d62728"], width=0.55, edgecolor="black")
ax.set_title("Secondary Metric: Macro-Averaged F1-Score", fontsize=13, fontweight="bold")
ax.set_ylabel("Macro-F1 (Higher is better)", fontsize=11)
ax.set_ylim(0, 1.08)
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.4f}", ha='center', va='bottom', fontweight='bold', fontsize=11)

ax = axes[1, 0]
bars = ax.bar(comparison_df["Model"], comparison_df["Validation MAE"], color=["#2ca02c", "#2b5c8f", "#d62728"], width=0.55, edgecolor="black")
ax.set_title("Secondary Metric: Mean Absolute Error (MAE)", fontsize=13, fontweight="bold")
ax.set_ylabel("Grade Distance (Lower is better)", fontsize=11)
ax.set_ylim(0, 1.25)
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.4f}", ha='center', va='bottom', fontweight='bold', fontsize=11)

ax = axes[1, 1]
x = np.arange(len(comparison_df))
width = 0.35
b1 = ax.bar(x - width/2, comparison_df["Validation Accuracy"] * 100, width, label="Accuracy (%)", color="#1f77b4", edgecolor="black")
b2 = ax.bar(x + width/2, comparison_df["Parameters (M)"], width, label="Parameters (M)", color="#ff7f0e", edgecolor="black")
ax.set_title("Efficiency Trade-off: Accuracy vs. Model Size", fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(comparison_df["Model"])
ax.set_ylabel("Value", fontsize=11)
ax.legend(loc="upper right", frameon=True)
for bar in b1:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 1.0, f"{yval:.1f}%", ha='center', va='bottom', fontsize=9)
for bar in b2:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 1.0, f"{yval:.1f}M", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "baseline_comparison.png")
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close()
print("Saved comparison plot to:", plot_path)

selection_data = {
    "selected_model": winner_name,
    "selection_criteria": "Highest validation Quadratic Weighted Kappa (QWK) on standardized DDR split",
    "validation_metrics": {
        "quadratic_weighted_kappa": float(winner_row["Validation QWK"]),
        "macro_f1": float(winner_row["Validation Macro-F1"]),
        "accuracy": float(winner_row["Validation Accuracy"]),
        "balanced_accuracy": float(winner_row["Validation Balanced Acc"]),
        "mean_absolute_error": float(winner_row["Validation MAE"]),
        "loss": float(winner_row["Validation Loss"])
    },
    "architecture_specifications": {
        "name": winner_name,
        "backbone_type": "Hierarchical Vision Transformer (Shifted Windows)",
        "parameter_count": winner_info["params"],
        "feature_dim": winner_info["feature_dim"],
        "spatial_feature_shape": winner_info["spatial_shape"],
        "num_spatial_tokens": winner_info["spatial_shape"][1] * winner_info["spatial_shape"][2],
        "checkpoint_path": winner_info["checkpoint"]
    },
    "downstream_integration_contracts": {
        "module_06_radiomics": {
            "spatial_grid": "3x3 sub-regions",
            "feature_alignment": "Spatial pooling matches 2D retinal anatomy"
        },
        "module_07_cross_attention": {
            "query_modality": "Visual spatial tokens (B, 49, 768)",
            "key_value_modality": "Radiomic spatial tokens (B, 9, 768)",
            "output_shape": "(B, 49, 768)"
        },
        "module_08_imbalance": {
            "strategy": "Effective Number of Samples Class-Balanced Loss & Cost-Sensitive Matrix"
        },
        "module_09_ordinal": {
            "strategy": "CORAL Ordinal Regression (4 binary rank classifiers)"
        }
    },
    "comparison_summary": comparison_df.to_dict(orient="records")
}

json_path = os.path.join(OUTPUT_DIR, "model_selection.json")
with open(json_path, "w") as f:
    json.dump(selection_data, f, indent=4)
print("Saved official model selection contract to:", json_path)
