import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    cohen_kappa_score,
    f1_score,
    balanced_accuracy_score,
    accuracy_score,
    mean_absolute_error,
    classification_report,
    confusion_matrix
)

from src.data.data_pipeline import CLASS_NAMES

def compute_classification_metrics(y_true, y_pred, num_classes: int = 5) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    
    clf_dict = classification_report(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        target_names=CLASS_NAMES[:num_classes],
        output_dict=True,
        zero_division=0
    )
    
    metrics = {
        "quadratic_weighted_kappa": float(qwk),
        "macro_f1": float(macro_f1),
        "balanced_accuracy": float(balanced_acc),
        "accuracy": float(acc),
        "mean_absolute_error": float(mae),
        "confusion_matrix": cm.tolist(),
        "per_class": {}
    }
    
    for i, name in enumerate(CLASS_NAMES[:num_classes]):
        if name in clf_dict:
            metrics["per_class"][f"grade_{i}_{name}"] = {
                "precision": float(clf_dict[name]["precision"]),
                "recall": float(clf_dict[name]["recall"]),
                "f1_score": float(clf_dict[name]["f1-score"]),
                "support": int(clf_dict[name]["support"])
            }
            
    return metrics, cm, clf_dict

def plot_confusion_matrix(cm: np.ndarray, save_path: str, title: str = "Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(8, 7))
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
    
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]:.1%})"
            
    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=CLASS_NAMES[:len(cm)],
        yticklabels=CLASS_NAMES[:len(cm)],
        cbar=True,
        ax=ax,
        linewidths=0.5,
        linecolor="#dddddd"
    )
    
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Grade", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Grade", fontsize=12, fontweight="bold")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

def save_evaluation_artifacts(metrics: dict, cm: np.ndarray, clf_dict: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
        
    report_rows = []
    for grade_idx, name in enumerate(CLASS_NAMES[:len(cm)]):
        if name in clf_dict:
            row = clf_dict[name]
            report_rows.append({
                "grade": grade_idx,
                "grade_name": name,
                "precision": round(row["precision"], 4),
                "recall": round(row["recall"], 4),
                "f1_score": round(row["f1-score"], 4),
                "support": row["support"]
            })
    for avg in ["macro avg", "weighted avg"]:
        if avg in clf_dict:
            report_rows.append({
                "grade": -1,
                "grade_name": avg,
                "precision": round(clf_dict[avg]["precision"], 4),
                "recall": round(clf_dict[avg]["recall"], 4),
                "f1_score": round(clf_dict[avg]["f1-score"], 4),
                "support": clf_dict[avg]["support"]
            })
    pd.DataFrame(report_rows).to_csv(os.path.join(output_dir, "classification_report.csv"), index=False)
    
    cm_plot_path = os.path.join(output_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, cm_plot_path, title=f"Confusion Matrix (QWK: {metrics['quadratic_weighted_kappa']:.4f})")
    
    print(f"Saved evaluation artifacts to {output_dir}")
