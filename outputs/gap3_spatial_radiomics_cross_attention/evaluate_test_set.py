import os
import sys
import json
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    cohen_kappa_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)

sys.path.append(os.path.abspath("."))
from outputs.gap3_spatial_radiomics_cross_attention.model import SpatialRadiomicsSwinCoralModel
from outputs.gap3_spatial_radiomics_cross_attention.train_spatial_cross_attention import (
    MultiModalDDRDataset,
    StandardRetinaTransform
)
from src.models.coral import coral_predict, coral_probabilities

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
OUTPUT_DIR = "outputs/gap3_spatial_radiomics_cross_attention"
CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")

GRADE_NAMES = [
    "Grade 0: No DR",
    "Grade 1: Mild NPDR",
    "Grade 2: Moderate NPDR",
    "Grade 3: Severe NPDR",
    "Grade 4: Proliferative DR"
]

def evaluate_test_set():
    print("=" * 70)
    print("  EVALUATING GAP 3 SPATIAL RADIOMICS CROSS-ATTENTION ON HELD-OUT TEST SET")
    print(f"  Device: {DEVICE}")
    print("=" * 70)
    
    test_manifest = "data_manifests/split_80_10_10/test_manifest.csv"
    test_cache = os.path.join(CACHE_DIR, "radiomics_test.npz")
    ckpt_path = os.path.join(OUTPUT_DIR, "best_spatial_radiomics_coral_model.pth")
    
    assert os.path.exists(test_manifest), f"Missing {test_manifest}"
    assert os.path.exists(test_cache), f"Missing {test_cache}"
    assert os.path.exists(ckpt_path), f"Missing checkpoint {ckpt_path}"
    
    transform = StandardRetinaTransform(is_training=False, img_size=224)
    test_ds = MultiModalDDRDataset(test_manifest, test_cache, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)
    
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    
    model = SpatialRadiomicsSwinCoralModel(
        num_classes=5,
        d_vis=768,
        d_rad=32,
        d_model=768,
        num_heads=8,
        dropout=0.0,
        pretrained_backbone=False
    )
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    print(f"Loaded trained checkpoint: {ckpt_path} (Trained Epoch {checkpoint.get('epoch', 'N/A')}, Val QWK: {checkpoint.get('val_qwk', 'N/A')})")
    
    all_targets = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for b_idx, batch in enumerate(test_loader):
            imgs = batch["image"].to(DEVICE)
            rads = batch["radiomics"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            
            rank_logits = model(imgs, rads)
            preds = coral_predict(rank_logits)
            probs = coral_probabilities(rank_logits)
            
            all_targets.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            
    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    probs_matrix = np.array(all_probs)
    
    acc = accuracy_score(y_true, y_pred)
    within_one = np.mean(np.abs(y_true - y_pred) <= 1)
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
    precisions = precision_score(y_true, y_pred, average=None, labels=[0, 1, 2, 3, 4], zero_division=0)
    recalls = recall_score(y_true, y_pred, average=None, labels=[0, 1, 2, 3, 4], zero_division=0)
    f1s = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2, 3, 4], zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
    
    specificities = []
    for g in range(5):
        tp = cm[g, g]
        fn = cm[g, :].sum() - tp
        fp = cm[:, g].sum() - tp
        tn = cm.sum() - (tp + fn + fp)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(float(spec))
        
    per_grade_list = []
    for g in range(5):
        per_grade_list.append({
            "Grade": g,
            "Grade Name": GRADE_NAMES[g],
            "Support": int(cm[g, :].sum()),
            "Precision": round(float(precisions[g]), 4),
            "Recall (Sensitivity)": round(float(recalls[g]), 4),
            "Specificity": round(float(specificities[g]), 4),
            "F1-Score": round(float(f1s[g]), 4)
        })
        
    errors = np.abs(y_true - y_pred)
    error_counts = {int(dist): int((errors == dist).sum()) for dist in range(5)}
    error_pcts = {int(dist): round(float((errors == dist).mean() * 100), 2) for dist in range(5)}
    
    results = {
        "model_name": "SpatialRadiomicsSwinCoralModel (Gap 3 Proposed)",
        "test_samples": len(y_true),
        "exact_accuracy": round(float(acc), 4),
        "within_one_grade_accuracy": round(float(within_one), 4),
        "quadratic_weighted_kappa": round(float(qwk), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "error_distance_counts": error_counts,
        "error_distance_percentages": error_pcts,
        "confusion_matrix": cm.tolist(),
        "per_grade_metrics": per_grade_list
    }
    
    results_json = os.path.join(OUTPUT_DIR, "test_evaluation_results.json")
    with open(results_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved test results JSON -> {results_json}")
    
    report_df = pd.DataFrame(per_grade_list)
    report_csv = os.path.join(OUTPUT_DIR, "test_classification_report.csv")
    report_df.to_csv(report_csv, index=False)
    print(f"Saved classification report CSV -> {report_csv}")
    
    cm_df = pd.DataFrame(cm, index=[f"True_G{i}" for i in range(5)], columns=[f"Pred_G{i}" for i in range(5)])
    cm_csv = os.path.join(OUTPUT_DIR, "test_confusion_matrix.csv")
    cm_df.to_csv(cm_csv)
    print(f"Saved confusion matrix CSV -> {cm_csv}")
    
    manifest_df = pd.read_csv(test_manifest)
    preds_df = pd.DataFrame({
        "image_id": manifest_df["image_id"],
        "filepath": manifest_df["filepath"],
        "true_grade": y_true,
        "predicted_grade": y_pred,
        "abs_error": errors,
        "prob_g0": probs_matrix[:, 0].round(4),
        "prob_g1": probs_matrix[:, 1].round(4),
        "prob_g2": probs_matrix[:, 2].round(4),
        "prob_g3": probs_matrix[:, 3].round(4),
        "prob_g4": probs_matrix[:, 4].round(4)
    })
    preds_csv = os.path.join(OUTPUT_DIR, "test_predictions.csv")
    preds_df.to_csv(preds_csv, index=False)
    print(f"Saved row-level predictions -> {preds_csv}")
    
    print("\n" + "=" * 70)
    print("  TEST EVALUATION SUMMARY: GAP 3 SPATIAL RADIOMICS CROSS-ATTENTION")
    print("=" * 70)
    print(f"  Total Test Samples:             {len(y_true)}")
    print(f"  Exact-Grade Accuracy:           {acc * 100:.2f}%")
    print(f"  Within-One-Grade Accuracy:      {within_one * 100:.2f}%")
    print(f"  Quadratic Weighted Kappa (QWK): {qwk:.4f}")
    print(f"  Macro F1-Score:                 {macro_f1:.4f}")
    print(f"  Weighted F1-Score:              {weighted_f1:.4f}")
    print("\n  Per-Grade Detailed Breakdown:")
    print(report_df.to_string(index=False))
    print("\n  Confusion Matrix:")
    print(cm_df.to_string())
    print("\n  Severity-Distance Error Distribution:")
    for dist in range(5):
        print(f"    Distance {dist}: {error_counts[dist]} images ({error_pcts[dist]}%)")
    print("=" * 70)
    
    return results

if __name__ == "__main__":
    evaluate_test_set()
