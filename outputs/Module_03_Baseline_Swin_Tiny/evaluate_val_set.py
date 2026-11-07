import os
import sys
import time
import json
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    cohen_kappa_score, f1_score, accuracy_score,
    recall_score, precision_score, confusion_matrix, classification_report,
    mean_absolute_error
)

sys.path.append(os.path.abspath("."))
from outputs.Module_03_Baseline_Swin_Tiny.train_full_ddr_swin_coral import (
    StandardRetinaTransform, FullDDRDataset, SwinTinyCoralModel
)
from src.models.coral import coral_predict, coral_probabilities

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
CLEAN_DIR = "outputs/Module_03_Baseline_Swin_Tiny/clean_run"

GRADE_NAMES = [
    "Grade 0: No DR",
    "Grade 1: Mild NPDR",
    "Grade 2: Moderate NPDR",
    "Grade 3: Severe NPDR",
    "Grade 4: Proliferative DR"
]

def evaluate_val_set():
    print("==================================================================")
    print("  EVALUATING SWIN-TINY BEST MODEL ON VALIDATION SPLIT (1,247 IMAGES)")
    print(f"  Device: {DEVICE}")
    print("==================================================================")
    
    ckpt_path = os.path.join(CLEAN_DIR, "best_model_clean.pth")
    val_manifest = "data_manifests/split_80_10_10/val_manifest.csv"
    
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    assert os.path.exists(val_manifest), f"Manifest not found: {val_manifest}"
    
    model = SwinTinyCoralModel(num_classes=5)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()
    print(f"Loaded checkpoint saved from epoch {ckpt['epoch']} (Val QWK: {ckpt['val_qwk']:.4f})")
    
    val_transform = StandardRetinaTransform(is_training=False, img_size=224)
    val_ds = FullDDRDataset(val_manifest, transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    
    val_df = pd.read_csv(val_manifest)
    print(f"Validation Dataset size: {len(val_ds)} images")
    
    all_preds = []
    all_targets = []
    all_probs = []
    
    t0 = time.time()
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs = imgs.to(DEVICE)
            logits = model(imgs)
            preds = coral_predict(logits, threshold=0.5)
            probs = coral_probabilities(logits)
            
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            
    eval_time = time.time() - t0
    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    probs_arr = np.array(all_probs)
    
    acc = accuracy_score(y_true, y_pred)
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    mae = mean_absolute_error(y_true, y_pred)
    
    pred_records = []
    for idx in range(len(val_df)):
        pred_records.append({
            "image_id": val_df.iloc[idx]["image_id"],
            "true_label": int(y_true[idx]),
            "predicted_grade": int(y_pred[idx]),
            "prob_g0": float(probs_arr[idx][0]),
            "prob_g1": float(probs_arr[idx][1]),
            "prob_g2": float(probs_arr[idx][2]),
            "prob_g3": float(probs_arr[idx][3]),
            "prob_g4": float(probs_arr[idx][4])
        })
    pred_df_out = pd.DataFrame(pred_records)
    pred_csv_path = os.path.join(CLEAN_DIR, "val_predictions.csv")
    pred_df_out.to_csv(pred_csv_path, index=False)
    print(f"Saved validation predictions to {pred_csv_path} ({len(pred_df_out)} rows)")
    
    cm = confusion_matrix(y_true, y_pred, labels=list(range(5)))
    per_grade_prec = precision_score(y_true, y_pred, average=None, labels=list(range(5)), zero_division=0)
    per_grade_rec = recall_score(y_true, y_pred, average=None, labels=list(range(5)), zero_division=0)
    per_grade_f1 = [f1_score(y_true == g, y_pred == g, zero_division=0) for g in range(5)]
    
    per_grade_df = pd.DataFrame({
        "Grade": list(range(5)),
        "Grade Name": GRADE_NAMES,
        "Support": np.bincount(y_true, minlength=5),
        "Precision": np.round(per_grade_prec, 4),
        "Recall": np.round(per_grade_rec, 4),
        "F1-Score": np.round(per_grade_f1, 4)
    })
    
    report_csv_path = os.path.join(CLEAN_DIR, "val_classification_report.csv")
    per_grade_df.to_csv(report_csv_path, index=False)
    
    print("\n================== VALIDATION RESULTS ==================")
    print(f"  Overall Accuracy:             {acc*100:.2f}%")
    print(f"  Quadratic Weighted Kappa (QWK): {qwk:.4f}")
    print(f"  Macro F1-Score:               {macro_f1:.4f}")
    print(f"  Weighted F1-Score:            {weighted_f1:.4f}")
    print(f"  MAE:                          {mae:.4f}")
    print(f"  Evaluation Time:              {eval_time:.1f}s")
    print("========================================================\n")
    print("Per-Grade Breakdown:")
    print(per_grade_df.to_string(index=False))
    
    val_results = {
        "model_name": "Swin-Tiny + CORAL (Full DDR)",
        "val_samples": len(y_true),
        "accuracy": round(acc, 4),
        "quadratic_weighted_kappa": round(qwk, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "mae": round(mae, 4),
        "confusion_matrix": cm.tolist(),
        "per_grade_metrics": per_grade_df.to_dict(orient="records")
    }
    with open(os.path.join(CLEAN_DIR, "val_evaluation_results.json"), "w") as f:
        json.dump(val_results, f, indent=2)
    print(f"Saved validation evaluation results to {CLEAN_DIR}/val_evaluation_results.json")

if __name__ == "__main__":
    evaluate_val_set()
