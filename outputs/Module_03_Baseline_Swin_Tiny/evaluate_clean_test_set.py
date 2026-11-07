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
    recall_score, precision_score, confusion_matrix, classification_report
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

def evaluate_test_set():
    print("==================================================================")
    print("  EVALUATING BEST MODEL ON HELD-OUT TEST SPLIT (1,251 IMAGES)")
    print(f"  Device: {DEVICE}")
    print("==================================================================")
    
    ckpt_path = os.path.join(CLEAN_DIR, "best_model_clean.pth")
    test_manifest = "data_manifests/split_80_10_10/test_manifest.csv"
    
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    assert os.path.exists(test_manifest), f"Manifest not found: {test_manifest}"
    
    model = SwinTinyCoralModel(num_classes=5)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()
    print(f"Loaded checkpoint saved from epoch {ckpt['epoch']} (Val QWK: {ckpt['val_qwk']:.4f})")
    
    test_transform = StandardRetinaTransform(is_training=False, img_size=224)
    test_ds = FullDDRDataset(test_manifest, transform=test_transform)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)
    
    print(f"Test Dataset size: {len(test_ds)} images")
    
    all_preds = []
    all_targets = []
    all_probs = []
    
    t0 = time.time()
    with torch.no_grad():
        for imgs, targets in test_loader:
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
    
    cm = confusion_matrix(y_true, y_pred, labels=list(range(5)))
    
    per_grade_prec = precision_score(y_true, y_pred, average=None, labels=list(range(5)), zero_division=0)
    per_grade_rec = recall_score(y_true, y_pred, average=None, labels=list(range(5)), zero_division=0)
    
    per_grade_spec = []
    for g in range(5):
        tn = np.sum(np.delete(np.delete(cm, g, axis=0), g, axis=1))
        fp = np.sum(cm[:, g]) - cm[g, g]
        spec = tn / (tn + fp + 1e-8)
        per_grade_spec.append(spec)
        
    per_grade_df = pd.DataFrame({
        "Grade": list(range(5)),
        "Grade Name": GRADE_NAMES,
        "Support": np.bincount(y_true, minlength=5),
        "Precision": np.round(per_grade_prec, 4),
        "Recall (Sensitivity)": np.round(per_grade_rec, 4),
        "Specificity": np.round(per_grade_spec, 4),
        "F1-Score": np.round([f1_score(y_true == g, y_pred == g, zero_division=0) for g in range(5)], 4)
    })
    
    print("\n================== HELD-OUT TEST RESULTS ==================")
    print(f"  Overall Accuracy:             {acc*100:.2f}%")
    print(f"  Quadratic Weighted Kappa (QWK): {qwk:.4f}")
    print(f"  Macro F1-Score:               {macro_f1:.4f}")
    print(f"  Weighted F1-Score:            {weighted_f1:.4f}")
    print(f"  Inference Throughput:         {len(y_true)/eval_time:.1f} imgs/sec ({eval_time:.1f}s total)")
    print("===========================================================\n")
    print("Per-Grade Breakdown:")
    print(per_grade_df.to_string(index=False))
    
    print("\nConfusion Matrix (Rows: Ground Truth, Cols: Predicted):")
    cm_df = pd.DataFrame(cm, index=[f"True G{i}" for i in range(5)], columns=[f"Pred G{i}" for i in range(5)])
    print(cm_df)
    
    cm_df.to_csv(os.path.join(CLEAN_DIR, "test_confusion_matrix.csv"))
    per_grade_df.to_csv(os.path.join(CLEAN_DIR, "test_classification_report.csv"), index=False)
    
    results = {
        "test_samples": len(y_true),
        "accuracy": round(acc, 4),
        "quadratic_weighted_kappa": round(qwk, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "confusion_matrix": cm.tolist(),
        "per_grade_metrics": per_grade_df.to_dict(orient="records")
    }
    with open(os.path.join(CLEAN_DIR, "test_evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nSaved test evaluation metrics to {CLEAN_DIR}/test_evaluation_results.json")
    
    print("\n=== SPOT-CHECKING KNOWN TEST IMAGES ===")
    test_df = pd.read_csv(test_manifest)
    for g in range(5):
        sample_row = test_df[test_df["numeric_label"] == g].iloc[0]
        fname = sample_row["image_id"]
        fp = sample_row["filepath"]
        
        with Image.open(fp) as im:
            t_img = test_transform(im.convert("RGB")).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            s_logits = model(t_img)
            p_grade = coral_predict(s_logits, threshold=0.5).item()
            p_probs = coral_probabilities(s_logits).squeeze().cpu().numpy().tolist()
            
        print(f"Sample Grade {g} ({fname}): Ground Truth = {g} | "
              f"Predicted = {p_grade} | Confidence = {p_probs[p_grade]*100:.1f}% | "
              f"Distribution = {[round(p, 3) for p in p_probs]}")

if __name__ == "__main__":
    evaluate_test_set()
