import os
import sys
import time
import json
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, f1_score,
    mean_absolute_error, confusion_matrix, classification_report
)

sys.path.append(os.path.abspath("."))
from src.data.data_pipeline import retina_circle_crop, apply_clahe_lab, IMAGENET_MEAN, IMAGENET_STD
from src.models.coral import CoralLayer, coral_predict, coral_probabilities
from outputs.gap3_spatial_radiomics_cross_attention.model import SpatialRadiomicsSwinCoralModel
from outputs.Module_11_Ablation_Studies.train_ablation_m1 import M1_LateConcatenationModel
from outputs.Module_11_Ablation_Studies.train_ablation_m2 import M2_GlobalRadiomicsModel
from outputs.Module_11_Ablation_Studies.train_ablation_m3 import M3_NominalSoftmaxModel
from outputs.Module_03_Baseline_Swin_Tiny.train_full_ddr_swin_coral import SwinTinyCoralModel
from outputs.Module_11_Ablation_Studies.train_ablation_m5 import M5_ConvNeXtMultimodalModel

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
OUTPUT_DIR = "outputs/Module_12_Final_Evaluation_on_Held_Out_Test_Split"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class StandardRetinaTransform:
    def __init__(self, img_size: int = 224):
        self.spatial_transforms = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

    def __call__(self, img_pil: Image.Image) -> torch.Tensor:
        img_np = np.array(img_pil)
        cropped_np = retina_circle_crop(img_np)
        enhanced_np = apply_clahe_lab(cropped_np)
        enhanced_pil = Image.fromarray(enhanced_np)
        return self.spatial_transforms(enhanced_pil)

class TestDDRDataset(Dataset):
    def __init__(self, manifest_csv: str, cache_npz: str, transform=None):
        self.df = pd.read_csv(manifest_csv).reset_index(drop=True)
        self.transform = transform
        self.filepaths = self.df["filepath"].tolist()
        self.labels = self.df["numeric_label"].tolist()
        self.image_ids = self.df["image_id"].tolist()
        
        cache_data = np.load(cache_npz)
        self.radiomics = cache_data["X_spatial_norm"].astype(np.float32)
        assert len(self.df) == len(self.radiomics), f"Manifest len ({len(self.df)}) != Cache len ({len(self.radiomics)})"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        fp = self.filepaths[idx]
        label = self.labels[idx]
        img_id = self.image_ids[idx]
        rad_tokens = torch.from_numpy(self.radiomics[idx])
        
        with Image.open(fp) as im:
            img = im.convert("RGB")
            if self.transform:
                img_tensor = self.transform(img)
            else:
                img_tensor = transforms.ToTensor()(img)
                
        return {
            "image": img_tensor,
            "radiomics": rad_tokens,
            "label": torch.tensor(label, dtype=torch.long),
            "image_id": img_id,
            "filepath": fp
        }

def evaluate_model(model_key: str, model: nn.Module, test_loader: DataLoader, is_coral: bool = True):
    print(f"\n---> Evaluating {model_key} on {len(test_loader.dataset)} test samples...")
    model.eval()
    
    all_image_ids = []
    all_filepaths = []
    all_y_true = []
    all_y_pred = []
    all_probs = []
    
    t0 = time.time()
    with torch.no_grad():
        for batch in test_loader:
            imgs = batch["image"].to(DEVICE)
            rads = batch["radiomics"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            img_ids = batch["image_id"]
            fps = batch["filepath"]
            
            if model_key == "M4a":
                logits = model(imgs)
            elif model_key == "M1":
                rads_flat = rads.reshape(rads.size(0), -1)
                logits = model(imgs, rads_flat)
            else:
                logits = model(imgs, rads)
                
            if is_coral:
                preds = coral_predict(logits, threshold=0.5)
                probs = coral_probabilities(logits)
            else:
                preds = torch.argmax(logits, dim=-1)
                probs = F.softmax(logits, dim=-1)
                
            all_image_ids.extend(img_ids)
            all_filepaths.extend(fps)
            all_y_true.extend(labels.cpu().numpy().tolist())
            all_y_pred.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            
    eval_duration = time.time() - t0
    y_true = np.array(all_y_true)
    y_pred = np.array(all_y_pred)
    probs_arr = np.array(all_probs)
    n = len(y_true)
    
    acc = accuracy_score(y_true, y_pred)
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    mae = mean_absolute_error(y_true, y_pred)
    
    diffs = np.abs(y_true - y_pred)
    exact_cnt = int(np.sum(diffs == 0))
    adj_cnt = int(np.sum(diffs == 1))
    cat_cnt = int(np.sum(diffs >= 2))
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
    clf_rep = classification_report(y_true, y_pred, labels=[0, 1, 2, 3, 4], output_dict=True, zero_division=0)
    
    pred_df = pd.DataFrame({
        "image_id": all_image_ids,
        "filepath": all_filepaths,
        "true_label": y_true,
        "predicted_grade": y_pred,
        "prob_g0": probs_arr[:, 0],
        "prob_g1": probs_arr[:, 1],
        "prob_g2": probs_arr[:, 2],
        "prob_g3": probs_arr[:, 3],
        "prob_g4": probs_arr[:, 4]
    })
    
    model_out_dir = os.path.join(OUTPUT_DIR, model_key)
    os.makedirs(model_out_dir, exist_ok=True)
    pred_df.to_csv(os.path.join(model_out_dir, "test_predictions.csv"), index=False)
    
    cm_df = pd.DataFrame(cm, index=[f"True_G{i}" for i in range(5)], columns=[f"Pred_G{i}" for i in range(5)])
    cm_df.to_csv(os.path.join(model_out_dir, "test_confusion_matrix.csv"))
    
    pd.DataFrame(clf_rep).transpose().to_csv(os.path.join(model_out_dir, "test_classification_report.csv"))
    
    per_grade = {}
    for g in range(5):
        per_grade[f"Grade_{g}"] = {
            "precision": round(float(clf_rep[str(g)]["precision"]), 4),
            "recall": round(float(clf_rep[str(g)]["recall"]), 4),
            "f1_score": round(float(clf_rep[str(g)]["f1-score"]), 4),
            "support": int(clf_rep[str(g)]["support"])
        }
        
    metrics = {
        "model_key": model_key,
        "test_samples": n,
        "test_accuracy": round(float(acc), 4),
        "test_exact_matches": exact_cnt,
        "test_qwk": round(float(qwk), 4),
        "test_macro_f1": round(float(macro_f1), 4),
        "test_weighted_f1": round(float(weighted_f1), 4),
        "test_mae": round(float(mae), 4),
        "test_adjacent_errors": adj_cnt,
        "test_adjacent_error_rate": round(float(adj_cnt / n), 4),
        "test_catastrophic_errors": cat_cnt,
        "test_catastrophic_error_rate": round(float(cat_cnt / n), 4),
        "eval_duration_sec": round(float(eval_duration), 2),
        "confusion_matrix": cm.tolist(),
        "per_grade_metrics": per_grade
    }
    
    with open(os.path.join(model_out_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"[{model_key}] Test QWK: {qwk:.4f} | Acc: {exact_cnt}/{n} ({acc*100:.2f}%) | Macro-F1: {macro_f1:.4f} | MAE: {mae:.4f}")
    print(f"[{model_key}] Adjacent Errors: {adj_cnt}/{n} ({adj_cnt/n*100:.2f}%) | Catastrophic Errors: {cat_cnt}/{n} ({cat_cnt/n*100:.2f}%)")
    print(f"[{model_key}] Duration: {eval_duration:.1f}s | Saved to {model_out_dir}/")
    
    return metrics

def run_all_test_evaluations():
    print("=" * 80)
    print("  MODULE 12: FINAL HELD-OUT TEST EVALUATION ON REAL DDR DATASET")
    print(f"  Device: {DEVICE} | Test Manifest: data_manifests/split_80_10_10/test_manifest.csv")
    print("=" * 80)
    
    test_manifest_path = "data_manifests/split_80_10_10/test_manifest.csv"
    test_cache_path = "outputs/gap3_spatial_radiomics_cross_attention/cache/radiomics_test.npz"
    
    transform = StandardRetinaTransform(img_size=224)
    test_ds = TestDDRDataset(test_manifest_path, test_cache_path, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)
    
    print(f"Loaded {len(test_ds)} held-out test fundus images.")
    
    model_specs = [
        {
            "key": "M0",
            "name": "M0: Full Proposed Reference",
            "desc": "Swin-T + 9 Spatial Radiomic Tokens + Cross-Attn + CORAL",
            "ckpt": "outputs/Module_11_Ablation_Studies/M0_Full_Proposed/best_model.pth",
            "is_coral": True,
            "params": 33.28,
            "init_fn": lambda: SpatialRadiomicsSwinCoralModel(num_classes=5, d_vis=768, d_rad=32, d_model=768, num_heads=8, dropout=0.2)
        },
        {
            "key": "M1",
            "name": "M1: Late Concatenation",
            "desc": "Swin-T (768d) + Flat Radiomics (288d) Late Concat + CORAL",
            "ckpt": "outputs/Module_11_Ablation_Studies/M1_Late_Concatenation/best_model.pth",
            "is_coral": True,
            "params": 28.53,
            "init_fn": lambda: M1_LateConcatenationModel(num_classes=5, d_vis=768, d_rad=288, dropout=0.2)
        },
        {
            "key": "M2",
            "name": "M2: Spatial Localization",
            "desc": "Swin-T + 1 Global Radiomics Token + Cross-Attn + CORAL",
            "ckpt": "outputs/Module_11_Ablation_Studies/M2_Global_Radiomics/best_model.pth",
            "is_coral": True,
            "params": 33.28,
            "init_fn": lambda: M2_GlobalRadiomicsModel(num_classes=5, d_vis=768, d_rad_global=288, d_model=768, num_heads=8, dropout=0.2)
        },
        {
            "key": "M3",
            "name": "M3: Ordinal Learning",
            "desc": "Swin-T + 9 Spatial Radiomic Tokens + Cross-Attn + Nominal Softmax",
            "ckpt": "outputs/Module_11_Ablation_Studies/M3_Nominal_Softmax/best_model.pth",
            "is_coral": False,
            "params": 32.89,
            "init_fn": lambda: M3_NominalSoftmaxModel(num_classes=5, d_vis=768, d_rad=32, d_model=768, num_heads=8, dropout=0.2)
        },
        {
            "key": "M4a",
            "name": "M4a: Visual-Only Baseline",
            "desc": "Swin-T Clean Baseline + CORAL Head",
            "ckpt": "outputs/Module_11_Ablation_Studies/M4a_Visual_Only/best_model.pth",
            "is_coral": True,
            "params": 27.52,
            "init_fn": lambda: SwinTinyCoralModel(num_classes=5)
        },
        {
            "key": "M5",
            "name": "M5: Backbone Adaptation",
            "desc": "ConvNeXt-S (7x7 feat) + 9 Spatial Radiomics + Cross-Attn + CORAL",
            "ckpt": "outputs/Module_11_Ablation_Studies/M5_ConvNeXt_Multimodal/best_model.pth",
            "is_coral": True,
            "params": 54.82,
            "init_fn": lambda: M5_ConvNeXtMultimodalModel(num_classes=5, d_vis=768, d_rad=32, d_model=768, num_heads=8, dropout=0.2)
        }
    ]
    
    all_test_metrics = []
    
    val_report_df = pd.read_csv("outputs/Module_11_Ablation_Studies/consolidated_ablation_report.csv").set_index("ablation_id")
    
    for spec in model_specs:
        k = spec["key"]
        print(f"\nLoading {spec['name']} from {spec['ckpt']}...")
        model = spec["init_fn"]().to(DEVICE)
        ckpt = torch.load(spec["ckpt"], map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        print(f"Loaded checkpoint state_dict successfully (saved from epoch {ckpt.get('epoch', 'N/A')}).")
        
        m_res = evaluate_model(k, model, test_loader, is_coral=spec["is_coral"])
        
        val_row = val_report_df.loc[k]
        
        comparison_row = {
            "ablation_id": k,
            "model_name": spec["name"],
            "description": spec["desc"],
            "params_m": spec["params"],
            "val_samples": int(val_row["val_samples"]),
            "test_samples": m_res["test_samples"],
            "val_qwk": float(val_row["val_qwk"]),
            "test_qwk": m_res["test_qwk"],
            "qwk_generalization_gap": round(float(m_res["test_qwk"] - val_row["val_qwk"]), 4),
            "val_accuracy": float(val_row["val_accuracy"]),
            "test_accuracy": m_res["test_accuracy"],
            "acc_generalization_gap": round(float(m_res["test_accuracy"] - val_row["val_accuracy"]), 4),
            "val_macro_f1": float(val_row["val_macro_f1"]),
            "test_macro_f1": m_res["test_macro_f1"],
            "val_mae": float(val_row["val_mae"]),
            "test_mae": m_res["test_mae"],
            "val_adjacent_error_rate": float(val_row["adjacent_error_rate"]),
            "test_adjacent_error_rate": m_res["test_adjacent_error_rate"],
            "test_adjacent_errors": m_res["test_adjacent_errors"],
            "val_catastrophic_error_rate": float(val_row["catastrophic_error_rate"]),
            "test_catastrophic_error_rate": m_res["test_catastrophic_error_rate"],
            "test_catastrophic_errors": m_res["test_catastrophic_errors"],
            "eval_duration_sec": m_res["eval_duration_sec"]
        }
        all_test_metrics.append(comparison_row)
        
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    summary_df = pd.DataFrame(all_test_metrics)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "consolidated_test_evaluation_report.csv"), index=False)
    
    with open(os.path.join(OUTPUT_DIR, "consolidated_test_evaluation_report.json"), "w") as f:
        json.dump(all_test_metrics, f, indent=2)
        
    print("\n" + "=" * 80)
    print("  ALL MODULE 12 TEST EVALUATIONS COMPLETED SUCCESSFULLY!")
    print(f"  Summary saved to {OUTPUT_DIR}/consolidated_test_evaluation_report.csv")
    print("=" * 80)
    print(summary_df[["ablation_id", "val_qwk", "test_qwk", "qwk_generalization_gap", "val_accuracy", "test_accuracy", "test_catastrophic_error_rate"]])

if __name__ == "__main__":
    run_all_test_evaluations()
