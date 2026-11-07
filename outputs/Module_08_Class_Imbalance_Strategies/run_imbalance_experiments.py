import os
import sys
import shutil
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

sys.path.append(os.path.abspath("."))

from src.models.fusion import VisualRadiomicFusionModel
from src.training.losses import (
    compute_effective_number_weights,
    build_loss_function,
    ClassBalancedFocalLoss,
    FocalLoss,
    CostSensitiveLoss
)
from src.data.data_pipeline import get_transforms, LABEL_MAPPING
from src.evaluation.metrics import compute_classification_metrics, save_evaluation_artifacts

OUTPUT_DIR = "outputs/Module_08_Class_Imbalance_Strategies"
os.makedirs(OUTPUT_DIR, exist_ok=True)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

class MultiModalDDRDataset(Dataset):
    def __init__(self, npz_path: str, images_root: str, transform=None):
        self.images_root = images_root
        self.transform = transform
        
        data = np.load(npz_path)
        self.spatial_radiomics = data["X_spatial_norm"].astype(np.float32)
        self.labels = data["y"].astype(np.int64)
        self.image_ids = data["image_ids"]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = str(self.image_ids[idx])
        label = int(self.labels[idx])
        
        img_path = os.path.join(self.images_root, img_id)
        if not os.path.exists(img_path):
            img_path = os.path.join(self.images_root, f"{img_id}.jpg")
            
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
            
        rad_tokens = torch.from_numpy(self.spatial_radiomics[idx])
        
        return {
            "image": image_tensor,
            "radiomics": rad_tokens,
            "label": torch.tensor(label, dtype=torch.long),
            "image_id": img_id
        }

def run_unit_verification(class_counts: list) -> bool:
    print("\n--- [Phase 1/5] Running Loss Function Verification Tests ---")
    dummy_logits = torch.randn(4, 5, requires_grad=True, device=DEVICE)
    dummy_targets = torch.tensor([0, 1, 3, 4], device=DEVICE)
    
    w = compute_effective_number_weights(class_counts, beta=0.9999)
    print(f"  Effective Number Weights: {w.tolist()}")
    assert len(w) == 5, "Expected 5 class weights"
    
    fl = FocalLoss(gamma=2.0)
    loss1 = fl(dummy_logits, dummy_targets)
    loss1.backward(retain_graph=True)
    print(f"  Focal Loss: {loss1.item():.4f} - Backward passed")
    assert not torch.isnan(loss1)
    
    cb_fl = ClassBalancedFocalLoss(class_counts, beta=0.9999, gamma=2.0)
    loss2 = cb_fl(dummy_logits, dummy_targets)
    loss2.backward(retain_graph=True)
    print(f"  CB-Focal Loss: {loss2.item():.4f} - Backward passed")
    assert not torch.isnan(loss2)
    
    cs = CostSensitiveLoss(num_classes=5)
    loss3 = cs(dummy_logits, dummy_targets)
    loss3.backward()
    print(f"  Cost-Sensitive Loss: {loss3.item():.4f} - Backward passed")
    assert not torch.isnan(loss3)
    
    print("  [SUCCESS] All imbalance loss functions verified.\n")
    return True

def train_single_strategy(
    strategy_name: str,
    criterion: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 5
) -> dict:
    print(f"\n>> Evaluating Strategy: {strategy_name}")
    model = VisualRadiomicFusionModel(
        num_classes=5,
        swin_checkpoint="outputs/Module_03_Baseline_Swin_Tiny/best_model.pth"
    ).to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    best_qwk = -1.0
    best_metrics = {}
    best_preds, best_probs = [], []
    history = []
    
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for batch in train_loader:
            imgs = batch["image"].to(DEVICE)
            rads = batch["radiomics"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(imgs, rads)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            
            train_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=-1)
            train_correct += (preds == labels).sum().item()
            train_total += imgs.size(0)
            
        scheduler.step()
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        model.eval()
        val_loss, val_targets, val_preds, val_probs = 0.0, [], [], []
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(DEVICE)
                rads = batch["radiomics"].to(DEVICE)
                labels = batch["label"].to(DEVICE)
                
                logits = model(imgs, rads)
                loss = criterion(logits, labels)
                val_loss += loss.item() * imgs.size(0)
                probs = torch.softmax(logits, dim=-1)
                
                val_targets.extend(labels.cpu().numpy())
                val_preds.extend(logits.argmax(dim=-1).cpu().numpy())
                val_probs.extend(probs.cpu().numpy())
                
        val_loss /= len(val_targets)
        val_targets = np.array(val_targets)
        val_preds = np.array(val_preds)
        val_probs = np.array(val_probs)
        
        metrics, cm, clf_dict = compute_classification_metrics(val_targets, val_preds)
        metrics["loss"] = val_loss
        epoch_time = time.time() - t0
        
        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(metrics["accuracy"], 4),
            "val_qwk": round(metrics["quadratic_weighted_kappa"], 4),
            "val_macro_f1": round(metrics["macro_f1"], 4),
            "val_mae": round(metrics["mean_absolute_error"], 4),
            "epoch_time_sec": round(epoch_time, 2)
        })
        
        if metrics["quadratic_weighted_kappa"] >= best_qwk:
            best_qwk = metrics["quadratic_weighted_kappa"]
            best_metrics = metrics
            best_cm = cm
            best_clf_dict = clf_dict
            best_preds = val_preds
            best_probs = val_probs
            best_state = model.state_dict().copy()
            
        print(f"  Epoch {epoch:02d} | Val Loss: {val_loss:.4f} Acc: {metrics['accuracy']:.3f} "
              f"QWK: {metrics['quadratic_weighted_kappa']:.4f} F1: {metrics['macro_f1']:.4f}")
        
    return {
        "name": strategy_name,
        "metrics": best_metrics,
        "cm": best_cm,
        "clf_dict": best_clf_dict,
        "history": history,
        "best_state": best_state,
        "preds": best_preds,
        "probs": best_probs,
        "val_targets": val_targets
    }

def main():
    t_start = time.time()
    print("==================================================================")
    print("  MODULE 08: CLASS IMBALANCE STRATEGIES EVALUATION")
    print(f"  Device: {DEVICE} (Apple Silicon MPS / CPU)")
    print("==================================================================")
    
    ddr_counts = [6266, 630, 4477, 236, 913]
    
    run_unit_verification(ddr_counts)
    
    eff_weights = compute_effective_number_weights(ddr_counts, beta=0.9999).tolist()
    inv_weights = (1.0 / (np.array(ddr_counts, dtype=np.float32) + 1e-5))
    inv_weights = (inv_weights / np.sum(inv_weights) * 5.0).tolist()
    
    weights_metadata = {
        "dataset": "DDR (Official 12,522 records)",
        "class_counts": {i: ddr_counts[i] for i in range(5)},
        "effective_number_weights_beta_0_9999": {i: round(eff_weights[i], 4) for i in range(5)},
        "inverse_frequency_weights": {i: round(inv_weights[i], 4) for i in range(5)},
        "minority_classes": [1, 3],
        "grade_3_boost_ratio": round(eff_weights[3] / eff_weights[0], 2)
    }
    weights_path = os.path.join(OUTPUT_DIR, "effective_weights.json")
    with open(weights_path, "w") as f:
        json.dump(weights_metadata, f, indent=4)
    print("Saved effective weights metadata ->", weights_path)
    
    print("\n--- [Phase 2/5] Building Multi-Modal DataLoaders ---")
    sample_img_dir = "data_manifests/sample_dataset/images"
    real_img_dir = "data/ddr/DR_grading/DR_grading"
    
    train_npz = "outputs/Module_06_Spatially_Localized_Radiomics/radiomic_features_train.npz"
    val_npz = "outputs/Module_06_Spatially_Localized_Radiomics/radiomic_features_val.npz"
    train_data = np.load(train_npz)
    
    sample_id = str(train_data["image_ids"][0])
    if os.path.exists(os.path.join(sample_img_dir, sample_id)) or os.path.exists(os.path.join(sample_img_dir, f"{sample_id}.jpg")):
        images_root = sample_img_dir
    else:
        images_root = real_img_dir
        
    train_transform = get_transforms(is_training=True, img_size=224)
    val_transform = get_transforms(is_training=False, img_size=224)
    
    train_ds = MultiModalDDRDataset(train_npz, images_root, transform=train_transform)
    val_ds = MultiModalDDRDataset(val_npz, images_root, transform=val_transform)
    
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)
    
    strategies = [
        ("Cross-Entropy (Baseline)", nn.CrossEntropyLoss()),
        ("Weighted Cross-Entropy", nn.CrossEntropyLoss(weight=torch.tensor(eff_weights, dtype=torch.float32, device=DEVICE))),
        ("Focal Loss (gamma=2.0)", FocalLoss(gamma=2.0)),
        ("Class-Balanced Focal Loss", ClassBalancedFocalLoss(ddr_counts, beta=0.9999, gamma=2.0).to(DEVICE))
    ]
    
    print("\n--- [Phase 3/5] Benchmarking Imbalance Loss Formulations ---")
    results = []
    for name, criterion in strategies:
        res = train_single_strategy(name, criterion, train_loader, val_loader, epochs=5)
        results.append(res)
        
    records = []
    for r in results:
        m = r["metrics"]
        records.append({
            "Strategy": r["name"],
            "Validation QWK": round(m["quadratic_weighted_kappa"], 4),
            "Validation Macro-F1": round(m["macro_f1"], 4),
            "Validation Accuracy": round(m["accuracy"], 4),
            "Validation Balanced Acc": round(m["balanced_accuracy"], 4),
            "Validation MAE": round(m["mean_absolute_error"], 4),
            "Validation Loss": round(m["loss"], 4)
        })
    comp_df = pd.DataFrame(records)
    comp_df = comp_df.sort_values(by="Validation QWK", ascending=False).reset_index(drop=True)
    comp_path = os.path.join(OUTPUT_DIR, "imbalance_comparison.csv")
    comp_df.to_csv(comp_path, index=False)
    print("\n[Comparison Summary]")
    print(comp_df.to_string())
    
    winner_name = comp_df.iloc[0]["Strategy"]
    winner = [r for r in results if r["name"] == winner_name][0]
    print(f"\n[Model Selection] Winning Imbalance Strategy: {winner_name}")
    
    torch.save(winner["best_state"], os.path.join(OUTPUT_DIR, "best_model.pth"))

    save_evaluation_artifacts(winner["metrics"], winner["cm"], winner["clf_dict"], OUTPUT_DIR)
    
    win_hist_df = pd.DataFrame(winner["history"])
    win_hist_df.to_csv(os.path.join(OUTPUT_DIR, "training_history.csv"), index=False)
    
    val_ids = np.load(val_npz)["image_ids"]
    pred_records = []
    for i in range(len(val_ids)):
        pred_records.append({
            "image_id": val_ids[i],
            "ground_truth": int(winner["val_targets"][i]),
            "ground_truth_label": LABEL_MAPPING.get(int(winner["val_targets"][i]), "Unknown"),
            "predicted_grade": int(winner["preds"][i]),
            "predicted_label": LABEL_MAPPING.get(int(winner["preds"][i]), "Unknown"),
            "prob_grade_0": round(float(winner["probs"][i, 0]), 4),
            "prob_grade_1": round(float(winner["probs"][i, 1]), 4),
            "prob_grade_2": round(float(winner["probs"][i, 2]), 4),
            "prob_grade_3": round(float(winner["probs"][i, 3]), 4),
            "prob_grade_4": round(float(winner["probs"][i, 4]), 4)
        })
    pd.DataFrame(pred_records).to_csv(os.path.join(OUTPUT_DIR, "predictions.csv"), index=False)
    
    print("\n--- [Phase 4/5] Generating Publication Visualizations ---")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    x = np.arange(len(comp_df))
    width = 0.35
    axes[0].bar(x - width/2, comp_df["Validation QWK"], width, label="Validation QWK", color="#2ca02c", edgecolor="black")
    axes[0].bar(x + width/2, comp_df["Validation Macro-F1"], width, label="Validation Macro-F1", color="#1f77b4", edgecolor="black")
    axes[0].set_title("Loss Strategy Comparison: QWK & Macro-F1", fontsize=12, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(comp_df["Strategy"], rotation=20, ha="right", fontsize=9)
    axes[0].set_ylabel("Metric Value", fontsize=11)
    axes[0].set_ylim(0, 1.15)
    axes[0].legend(loc="lower right")
    
    axes[1].bar(
        [f"Grade {i}\n({ddr_counts[i]} cases)" for i in range(5)],
        eff_weights,
        color=["#7f7f7f", "#ff7f0e", "#7f7f7f", "#d62728", "#1f77b4"],
        edgecolor="black"
    )
    axes[1].set_title("Effective Number Class Weights (beta=0.9999)\nMinority Grades 1 & 3 receive substantial boosts", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Class Weight (W_c)", fontsize=11)
    for i, w_val in enumerate(eff_weights):
        axes[1].text(i, w_val + 0.08, f"{w_val:.2f}x", ha='center', fontweight='bold', fontsize=10)
        
    plt.tight_layout()
    chart_path = os.path.join(OUTPUT_DIR, "imbalance_comparison.png")
    plt.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved comparison chart ->", chart_path)
    
    shutil.copy("src/training/losses.py", os.path.join(OUTPUT_DIR, "imbalance_strategies.py"))
    
    total_time = time.time() - t_start
    print(f"\n==================================================================")
    print(f"  MODULE 08 COMPLETED IN {total_time:.1f}s")
    print(f"  Winning Loss Strategy: {winner_name}")

    print(f"  Primary Metric (QWK):  {winner['metrics']['quadratic_weighted_kappa']:.4f}")
    print(f"  Secondary (Macro-F1):  {winner['metrics']['macro_f1']:.4f}")
    print(f"  Validation Loss:       {winner['metrics']['loss']:.4f}")
    print(f"==================================================================")

if __name__ == "__main__":
    main()
