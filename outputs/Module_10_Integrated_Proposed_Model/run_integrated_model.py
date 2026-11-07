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

from src.models.integrated_model import IntegratedRetinopathyModel
from src.models.coral import coral_predict, coral_probabilities
from src.data.data_pipeline import get_transforms, LABEL_MAPPING
from src.evaluation.metrics import compute_classification_metrics, save_evaluation_artifacts

OUTPUT_DIR = "outputs/Module_10_Integrated_Proposed_Model"
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

def run_unit_verification(model: IntegratedRetinopathyModel) -> bool:
    print("\n--- [Phase 1/5] Running Integrated Model Verification Tests ---")
    model.eval()
    dummy_img = torch.randn(2, 3, 224, 224, device=DEVICE)
    dummy_rad = torch.randn(2, 9, 32, device=DEVICE)
    dummy_labels = torch.tensor([0, 4], device=DEVICE)
    
    rank_logits, aux_logits = model(dummy_img, dummy_rad, return_aux=True)
    print(f"  Forward Pass: Image + Radiomics -> Rank Logits {rank_logits.shape}, Aux Logits {aux_logits.shape}")
    assert rank_logits.shape == (2, 4), f"Expected rank logits (2, 4), got {rank_logits.shape}"
    assert aux_logits.shape == (2, 5), f"Expected aux logits (2, 5), got {aux_logits.shape}"
    
    pred_dict = model.predict(dummy_img, dummy_rad)
    print(f"  Predictions: {pred_dict['predicted_grade'].tolist()} | Probabilities: {pred_dict['class_probabilities'].shape}")
    assert pred_dict["class_probabilities"].shape == (2, 5)
    assert pred_dict["attention_weights"].shape == (2, 49, 9)
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    
    r_log, a_log = model(dummy_img, dummy_rad, return_aux=True)
    total_loss, l_coral, l_aux = model.compute_hybrid_loss(r_log, a_log, dummy_labels, lambda_aux=0.3)
    total_loss.backward()
    optimizer.step()
    
    print(f"  Hybrid Loss: Total={total_loss.item():.4f} (CORAL={l_coral.item():.4f}, Aux={l_aux.item():.4f}) - Backward passed.")
    assert not torch.isnan(total_loss), "NaN in hybrid loss!"
    print("  [SUCCESS] Integrated model verification test passed: Dual heads and hybrid supervision verified.\n")
    return True

def train_integrated_model(
    model: IntegratedRetinopathyModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 5,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    lambda_aux: float = 0.3
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    best_qwk = -1.0
    best_checkpoint_path = os.path.join(OUTPUT_DIR, "best_model.pth")
    history = []
    
    print(f"[Trainer] Training Integrated Model on {DEVICE} for {epochs} epochs (lambda_aux={lambda_aux})...")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for batch in train_loader:
            imgs = batch["image"].to(DEVICE)
            rads = batch["radiomics"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            
            optimizer.zero_grad()
            rank_logits, aux_logits = model(imgs, rads, return_aux=True)
            loss, _, _ = model.compute_hybrid_loss(rank_logits, aux_logits, labels, lambda_aux=lambda_aux)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            
            train_loss += loss.item() * imgs.size(0)
            preds = coral_predict(rank_logits)
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
                
                rank_logits, aux_logits = model(imgs, rads, return_aux=True)
                loss, _, _ = model.compute_hybrid_loss(rank_logits, aux_logits, labels, lambda_aux=lambda_aux)
                val_loss += loss.item() * imgs.size(0)
                
                preds = coral_predict(rank_logits)
                probs = coral_probabilities(rank_logits)
                
                val_targets.extend(labels.cpu().numpy())
                val_preds.extend(preds.cpu().numpy())
                val_probs.extend(probs.cpu().numpy())
                
        val_loss /= len(val_targets)
        val_targets = np.array(val_targets)
        val_preds = np.array(val_preds)
        val_probs = np.array(val_probs)
        
        metrics, cm, clf_dict = compute_classification_metrics(val_targets, val_preds)
        epoch_time = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]
        
        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(metrics["accuracy"], 4),
            "val_qwk": round(metrics["quadratic_weighted_kappa"], 4),
            "val_macro_f1": round(metrics["macro_f1"], 4),
            "val_mae": round(metrics["mean_absolute_error"], 4),
            "lr": round(current_lr, 6),
            "epoch_time_sec": round(epoch_time, 2)
        })
        
        print(f"Epoch {epoch:02d}/{epochs:02d} [{epoch_time:.1f}s] | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
              f"Val Loss: {val_loss:.4f} Acc: {metrics['accuracy']:.3f} "
              f"QWK: {metrics['quadratic_weighted_kappa']:.4f} "
              f"F1: {metrics['macro_f1']:.4f} MAE: {metrics['mean_absolute_error']:.3f}")
        
        if metrics["quadratic_weighted_kappa"] >= best_qwk:
            best_qwk = metrics["quadratic_weighted_kappa"]
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"  --> Saved new best checkpoint to {best_checkpoint_path} (QWK: {best_qwk:.4f})")
            
    history_df = pd.DataFrame(history)
    history_path = os.path.join(OUTPUT_DIR, "training_history.csv")
    history_df.to_csv(history_path, index=False)
    
    model.load_state_dict(torch.load(best_checkpoint_path, map_location=DEVICE))
    return history_df, best_qwk

def generate_architecture_diagram():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    ax.text(0.5, 0.95, "Integrated Proposed Architecture: Spatial Visual-Radiomic CORAL Network",
            ha='center', va='center', fontsize=15, fontweight='bold')
    
    box_blue = dict(boxstyle="round,pad=0.5", fc="#e1f5fe", ec="#0288d1", lw=2)
    box_green = dict(boxstyle="round,pad=0.5", fc="#e8f8f5", ec="#27ae60", lw=2)
    box_orange = dict(boxstyle="round,pad=0.5", fc="#fff3e0", ec="#f57c00", lw=2)
    box_red = dict(boxstyle="round,pad=0.5", fc="#fbe9e7", ec="#d84315", lw=2)
    box_purple = dict(boxstyle="round,pad=0.5", fc="#f3e5f5", ec="#8e24aa", lw=2)
    
    ax.text(0.18, 0.75, "Fundus Image\n(B, 3, 224, 224)", ha='center', va='center', bbox=box_blue, fontsize=10, fontweight='bold')
    ax.text(0.82, 0.75, "Fundus Sub-Regions\n(3x3 Anatomical Grid)", ha='center', va='center', bbox=box_green, fontsize=10, fontweight='bold')
    
    ax.text(0.18, 0.55, "Swin-Tiny Backbone\n(Hierarchical ViT)\n-> 49 Visual Tokens (d=768)", ha='center', va='center', bbox=box_blue, fontsize=10)
    ax.text(0.82, 0.55, "Spatial Radiomics Extractor\n(GLCM + LBP + First-Order + Morph)\n-> 9 Regional Tokens (d=32)", ha='center', va='center', bbox=box_green, fontsize=10)
    
    ax.text(0.5, 0.40, "Spatial Multi-Head Cross-Attention Fusion (8 Heads, d=768)\nQueries: Visual Tokens (B, 49, 768) | Keys & Values: Radiomic Tokens (B, 9, 768)\nOutput: Contextualized Visual-Radiomic Tokens (B, 49, 768)",
            ha='center', va='center', bbox=box_orange, fontsize=11, fontweight='bold')
    
    ax.text(0.5, 0.25, "Global Contextual Average Pooling -> Feature Vector (B, 768)",
            ha='center', va='center', bbox=box_purple, fontsize=10, fontweight='bold')
    
    ax.text(0.28, 0.10, "Primary Head: CORAL Ordinal Layer\nShared Weight w in R^{768} + 4 Ordered Biases\n-> 4 Monotonic Rank Decisions P(y > k)",
            ha='center', va='center', bbox=box_red, fontsize=10, fontweight='bold')
    ax.text(0.72, 0.10, "Auxiliary Head: Softmax Classifier\nLinear(768 -> 5 Classes)\n-> Hybrid Supervision L_total = L_CORAL + 0.3*L_CE",
            ha='center', va='center', bbox=box_red, fontsize=10)
    
    arrows = [
        ((0.18, 0.70), (0.18, 0.60)),
        ((0.82, 0.70), (0.82, 0.60)),
        ((0.25, 0.50), (0.40, 0.43)),
        ((0.75, 0.50), (0.60, 0.43)),
        ((0.50, 0.35), (0.50, 0.28)),
        ((0.45, 0.21), (0.33, 0.14)),
        ((0.55, 0.21), (0.67, 0.14))
    ]
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start, arrowprops=dict(facecolor='black', edgecolor='black', width=1.5, headwidth=8))
        
    diag_path = os.path.join(OUTPUT_DIR, "model_architecture.png")
    plt.savefig(diag_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved architecture diagram ->", diag_path)

def main():
    t_start = time.time()
    print("==================================================================")
    print("  MODULE 10: INTEGRATED PROPOSED MODEL")
    print(f"  Device: {DEVICE} (Apple Silicon MPS / CPU)")
    print("==================================================================")
    
    print("\n--- [Phase 2/5] Initializing Integrated Retinopathy Network ---")
    model = IntegratedRetinopathyModel(
        num_classes=5,
        d_rad=32,
        d_model=768,
        num_heads=8,
        dropout=0.2,
        pretrained_checkpoint="outputs/Module_07_Visual_Radiomic_Cross_Attention_Fusion/best_model.pth"
    ).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Integrated Model: Total Parameters = {total_params:,} | Trainable = {trainable_params:,}")
    
    run_unit_verification(model)
    
    print("--- [Phase 3/5] Building Multi-Modal DataLoaders ---")
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
    
    print("\n--- [Phase 4/5] Executing Integrated Model Training ---")
    history_df, best_qwk = train_integrated_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=5,
        lr=1e-4,
        weight_decay=1e-2,
        lambda_aux=0.3
    )
    
    model.eval()
    val_targets, val_preds, val_probs, val_ids = [], [], [], []
    val_loss = 0.0
    
    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["image"].to(DEVICE)
            rads = batch["radiomics"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            
            rank_logits, aux_logits = model(imgs, rads, return_aux=True)
            loss, _, _ = model.compute_hybrid_loss(rank_logits, aux_logits, labels, lambda_aux=0.3)
            val_loss += loss.item() * imgs.size(0)
            
            preds = coral_predict(rank_logits)
            probs = coral_probabilities(rank_logits)
            
            val_targets.extend(labels.cpu().numpy())
            val_preds.extend(preds.cpu().numpy())
            val_probs.extend(probs.cpu().numpy())
            val_ids.extend(batch["image_id"])
            
    val_loss /= len(val_targets)
    val_targets = np.array(val_targets)
    val_preds = np.array(val_preds)
    val_probs = np.array(val_probs)
    
    final_metrics, cm, clf_dict = compute_classification_metrics(val_targets, val_preds)
    final_metrics["loss"] = val_loss
    
    save_evaluation_artifacts(final_metrics, cm, clf_dict, OUTPUT_DIR)
    
    pred_records = []
    for i in range(len(val_ids)):
        pred_records.append({
            "image_id": val_ids[i],
            "ground_truth": int(val_targets[i]),
            "ground_truth_label": LABEL_MAPPING.get(int(val_targets[i]), "Unknown"),
            "predicted_grade": int(val_preds[i]),
            "predicted_label": LABEL_MAPPING.get(int(val_preds[i]), "Unknown"),
            "prob_grade_0": round(float(val_probs[i, 0]), 4),
            "prob_grade_1": round(float(val_probs[i, 1]), 4),
            "prob_grade_2": round(float(val_probs[i, 2]), 4),
            "prob_grade_3": round(float(val_probs[i, 3]), 4),
            "prob_grade_4": round(float(val_probs[i, 4]), 4)
        })
    pd.DataFrame(pred_records).to_csv(os.path.join(OUTPUT_DIR, "predictions.csv"), index=False)
    print("Saved predictions ->", os.path.join(OUTPUT_DIR, "predictions.csv"))
    
    generate_architecture_diagram()
    
    shutil.copy("src/models/integrated_model.py", os.path.join(OUTPUT_DIR, "integrated_model.py"))
    
    total_time = time.time() - t_start
    print(f"\n==================================================================")
    print(f"  MODULE 10 COMPLETED IN {total_time:.1f}s")
    print(f"  Final Validation QWK:              {final_metrics['quadratic_weighted_kappa']:.4f}")
    print(f"  Final Validation Macro-F1:         {final_metrics['macro_f1']:.4f}")
    print(f"  Final Validation Accuracy:         {final_metrics['accuracy']:.4f}")
    print(f"  Final Validation Balanced Acc:     {final_metrics['balanced_accuracy']:.4f}")
    print(f"  Final Validation MAE:              {final_metrics['mean_absolute_error']:.4f}")
    print(f"  Final Validation Hybrid Loss:      {final_metrics['loss']:.4f}")
    print(f"==================================================================")
    print(f"[SUCCESS] Module 10 deliverables saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
