import os
import sys
import shutil
import time
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

sys.path.append(os.path.abspath("."))

from src.models.fusion import VisualRadiomicFusionModel, SpatialCrossAttention
from src.data.data_pipeline import get_transforms, LABEL_MAPPING, create_stratified_splits
from src.features.radiomics import REGION_NAMES
from src.evaluation.metrics import compute_classification_metrics, save_evaluation_artifacts

OUTPUT_DIR = "outputs/Module_07_Visual_Radiomic_Cross_Attention_Fusion"
os.makedirs(OUTPUT_DIR, exist_ok=True)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

class MultiModalDDRDataset(Dataset):
    def __init__(
        self,
        manifest_df: pd.DataFrame,
        npz_path: str,
        images_root: str,
        transform=None
    ):
        self.manifest_df = manifest_df.reset_index(drop=True)
        self.images_root = images_root
        self.transform = transform
        
        data = np.load(npz_path)
        self.spatial_radiomics = data["X_spatial_norm"].astype(np.float32)
        self.labels = data["y"].astype(np.int64)
        self.image_ids = data["image_ids"]
        
        self.id_to_idx = {str(iid): idx for idx, iid in enumerate(self.image_ids)}

    def __len__(self):
        return len(self.manifest_df)

    def __getitem__(self, idx):
        row = self.manifest_df.iloc[idx]
        img_id = str(row["image_id"])
        label = int(row["numeric_label"])
        
        img_path = os.path.join(self.images_root, img_id)
        if not os.path.exists(img_path):
            img_path = os.path.join(self.images_root, f"{img_id}.jpg")
            
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
            
        npz_idx = self.id_to_idx.get(img_id, idx)
        rad_tokens = torch.from_numpy(self.spatial_radiomics[npz_idx])
        
        return {
            "image": image_tensor,
            "radiomics": rad_tokens,
            "label": torch.tensor(label, dtype=torch.long),
            "image_id": img_id
        }

def run_unit_verification(model: nn.Module) -> bool:
    print("\n--- [Phase 1/5] Running Multi-Modal Forward & Backward Verification Test ---")
    model.eval()
    dummy_img = torch.randn(2, 3, 224, 224, device=DEVICE)
    dummy_rad = torch.randn(2, 9, 32, device=DEVICE)
    
    logits = model(dummy_img, dummy_rad)
    print(f"  Forward Pass: Image {dummy_img.shape} + Radiomics {dummy_rad.shape} -> Logits {logits.shape}")
    assert logits.shape == (2, 5), f"Expected logits (2, 5), got {logits.shape}"
    assert not torch.isnan(logits).any(), "NaN in fusion logits!"
    
    logits, attn_weights = model.forward_with_attention(dummy_img, dummy_rad)
    print(f"  Cross-Attention: 49 Visual Tokens x 9 Radiomic Regions -> Map {attn_weights.shape}")
    assert attn_weights.shape == (2, 49, 9), f"Expected attn weights (2, 49, 9), got {attn_weights.shape}"
    
    model.train()
    dummy_labels = torch.tensor([0, 4], device=DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    
    out = model(dummy_img, dummy_rad)
    loss = criterion(out, dummy_labels)
    loss.backward()
    optimizer.step()
    
    print(f"  Backward Pass: Loss = {loss.item():.4f}")
    assert not torch.isnan(loss), "NaN in backward loss!"
    print("  [SUCCESS] Multi-Modal Verification test passed: All tensor shapes and gradients verified.\n")
    return True

def get_dataset_splits():
    train_npz = "outputs/Module_06_Spatially_Localized_Radiomics/radiomic_features_train.npz"
    val_npz = "outputs/Module_06_Spatially_Localized_Radiomics/radiomic_features_val.npz"
    train_data = np.load(train_npz)
    val_data = np.load(val_npz)
    
    train_df = pd.DataFrame({
        "image_id": train_data["image_ids"],
        "numeric_label": train_data["y"]
    })
    val_df = pd.DataFrame({
        "image_id": val_data["image_ids"],
        "numeric_label": val_data["y"]
    })
    
    sample_img_dir = "data_manifests/sample_dataset/images"
    real_img_dir = "data/ddr/DR_grading/DR_grading"
    sample_id = str(train_data["image_ids"][0])
    if os.path.exists(os.path.join(sample_img_dir, sample_id)) or os.path.exists(os.path.join(sample_img_dir, f"{sample_id}.jpg")):
        images_root = sample_img_dir
    else:
        images_root = real_img_dir
        
    print(f"[Dataset] Aligned with radiomics: Loaded {len(train_df)} train samples, {len(val_df)} val samples from {images_root}.")
    return train_df, val_df, images_root

def train_fusion_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 5,
    lr: float = 1e-4,
    weight_decay: float = 1e-2
):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    best_qwk = -1.0
    best_checkpoint_path = os.path.join(OUTPUT_DIR, "best_model.pth")
    history = []
    
    print(f"[Trainer] Training Multi-Modal Cross-Attention model on {DEVICE} for {epochs} epochs...")
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
    print(f"[Trainer] Saved training history -> {history_path}")
    
    model.load_state_dict(torch.load(best_checkpoint_path, map_location=DEVICE))
    return history_df, best_qwk

def visualize_cross_attention(
    model: nn.Module,
    val_loader: DataLoader
):
    print("\n--- [Phase 5/5] Visualizing Visual-Radiomic Cross-Attention Maps ---")
    model.eval()
    
    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["image"].to(DEVICE)
            rads = batch["radiomics"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            img_ids = batch["image_id"]
            
            logits, attn_weights = model.forward_with_attention(imgs, rads)
            break
            
    sample_attn = attn_weights[0].cpu().numpy()
    sample_img_id = img_ids[0]
    sample_label = labels[0].item()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    ax = axes[0]
    sns.heatmap(sample_attn, cmap="viridis", ax=ax, cbar=True)
    ax.set_title(f"Visual (49 tokens) -> Radiomic (9 regions) Attention Matrix\nSample: {sample_img_id} (Grade {sample_label})", fontsize=11, fontweight="bold")
    ax.set_xlabel("Anatomical Radiomic Regions (0 to 8)", fontsize=10)
    ax.set_ylabel("Visual Spatial Tokens (0 to 48)", fontsize=10)
    ax.set_xticks(np.arange(9) + 0.5)
    ax.set_xticklabels([r.replace("R", "") for r in REGION_NAMES], rotation=45, ha="right", fontsize=8)
    
    ax = axes[1]
    region_importance = sample_attn.mean(axis=0)
    region_grid = region_importance.reshape(3, 3)
    
    sns.heatmap(region_grid, annot=True, fmt=".3f", cmap="magma", ax=ax, cbar=True)
    ax.set_title("Anatomical Radiomic Attention Allocation Grid", fontsize=12, fontweight="bold")
    ax.set_xticklabels(["Temporal", "Center", "Nasal"])
    ax.set_yticklabels(["Superior", "Mid", "Inferior"])
    
    max_idx = int(np.argmax(region_importance))
    max_region = REGION_NAMES[max_idx]
    ax.set_xlabel(f"Peak Attention Region: {max_region} ({region_importance[max_idx]:.3f})", fontsize=11, fontweight="bold", color="darkred")
    
    plt.tight_layout()
    attn_vis_path = os.path.join(OUTPUT_DIR, "cross_attention_map.png")
    plt.savefig(attn_vis_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved cross-attention visualization -> {attn_vis_path}")

def main():
    t_start = time.time()
    print("==================================================================")
    print("  MODULE 07: SPATIAL VISUAL-RADIOMIC CROSS-ATTENTION FUSION")
    print(f"  Device: {DEVICE} (Apple Silicon MPS / CPU)")
    print("==================================================================")
    
    print("\n--- [Phase 2/5] Initializing Visual-Radiomic Fusion Model ---")
    model = VisualRadiomicFusionModel(
        num_classes=5,
        swin_checkpoint="outputs/Module_03_Baseline_Swin_Tiny/best_model.pth",
        d_rad=32,
        d_model=768,
        num_heads=8,
        dropout=0.2
    )
    model = model.to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Fusion Model: Total Parameters = {total_params:,} | Trainable = {trainable_params:,}")
    
    run_unit_verification(model)
    
    print("--- [Phase 3/5] Building Multi-Modal DataLoaders ---")
    train_df, val_df, images_root = get_dataset_splits()
    train_transform = get_transforms(is_training=True, img_size=224)
    val_transform = get_transforms(is_training=False, img_size=224)

    train_dataset = MultiModalDDRDataset(
        manifest_df=train_df,
        npz_path="outputs/Module_06_Spatially_Localized_Radiomics/radiomic_features_train.npz",
        images_root=images_root,
        transform=train_transform
    )
    val_dataset = MultiModalDDRDataset(
        manifest_df=val_df,
        npz_path="outputs/Module_06_Spatially_Localized_Radiomics/radiomic_features_val.npz",
        images_root=images_root,
        transform=val_transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    print("\n--- [Phase 4/5] Executing Multi-Modal Fusion Training ---")
    history_df, best_qwk = train_fusion_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=5,
        lr=1e-4,
        weight_decay=1e-2
    )
    
    model.eval()
    val_targets, val_preds, val_probs, val_ids = [], [], [], []
    val_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
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
    pred_df = pd.DataFrame(pred_records)
    pred_path = os.path.join(OUTPUT_DIR, "predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"Saved predictions -> {pred_path}")
    
    visualize_cross_attention(model, val_loader)
    
    shutil.copy("src/models/fusion.py", os.path.join(OUTPUT_DIR, "cross_attention_fusion.py"))
    
    total_time = time.time() - t_start
    print(f"\n==================================================================")
    print(f"  CROSS-ATTENTION FUSION TRAINING COMPLETED IN {total_time:.1f}s")
    print(f"  Final Validation QWK:              {final_metrics['quadratic_weighted_kappa']:.4f}")
    print(f"  Final Validation Macro-F1:         {final_metrics['macro_f1']:.4f}")
    print(f"  Final Validation Accuracy:         {final_metrics['accuracy']:.4f}")
    print(f"  Final Validation Balanced Acc:     {final_metrics['balanced_accuracy']:.4f}")
    print(f"  Final Validation MAE:              {final_metrics['mean_absolute_error']:.4f}")
    print(f"==================================================================")
    print(f"[SUCCESS] Module 07 deliverables saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
