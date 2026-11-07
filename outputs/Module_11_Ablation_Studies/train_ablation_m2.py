import os
import sys
import time
import json
import numpy as np
import pandas as pd
from PIL import Image
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.models as tv_models
from torchvision import transforms
from sklearn.metrics import (
    cohen_kappa_score, f1_score, accuracy_score,
    recall_score, precision_score, confusion_matrix, classification_report,
    mean_absolute_error
)

sys.path.append(os.path.abspath("."))
from src.data.data_pipeline import retina_circle_crop, apply_clahe_lab, IMAGENET_MEAN, IMAGENET_STD
from src.models.coral import CoralLayer, coral_loss, coral_predict, coral_probabilities

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
OUTPUT_DIR = "outputs/Module_11_Ablation_Studies/M2_Global_Radiomics"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class StandardRetinaTransform:
    def __init__(self, is_training: bool = False, img_size: int = 224):
        self.is_training = is_training
        if is_training:
            self.spatial_transforms = transforms.Compose([
                transforms.Resize((img_size + 32, img_size + 32)),
                transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
        else:
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

class MultiModalDDRDataset(Dataset):
    def __init__(self, manifest_csv: str, cache_npz: str, transform=None):
        self.df = pd.read_csv(manifest_csv).reset_index(drop=True)
        self.transform = transform
        self.filepaths = self.df["filepath"].tolist()
        self.labels = self.df["numeric_label"].tolist()
        self.image_ids = self.df["image_id"].tolist()
        
        cache_data = np.load(cache_npz)
        self.radiomics = cache_data["X_spatial_norm"].astype(np.float32)
        assert len(self.df) == len(self.radiomics), "Mismatch between manifest and radiomics cache"

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
            "image_id": img_id
        }

class GlobalRadiomicsCrossAttention(nn.Module):
    def __init__(self, d_vis: int = 768, d_rad_global: int = 288, d_model: int = 768, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_vis = d_vis
        self.d_rad_global = d_rad_global
        self.d_model = d_model
        
        self.proj_q = nn.Linear(d_vis, d_model)
        self.proj_k = nn.Linear(d_rad_global, d_model)
        self.proj_v = nn.Linear(d_rad_global, d_model)
        
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, vis_tokens: torch.Tensor, rad_global: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.proj_q(vis_tokens)
        k = self.proj_k(rad_global)
        v = self.proj_v(rad_global)
        
        attn_out, attn_weights = self.mha(q, k, v, need_weights=True)
        x = self.norm1(vis_tokens + self.dropout(attn_out))
        out = self.norm2(x + self.dropout(self.ffn(x)))
        return out, attn_weights

class M2_GlobalRadiomicsModel(nn.Module):
    def __init__(self, num_classes: int = 5, d_vis: int = 768, d_rad_global: int = 288, d_model: int = 768, num_heads: int = 8, dropout: float = 0.2):
        super().__init__()
        self.num_classes = num_classes
        self.d_model = d_model
        
        base = tv_models.swin_t(weights=None)
        cache_dir = os.path.expanduser("~/.cache/torch/hub/checkpoints")
        ckpt_path = os.path.join(cache_dir, "swin_t-704ceda3.pth")
        if os.path.exists(ckpt_path):
            st = torch.load(ckpt_path, map_location="cpu")
            base.load_state_dict(st, strict=False)
            print(f"[M2 Model] Loaded ImageNet-1k Swin-Tiny weights: {ckpt_path}")
        else:
            print("[M2 Model] Warning: Cached weights not found, using default initialization.")
        base.head = nn.Identity()
        self.backbone = base
        
        self.cross_attention = GlobalRadiomicsCrossAttention(
            d_vis=d_vis,
            d_rad_global=d_rad_global,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.dropout = nn.Dropout(dropout)
        self.coral_head = CoralLayer(in_features=d_model, num_classes=num_classes)

    def forward(self, images: torch.Tensor, radiomics: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.features(images)
        feats = self.backbone.norm(feats)
        vis_tokens = feats.flatten(1, 2)
        
        rad_global = radiomics.reshape(radiomics.size(0), 1, -1)
        
        fused_tokens, _ = self.cross_attention(vis_tokens, rad_global)
        
        pooled = fused_tokens.mean(dim=1)
        pooled = self.dropout(pooled)
        
        rank_logits = self.coral_head(pooled)
        return rank_logits

def verification_test():
    print("=== UNIT VERIFICATION: M2 Global Radiomics ===")
    model = M2_GlobalRadiomicsModel().to(DEVICE)
    model.train()
    dummy_imgs = torch.randn(4, 3, 224, 224, device=DEVICE)
    dummy_rads = torch.randn(4, 9, 32, device=DEVICE)
    dummy_labels = torch.tensor([0, 1, 2, 3], device=DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    
    logits = model(dummy_imgs, dummy_rads)
    loss = coral_loss(logits, dummy_labels, num_classes=5)
    loss.backward()
    
    assert logits.shape == (4, 4), f"Unexpected logits shape: {logits.shape}"
    assert torch.isfinite(loss), "Verification test loss is NaN or Inf"
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Gradient is None for {name}"
            assert torch.isfinite(p.grad).all(), f"Gradient has NaN/Inf in {name}"
            
    print(f">> Verification test PASSED: Logits shape={logits.shape}, Loss={loss.item():.4f}, Gradients all finite.")
    del model, dummy_imgs, dummy_rads, dummy_labels, logits, loss
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

def train_m2(epochs: int = 5, batch_size: int = 32, lr: float = 1e-4, seed: int = 42):
    print("=" * 70)
    print("  TRAINING ABLATION M2: SPATIAL LOCALIZATION (GLOBAL RADIOMICS)")
    print(f"  Device: {DEVICE} | Batch Size: {batch_size} | Epochs: {epochs} | Seed: {seed}")
    print(f"  Output Directory: {OUTPUT_DIR}")
    print("=" * 70)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    train_manifest = "data_manifests/split_80_10_10/train_manifest.csv"
    val_manifest = "data_manifests/split_80_10_10/val_manifest.csv"
    train_cache = "outputs/gap3_spatial_radiomics_cross_attention/cache/radiomics_train.npz"
    val_cache = "outputs/gap3_spatial_radiomics_cross_attention/cache/radiomics_val.npz"
    
    train_transform = StandardRetinaTransform(is_training=True, img_size=224)
    val_transform = StandardRetinaTransform(is_training=False, img_size=224)
    
    train_ds = MultiModalDDRDataset(train_manifest, train_cache, transform=train_transform)
    val_ds = MultiModalDDRDataset(val_manifest, val_cache, transform=val_transform)
    
    train_labels = np.array(train_ds.labels)
    class_counts = np.bincount(train_labels, minlength=5)
    class_weights = 1.0 / (class_counts + 1e-5)
    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    model = M2_GlobalRadiomicsModel().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[M2 Parameters] Total: {total_params:,} ({total_params/1e6:.2f}M) | Trainable: {trainable_params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    best_qwk = -1.0
    best_epoch = 0
    history = []
    t_start = time.time()
    
    for epoch in range(1, epochs + 1):
        t_ep_start = time.time()
        model.train()
        train_loss, train_samples = 0.0, 0
        
        for b_idx, batch in enumerate(train_loader):
            imgs = batch["image"].to(DEVICE)
            rads = batch["radiomics"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            
            optimizer.zero_grad()
            rank_logits = model(imgs, rads)
            loss = coral_loss(rank_logits, labels, num_classes=5)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            
            train_loss += loss.item() * imgs.size(0)
            train_samples += imgs.size(0)
            
            if (b_idx + 1) % 50 == 0 or (b_idx + 1) == len(train_loader):
                print(f"  Epoch {epoch:02d} [{b_idx + 1:03d}/{len(train_loader):03d}] Loss: {loss.item():.4f}")
                
        scheduler.step()
        epoch_train_loss = train_loss / max(1, train_samples)
        
        model.eval()
        val_preds, val_targets, val_probs = [], [], []
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(DEVICE)
                rads = batch["radiomics"].to(DEVICE)
                labels = batch["label"].to(DEVICE)
                
                logits = model(imgs, rads)
                loss_v = coral_loss(logits, labels, num_classes=5)
                val_loss += loss_v.item() * imgs.size(0)
                
                preds = coral_predict(logits, threshold=0.5)
                probs = coral_probabilities(logits)
                
                val_preds.extend(preds.cpu().numpy().tolist())
                val_targets.extend(labels.cpu().numpy().tolist())
                val_probs.extend(probs.cpu().numpy().tolist())
                
        y_true = np.array(val_targets)
        y_pred = np.array(val_preds)
        probs_arr = np.array(val_probs)
        
        epoch_val_loss = val_loss / max(1, len(val_ds))
        acc = accuracy_score(y_true, y_pred)
        qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        mae = mean_absolute_error(y_true, y_pred)
        adj_err = np.mean(np.abs(y_true - y_pred) == 1)
        cat_err = np.mean(np.abs(y_true - y_pred) >= 2)
        
        ep_duration = time.time() - t_ep_start
        
        ep_record = {
            "epoch": epoch,
            "train_loss": round(float(epoch_train_loss), 4),
            "val_loss": round(float(epoch_val_loss), 4),
            "val_acc": round(float(acc), 4),
            "val_qwk": round(float(qwk), 4),
            "val_macro_f1": round(float(macro_f1), 4),
            "val_weighted_f1": round(float(weighted_f1), 4),
            "val_mae": round(float(mae), 4),
            "adjacent_error_rate": round(float(adj_err), 4),
            "catastrophic_error_rate": round(float(cat_err), 4),
            "duration_sec": round(float(ep_duration), 1)
        }
        history.append(ep_record)
        
        print(f"Epoch {epoch:02d} Summary: Val QWK={qwk:.4f} | Acc={acc*100:.2f}% | Macro-F1={macro_f1:.4f} | MAE={mae:.4f} | AdjErr={adj_err*100:.2f}% | CatErr={cat_err*100:.2f}% ({ep_duration:.1f}s)")
        
        if qwk > best_qwk:
            best_qwk = qwk
            best_epoch = epoch
            print(f"  >>> New Best QWK: {best_qwk:.4f} at Epoch {epoch}! Saving checkpoint...")
            
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_qwk": qwk,
                "val_acc": acc,
                "val_macro_f1": macro_f1,
                "val_metrics": ep_record
            }, os.path.join(OUTPUT_DIR, "best_model.pth"))
            
            val_df = pd.read_csv(val_manifest)
            pred_records = []
            for i in range(len(val_df)):
                pred_records.append({
                    "image_id": val_df.iloc[i]["image_id"],
                    "true_label": int(y_true[i]),
                    "predicted_grade": int(y_pred[i]),
                    "prob_g0": float(probs_arr[i][0]),
                    "prob_g1": float(probs_arr[i][1]),
                    "prob_g2": float(probs_arr[i][2]),
                    "prob_g3": float(probs_arr[i][3]),
                    "prob_g4": float(probs_arr[i][4])
                })
            pd.DataFrame(pred_records).to_csv(os.path.join(OUTPUT_DIR, "predictions.csv"), index=False)
            
            cm = confusion_matrix(y_true, y_pred, labels=list(range(5)))
            pd.DataFrame(cm, index=[f"True_G{i}" for i in range(5)], columns=[f"Pred_G{i}" for i in range(5)]).to_csv(os.path.join(OUTPUT_DIR, "confusion_matrix.csv"))
            
            cr = classification_report(y_true, y_pred, digits=4, output_dict=True, zero_division=0)
            pd.DataFrame(cr).transpose().to_csv(os.path.join(OUTPUT_DIR, "classification_report.csv"))
            
            metrics_summary = {
                "model_name": "M2_Global_Radiomics",
                "ablation_description": "Global radiomics (1 token, 288-d) cross-attending with 49 Swin-Tiny visual tokens -> CORAL",
                "dataset": "Real DDR (split_80_10_10)",
                "val_samples": len(val_ds),
                "best_epoch": best_epoch,
                "best_val_qwk": round(best_qwk, 4),
                "val_accuracy": round(acc, 4),
                "val_macro_f1": round(macro_f1, 4),
                "val_weighted_f1": round(weighted_f1, 4),
                "val_mae": round(mae, 4),
                "adjacent_error_rate": round(adj_err, 4),
                "catastrophic_error_rate": round(cat_err, 4),
                "total_parameters": total_params,
                "trainable_parameters": trainable_params
            }
            with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as fp:
                json.dump(metrics_summary, fp, indent=2)

    total_time = time.time() - t_start
    print(f"\nTraining completed in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Best Checkpoint: Epoch {best_epoch} with Val QWK: {best_qwk:.4f}")
    
    pd.DataFrame(history).to_csv(os.path.join(OUTPUT_DIR, "training_history.csv"), index=False)
    
    if len(history) >= 2 and history[-1]["val_qwk"] - history[-2]["val_qwk"] > 0.03:
        print("[NOTICE] Warning: Model was still improving substantially at Epoch 5 (+{:.4f} QWK). Possible undertraining flagged.".format(history[-1]["val_qwk"] - history[-2]["val_qwk"]))
        
    config_dict = {
        "model": {"name": "M2_Global_Radiomics", "backbone": "Swin-Tiny", "fusion": "Global Radiomics Cross-Attention (49x1, d=768)", "head": "CORAL"},
        "training": {
            "dataset": "Real DDR (split_80_10_10)",
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "seed": seed,
            "optimizer": "AdamW",
            "weight_decay": 0.01,
            "scheduler": "CosineAnnealingLR",
            "device": str(DEVICE),
            "imbalance_handling": "WeightedRandomSampler (inverse class frequencies)"
        },
        "results": {
            "best_epoch": best_epoch,
            "best_val_qwk": round(best_qwk, 4),
            "training_duration_sec": round(total_time, 1)
        }
    }
    with open(os.path.join(OUTPUT_DIR, "config.yaml"), "w") as fp:
        json.dump(config_dict, fp, indent=2)

if __name__ == "__main__":
    verification_test()
    train_m2(epochs=5, batch_size=32, lr=1e-4, seed=42)
