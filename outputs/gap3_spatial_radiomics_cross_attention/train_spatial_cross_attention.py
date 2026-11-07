import os
import sys
import time
import json
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score, recall_score, precision_score

sys.path.append(os.path.abspath("."))
from outputs.gap3_spatial_radiomics_cross_attention.model import SpatialRadiomicsSwinCoralModel
from src.data.data_pipeline import retina_circle_crop, apply_clahe_lab, IMAGENET_MEAN, IMAGENET_STD
from src.models.coral import coral_loss, coral_predict, coral_probabilities

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
OUTPUT_DIR = "outputs/gap3_spatial_radiomics_cross_attention"
CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")
os.makedirs(OUTPUT_DIR, exist_ok=True)

class StandardRetinaTransform:
    def __init__(self, is_training: bool = False, img_size: int = 224):
        self.is_training = is_training
        self.img_size = img_size
        
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
        
        cache_data = np.load(cache_npz)
        self.radiomics = cache_data["X_spatial_norm"].astype(np.float32)
        assert len(self.df) == len(self.radiomics), f"Mismatch between manifest ({len(self.df)}) and cache ({len(self.radiomics)})"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        fp = self.filepaths[idx]
        label = self.labels[idx]
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
            "label": torch.tensor(label, dtype=torch.long)
        }

def compute_epoch_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_grade_recall = recall_score(y_true, y_pred, average=None, labels=[0, 1, 2, 3, 4], zero_division=0)
    return {
        "qwk": round(float(qwk), 4),
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "rec_g0": round(float(per_grade_recall[0]), 4),
        "rec_g1": round(float(per_grade_recall[1]), 4),
        "rec_g2": round(float(per_grade_recall[2]), 4),
        "rec_g3": round(float(per_grade_recall[3]), 4),
        "rec_g4": round(float(per_grade_recall[4]), 4)
    }

def train_model(epochs: int = 8, batch_size: int = 32, lr: float = 1e-4):
    print("=" * 70)
    print("  TRAINING GAP 3: SPATIAL RADIOMICS CROSS-ATTENTION SWIN-CORAL MODEL")
    print(f"  Device: {DEVICE} | Batch Size: {batch_size} | Epochs: {epochs}")
    print(f"  Checkpoints Directory: {OUTPUT_DIR}")
    print("=" * 70)
    
    train_manifest = "data_manifests/split_80_10_10/train_manifest.csv"
    val_manifest = "data_manifests/split_80_10_10/val_manifest.csv"
    
    train_cache = os.path.join(CACHE_DIR, "radiomics_train.npz")
    val_cache = os.path.join(CACHE_DIR, "radiomics_val.npz")
    
    assert os.path.exists(train_cache), f"Missing cache {train_cache}. Run cache_split_radiomics.py first."
    assert os.path.exists(val_cache), f"Missing cache {val_cache}. Run cache_split_radiomics.py first."
    
    train_transform = StandardRetinaTransform(is_training=True, img_size=224)
    val_transform = StandardRetinaTransform(is_training=False, img_size=224)
    
    train_ds = MultiModalDDRDataset(train_manifest, train_cache, transform=train_transform)
    val_ds = MultiModalDDRDataset(val_manifest, val_cache, transform=val_transform)
    
    train_labels = train_ds.labels
    class_counts = pd.Series(train_labels).value_counts().sort_index().to_dict()
    print(f"Train Class Distribution: {class_counts}")
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[lbl] for lbl in train_labels]
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda")
    )
    
    print(f"Dataloaders ready: Train={len(train_ds)} (Batches={len(train_loader)}), Val={len(val_ds)} (Batches={len(val_loader)})")
    
    clean_ckpt = "outputs/Module_03_Baseline_Swin_Tiny/clean_run/best_model_clean.pth"
    model = SpatialRadiomicsSwinCoralModel(
        num_classes=5,
        d_vis=768,
        d_rad=32,
        d_model=768,
        num_heads=8,
        dropout=0.2,
        pretrained_backbone=True,
        backbone_checkpoint=clean_ckpt if os.path.exists(clean_ckpt) else None
    ).to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    best_val_qwk = -1.0
    best_epoch = -1
    best_checkpoint_path = os.path.join(OUTPUT_DIR, "best_spatial_radiomics_coral_model.pth")
    history = []
    
    for epoch in range(1, epochs + 1):
        t_ep = time.time()
        model.train()
        total_loss, total_samples = 0.0, 0
        
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
            
            total_loss += loss.item() * imgs.size(0)
            total_samples += imgs.size(0)
            
            if (b_idx + 1) % 50 == 0 or (b_idx + 1) == len(train_loader):
                print(f"  Epoch {epoch:02d} [{b_idx + 1:03d}/{len(train_loader):03d}] Loss: {loss.item():.4f}")
                
        scheduler.step()
        train_loss = total_loss / total_samples
        
        model.eval()
        val_loss_total, val_samples = 0.0, 0
        all_val_targets, all_val_preds = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(DEVICE)
                rads = batch["radiomics"].to(DEVICE)
                labels = batch["label"].to(DEVICE)
                
                rank_logits = model(imgs, rads)
                loss = coral_loss(rank_logits, labels, num_classes=5)
                preds = coral_predict(rank_logits)
                
                val_loss_total += loss.item() * imgs.size(0)
                val_samples += imgs.size(0)
                all_val_targets.extend(labels.cpu().numpy().tolist())
                all_val_preds.extend(preds.cpu().numpy().tolist())
                
        val_loss = val_loss_total / val_samples
        val_metrics = compute_epoch_metrics(np.array(all_val_targets), np.array(all_val_preds))
        duration = round(time.time() - t_ep, 1)
        
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_qwk": val_metrics["qwk"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_accuracy": val_metrics["accuracy"],
            "rec_g0": val_metrics["rec_g0"],
            "rec_g1": val_metrics["rec_g1"],
            "rec_g2": val_metrics["rec_g2"],
            "rec_g3": val_metrics["rec_g3"],
            "rec_g4": val_metrics["rec_g4"],
            "duration_sec": duration
        }
        history.append(row)
        
        print(f"\n>>> Epoch {epoch:02d} Summary [{duration}s]: Train Loss={train_loss:.4f} | "
              f"Val Loss={val_loss:.4f} | Val QWK={val_metrics['qwk']:.4f} | "
              f"Val Acc={val_metrics['accuracy']:.4f} | Val Macro-F1={val_metrics['macro_f1']:.4f}")
        print(f"    Validation Recalls: G0={val_metrics['rec_g0']:.3f}, G1={val_metrics['rec_g1']:.3f}, "
              f"G2={val_metrics['rec_g2']:.3f}, G3={val_metrics['rec_g3']:.3f}, G4={val_metrics['rec_g4']:.3f}")
              
        if val_metrics["qwk"] > best_val_qwk:
            best_val_qwk = val_metrics["qwk"]
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_qwk": best_val_qwk,
                "val_metrics": val_metrics,
                "config": {
                    "num_classes": 5,
                    "d_vis": 768,
                    "d_rad": 32,
                    "d_model": 768,
                    "num_heads": 8
                }
            }, best_checkpoint_path)
            print(f"    [*] NEW BEST MODEL SAVED to {best_checkpoint_path} (Val QWK: {best_val_qwk:.4f})\n")
        else:
            print(f"    [-] Val QWK ({val_metrics['qwk']:.4f}) did not beat best ({best_val_qwk:.4f})\n")
            
    history_df = pd.DataFrame(history)
    history_csv = os.path.join(OUTPUT_DIR, "training_history.csv")
    history_df.to_csv(history_csv, index=False)
    print(f"\nTraining completed! Best Epoch: {best_epoch:02d} with Val QWK: {best_val_qwk:.4f}")
    print(f"History saved to {history_csv}")

if __name__ == "__main__":
    train_model(epochs=6, batch_size=32, lr=1e-4)
