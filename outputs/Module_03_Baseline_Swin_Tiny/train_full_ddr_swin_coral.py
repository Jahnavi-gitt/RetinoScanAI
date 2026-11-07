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
import torchvision.models as tv_models
from torchvision import transforms
from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score, recall_score, precision_score

sys.path.append(os.path.abspath("."))
from src.data.data_pipeline import retina_circle_crop, apply_clahe_lab, IMAGENET_MEAN, IMAGENET_STD
from src.models.coral import CoralLayer, coral_loss, coral_predict, coral_probabilities

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
OUTPUT_DIR = "outputs/Module_03_Baseline_Swin_Tiny/clean_run"
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

class FullDDRDataset(Dataset):
    def __init__(self, manifest_csv: str, transform=None):
        self.df = pd.read_csv(manifest_csv).reset_index(drop=True)
        self.transform = transform
        self.filepaths = self.df["filepath"].tolist()
        self.labels = self.df["numeric_label"].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        fp = self.filepaths[idx]
        label = self.labels[idx]
        with Image.open(fp) as im:
            img = im.convert("RGB")
            if self.transform:
                tensor = self.transform(img)
            else:
                tensor = transforms.ToTensor()(img)
        return tensor, torch.tensor(label, dtype=torch.long)

class SwinTinyCoralModel(nn.Module):
    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.num_classes = num_classes
        
        cache_dir = os.path.expanduser("~/.cache/torch/hub/checkpoints")
        ckpt_path = os.path.join(cache_dir, "swin_t-704ceda3.pth")
        base = tv_models.swin_t(weights=None)
        if os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location="cpu")
            base.load_state_dict(state)
            print(f"[Swin-Tiny] Loaded ImageNet-1k weights from {ckpt_path}")
        else:
            print("[Swin-Tiny] Warning: Cached weights not found, using randomly initialized backbone.")
            
        base.head = nn.Identity()
        self.backbone = base
        self.coral_head = CoralLayer(in_features=768, num_classes=num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        logits = self.coral_head(feats)
        return logits

def train_full_model(epochs: int = 8, batch_size: int = 32, lr: float = 1e-4, patience: int = 4):
    print("==================================================================")
    print("  TRAINING SWIN-TINY + CORAL ON FULL CLEAN DDR DATASET (10,024 IMGS)")
    print(f"  Device: {DEVICE} | Batch Size: {batch_size} | Epochs: {epochs}")
    print(f"  Output Checkpoint: {OUTPUT_DIR}/best_model_clean.pth")
    print("==================================================================")
    
    train_manifest = "data_manifests/split_80_10_10/train_manifest.csv"
    val_manifest = "data_manifests/split_80_10_10/val_manifest.csv"
    
    train_transform = StandardRetinaTransform(is_training=True, img_size=224)
    val_transform = StandardRetinaTransform(is_training=False, img_size=224)
    
    train_ds = FullDDRDataset(train_manifest, transform=train_transform)
    val_ds = FullDDRDataset(val_manifest, transform=val_transform)
    
    train_labels = np.array(train_ds.labels)
    class_counts = np.bincount(train_labels, minlength=5)
    print(f"Train Class Counts: {class_counts.tolist()}")
    
    class_weights = len(train_labels) / (5.0 * np.maximum(class_counts, 1))
    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(train_ds),
        replacement=True
    )
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"DataLoaders ready: Train={len(train_ds)}, Val={len(val_ds)}")
    
    model = SwinTinyCoralModel(num_classes=5).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    best_qwk = -1.0
    epochs_no_improve = 0
    history = []
    
    t_start = time.time()
    
    for epoch in range(1, epochs + 1):
        t_ep = time.time()
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for imgs, targets in train_loader:
            imgs = imgs.to(DEVICE)
            targets = targets.to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(imgs)
            loss = coral_loss(logits, targets, num_classes=5)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
        scheduler.step()
        avg_train_loss = train_loss / max(1, train_batches)
        
        model.eval()
        val_loss = 0.0
        val_batches = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(DEVICE)
                targets = targets.to(DEVICE)
                logits = model(imgs)
                loss = coral_loss(logits, targets, num_classes=5)
                val_loss += loss.item()
                val_batches += 1
                
                preds = coral_predict(logits, threshold=0.5)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_targets.extend(targets.cpu().numpy().tolist())
                
        avg_val_loss = val_loss / max(1, val_batches)
        y_true = np.array(all_targets)
        y_pred = np.array(all_preds)
        
        val_qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
        val_acc = accuracy_score(y_true, y_pred)
        val_macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        per_grade_rec = recall_score(y_true, y_pred, average=None, labels=list(range(5)), zero_division=0)
        
        ep_duration = time.time() - t_ep
        rec_str = " | ".join([f"G{i}:{per_grade_rec[i]:.2f}" for i in range(5)])
        
        print(f"Epoch {epoch:02d}/{epochs:02d} [{ep_duration:.1f}s] - "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val QWK: {val_qwk:.4f} | "
              f"Val F1: {val_macro_f1:.4f} | "
              f"Val Acc: {val_acc*100:.1f}% | "
              f"Recalls: [{rec_str}]")
        
        ep_stats = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "val_qwk": round(val_qwk, 4),
            "val_macro_f1": round(val_macro_f1, 4),
            "val_accuracy": round(val_acc, 4),
            "rec_g0": round(float(per_grade_rec[0]), 4),
            "rec_g1": round(float(per_grade_rec[1]), 4),
            "rec_g2": round(float(per_grade_rec[2]), 4),
            "rec_g3": round(float(per_grade_rec[3]), 4),
            "rec_g4": round(float(per_grade_rec[4]), 4),
            "duration_sec": round(ep_duration, 1)
        }
        history.append(ep_stats)
        
        if val_qwk > best_qwk:
            best_qwk = val_qwk
            epochs_no_improve = 0
            ckpt_path = os.path.join(OUTPUT_DIR, "best_model_clean.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_qwk": val_qwk,
                "val_macro_f1": val_macro_f1,
                "val_accuracy": val_acc,
                "per_grade_recall": per_grade_rec.tolist()
            }, ckpt_path)
            print(f"  >>> Best model saved (QWK = {val_qwk:.4f}) to {ckpt_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  >>> Early stopping triggered after {patience} epochs without QWK improvement.")
                break
                
    total_time = time.time() - t_start
    print(f"\nTraining completed in {total_time/60:.2f} mins. Best Validation QWK = {best_qwk:.4f}")
    
    hist_df = pd.DataFrame(history)
    hist_csv = os.path.join(OUTPUT_DIR, "training_history.csv")
    hist_df.to_csv(hist_csv, index=False)
    print(f"Saved training history to {hist_csv}")

if __name__ == "__main__":
    train_full_model(epochs=8, batch_size=32, lr=1e-4, patience=4)
