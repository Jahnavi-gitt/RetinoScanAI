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
from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score, mean_absolute_error, classification_report

sys.path.append(os.path.abspath("."))
from src.data.data_pipeline import retina_circle_crop, apply_clahe_lab, IMAGENET_MEAN, IMAGENET_STD
from outputs.Module_04_Baseline_EfficientNet_B3.efficientnet_b3 import EfficientNetB3Baseline

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
OUTPUT_DIR = "outputs/Module_04_Baseline_EfficientNet_B3/full_ddr"
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
        self.image_ids = self.df["image_id"].tolist()

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        fp = self.filepaths[idx]
        label = self.labels[idx]
        image_id = self.image_ids[idx]
        
        try:
            img = Image.open(fp).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (0, 0, 0))
            
        if self.transform is not None:
            tensor = self.transform(img)
        else:
            tensor = transforms.ToTensor()(img)
            
        return tensor, label, image_id

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_ids = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels, ids in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(labels.cpu().numpy().tolist())
            all_ids.extend(ids)
            all_probs.extend(probs.cpu().numpy().tolist())
            
    val_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    qwk = cohen_kappa_score(all_targets, all_preds, weights="quadratic")
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)
    mae = mean_absolute_error(all_targets, all_preds)
    
    return {
        "loss": val_loss,
        "accuracy": acc,
        "qwk": qwk,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "mae": mae,
        "preds": all_preds,
        "targets": all_targets,
        "ids": all_ids,
        "probs": all_probs
    }

def train_efficientnet(epochs: int = 5, batch_size: int = 32, lr: float = 1e-4):
    log_file = os.path.join(OUTPUT_DIR, "run_log.txt")
    
    def log_print(msg):
        print(msg)
        with open(log_file, "a") as f:
            f.write(msg + "\n")
            
    with open(log_file, "w") as f:
        f.write("=== EFFICIENTNET-B3 FULL DDR TRAINING LOG ===\n")

    log_print(f"Device: {DEVICE} (Apple Silicon MPS / CPU)")
    log_print(f"Batch Size: {batch_size} | Epochs: {epochs} | LR: {lr}")
    
    train_manifest = "data_manifests/split_80_10_10/train_manifest.csv"
    val_manifest = "data_manifests/split_80_10_10/val_manifest.csv"
    
    train_transform = StandardRetinaTransform(is_training=True, img_size=224)
    val_transform = StandardRetinaTransform(is_training=False, img_size=224)
    
    train_ds = FullDDRDataset(train_manifest, transform=train_transform)
    val_ds = FullDDRDataset(val_manifest, transform=val_transform)
    
    log_print(f"Manifest Verified: Train={len(train_ds)} samples ({len(set(train_ds.image_ids))} unique IDs)")
    log_print(f"Manifest Verified: Val={len(val_ds)} samples ({len(set(val_ds.image_ids))} unique IDs)")
    
    train_labels = np.array(train_ds.labels)
    class_counts = np.bincount(train_labels, minlength=5)
    log_print(f"Train Class Counts: {class_counts.tolist()}")
    
    class_weights = len(train_labels) / (5.0 * np.maximum(class_counts, 1))
    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(train_ds),
        replacement=True
    )
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    log_print("Initializing EfficientNet-B3 Baseline with ImageNet-1K pretrained weights...")
    model = EfficientNetB3Baseline(num_classes=5, pretrained=True).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    log_print(f"EfficientNet-B3: Total Parameters = {total_params:,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    best_qwk = -1.0
    best_epoch = 0
    history = []
    start_epoch = 1
    
    ckpt_path = os.path.join(OUTPUT_DIR, "best_model.pth")
    if os.path.exists(ckpt_path):
        try:
            ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if "model_state_dict" in ckpt_data and "epoch" in ckpt_data:
                model.load_state_dict(ckpt_data["model_state_dict"])
                best_qwk = ckpt_data.get("val_qwk", -1.0)
                best_epoch = ckpt_data.get("epoch", 0)
                start_epoch = best_epoch + 1
                log_print(f"Resuming training from Epoch {start_epoch} (Checkpoint from Epoch {best_epoch}, Val QWK: {best_qwk:.4f})")
                hist_csv = os.path.join(OUTPUT_DIR, "training_history.csv")
                if os.path.exists(hist_csv):
                    history = pd.read_csv(hist_csv).to_dict(orient="records")
        except Exception as e:
            log_print(f"Note: starting fresh ({e})")
            
    t_start = time.time()
    
    for epoch in range(start_epoch, epochs + 1):
        t_ep0 = time.time()
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0
        
        for step, (images, labels, _) in enumerate(train_loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=-1)
            train_correct += (preds == labels).sum().item()
            total_train += labels.size(0)
            
            if (step + 1) % 50 == 0 or (step + 1) == len(train_loader):
                print(f"Epoch [{epoch}/{epochs}] Step [{step+1}/{len(train_loader)}] Loss: {loss.item():.4f}")
                
        scheduler.step()
        
        ep_train_loss = train_loss / total_train
        ep_train_acc = train_correct / total_train
        
        val_res = evaluate(model, val_loader, criterion)
        ep_time = time.time() - t_ep0
        
        log_print(
            f"Epoch {epoch:02d}/{epochs:02d} [{ep_time:.1f}s] | "
            f"Train Loss: {ep_train_loss:.4f} Acc: {ep_train_acc:.3f} | "
            f"Val Loss: {val_res['loss']:.4f} Acc: {val_res['accuracy']:.3f} "
            f"QWK: {val_res['qwk']:.4f} F1: {val_res['macro_f1']:.4f} MAE: {val_res['mae']:.3f}"
        )
        
        history.append({
            "epoch": epoch,
            "train_loss": round(ep_train_loss, 4),
            "train_acc": round(ep_train_acc, 4),
            "val_loss": round(val_res["loss"], 4),
            "val_acc": round(val_res["accuracy"], 4),
            "val_qwk": round(val_res["qwk"], 4),
            "val_macro_f1": round(val_res["macro_f1"], 4),
            "val_weighted_f1": round(val_res["weighted_f1"], 4),
            "val_mae": round(val_res["mae"], 4),
            "duration_sec": round(ep_time, 1)
        })
        pd.DataFrame(history).to_csv(os.path.join(OUTPUT_DIR, "training_history.csv"), index=False)
        
        if val_res["qwk"] > best_qwk:
            best_qwk = val_res["qwk"]
            best_epoch = epoch
            ckpt_path = os.path.join(OUTPUT_DIR, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_qwk": best_qwk,
                "val_accuracy": val_res["accuracy"],
                "val_macro_f1": val_res["macro_f1"]
            }, ckpt_path)
            log_print(f"  --> Saved new best checkpoint to {ckpt_path} (Val QWK: {best_qwk:.4f})")
            
            pred_df = pd.DataFrame({
                "image_id": val_res["ids"],
                "true_label": val_res["targets"],
                "predicted_grade": val_res["preds"],
                "prob_g0": [p[0] for p in val_res["probs"]],
                "prob_g1": [p[1] for p in val_res["probs"]],
                "prob_g2": [p[2] for p in val_res["probs"]],
                "prob_g3": [p[3] for p in val_res["probs"]],
                "prob_g4": [p[4] for p in val_res["probs"]],
            })
            pred_df.to_csv(os.path.join(OUTPUT_DIR, "predictions.csv"), index=False)
            
            clf_rep = classification_report(val_res["targets"], val_res["preds"], output_dict=True, zero_division=0)
            pd.DataFrame(clf_rep).transpose().to_csv(os.path.join(OUTPUT_DIR, "classification_report.csv"))
            
            metrics_summary = {
                "model_name": "EfficientNet-B3 Baseline (Full DDR)",
                "dataset": "OIA-DDR Clean Split (80/10/10)",
                "train_samples": len(train_ds),
                "val_samples": len(val_ds),
                "best_epoch": best_epoch,
                "best_val_qwk": round(best_qwk, 4),
                "val_accuracy": round(val_res["accuracy"], 4),
                "val_macro_f1": round(val_res["macro_f1"], 4),
                "val_weighted_f1": round(val_res["weighted_f1"], 4),
                "val_mae": round(val_res["mae"], 4),
                "val_loss": round(val_res["loss"], 4)
            }
            with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as fp:
                json.dump(metrics_summary, fp, indent=2)

    total_time = time.time() - t_start
    log_print(f"\nTraining completed in {total_time:.1f}s ({total_time/60:.1f} min)")
    log_print(f"Best Checkpoint: Epoch {best_epoch} with Val QWK: {best_qwk:.4f}")
    
    pd.DataFrame(history).to_csv(os.path.join(OUTPUT_DIR, "training_history.csv"), index=False)
    
    config_dict = {
        "model": {"name": "EfficientNet-B3", "num_classes": 5, "pretrained": True},
        "training": {
            "dataset": "Real DDR (split_80_10_10)",
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
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
    train_efficientnet(epochs=5, batch_size=32, lr=1e-4)
