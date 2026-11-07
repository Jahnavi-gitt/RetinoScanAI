import os
import sys
import time
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.abspath("."))

from src.models.integrated_model import IntegratedRetinopathyModel
from src.models.coral import coral_predict, coral_probabilities
from src.data.data_pipeline import get_transforms, LABEL_MAPPING
from src.evaluation.metrics import compute_classification_metrics

OUTPUT_DIR = "outputs/Module_10_Integrated_Proposed_Model"
os.makedirs(OUTPUT_DIR, exist_ok=True)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

class RealDDRMultiModalDataset(Dataset):
    def __init__(self, npz_path: str, manifest_path: str, transform=None):
        self.transform = transform
        data = np.load(npz_path)
        self.spatial_radiomics = data["X_spatial_norm"].astype(np.float32)
        self.labels = data["y"].astype(np.int64)
        self.image_ids = data["image_ids"]
        
        manifest_df = pd.read_csv(manifest_path)
        id_to_fp = dict(zip(manifest_df["image_id"], manifest_df["filepath"]))
        self.filepaths = [id_to_fp.get(str(img_id)) for img_id in self.image_ids]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        fp = self.filepaths[idx]
        label = int(self.labels[idx])
        img_id = str(self.image_ids[idx])
        
        image = Image.open(fp).convert("RGB")
        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
            
        rad_tokens = torch.from_numpy(self.spatial_radiomics[idx])
        rad_tokens = torch.clamp(rad_tokens, -3.0, 3.0)
        
        return {
            "image": image_tensor,
            "radiomics": rad_tokens,
            "label": torch.tensor(label, dtype=torch.long),
            "image_id": img_id
        }

def train_real_ddr_model():
    print("==================================================================")
    print("  TRAINING INTEGRATED PROPOSED MODEL ON REAL DDR CLINICAL RETINAS")
    print(f"  Device: {DEVICE}")
    print("==================================================================")
    t0 = time.time()
    
    train_npz = "outputs/Module_06_Spatially_Localized_Radiomics/radiomic_features_train.npz"
    val_npz = "outputs/Module_06_Spatially_Localized_Radiomics/radiomic_features_val.npz"
    train_manifest = "outputs/Module_06_Spatially_Localized_Radiomics/active_real_train_manifest.csv"
    val_manifest = "outputs/Module_06_Spatially_Localized_Radiomics/active_real_val_manifest.csv"
    
    train_transform = get_transforms(is_training=True, img_size=224)
    val_transform = get_transforms(is_training=False, img_size=224)
    
    train_ds = RealDDRMultiModalDataset(train_npz, train_manifest, transform=train_transform)
    val_ds = RealDDRMultiModalDataset(val_npz, val_manifest, transform=val_transform)
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)
    
    print(f"DataLoaders ready: Train={len(train_ds)}, Val={len(val_ds)}")
    
    model = IntegratedRetinopathyModel(num_classes=5, pretrained_checkpoint=None).to(DEVICE)
    
    epochs = 6
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    best_qwk = -1.0
    best_state = None
    
    for epoch in range(1, epochs + 1):
        t_ep = time.time()
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for batch in train_loader:
            imgs = batch["image"].to(DEVICE)
            rads = batch["radiomics"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            
            optimizer.zero_grad()
            rank_logits, aux_logits = model(imgs, rads, return_aux=True)
            loss, _, _ = model.compute_hybrid_loss(rank_logits, aux_logits, labels, lambda_aux=0.3)
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
        val_targets, val_preds = [], []
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(DEVICE)
                rads = batch["radiomics"].to(DEVICE)
                labels = batch["label"].to(DEVICE)
                
                rank_logits, _ = model(imgs, rads, return_aux=True)
                preds = coral_predict(rank_logits)
                
                val_targets.extend(labels.cpu().numpy())
                val_preds.extend(preds.cpu().numpy())
                
        metrics, cm, _ = compute_classification_metrics(np.array(val_targets), np.array(val_preds))
        val_qwk = metrics["quadratic_weighted_kappa"]
        val_acc = metrics["accuracy"]
        val_mae = metrics["mean_absolute_error"]
        
        if val_qwk >= best_qwk:
            best_qwk = val_qwk
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
        ep_time = time.time() - t_ep
        print(f"Epoch {epoch:02d}/{epochs:02d} [{ep_time:.1f}s] | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
              f"Val QWK: {val_qwk:.4f} Acc: {val_acc:.3f} MAE: {val_mae:.3f}")
              
    model.load_state_dict(best_state)
    best_path = os.path.join(OUTPUT_DIR, "best_model.pth")
    torch.save(model.state_dict(), best_path)
    print(f"\nSaved best real model checkpoint -> {best_path} (Best QWK: {best_qwk:.4f})")
    
    swin_path = "outputs/Module_03_Baseline_Swin_Tiny/best_model.pth"
    torch.save(model.backbone.state_dict(), swin_path)
    print(f"Saved fine-tuned Swin-Tiny backbone -> {swin_path}")
    
    total_time = time.time() - t0
    print(f"[SUCCESS] Real DDR training completed in {total_time:.1f}s.")

if __name__ == "__main__":
    train_real_ddr_model()
