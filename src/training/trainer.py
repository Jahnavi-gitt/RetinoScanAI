import os
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.evaluation.metrics import compute_classification_metrics, save_evaluation_artifacts

def get_optimizer_and_scheduler(model, lr: float = 1e-4, weight_decay: float = 1e-2, epochs: int = 10):
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    return optimizer, scheduler

def save_predictions_csv(y_true, y_pred, y_probs, image_ids, save_path: str):
    records = []
    for i in range(len(y_true)):
        rec = {
            "image_id": str(image_ids[i]),
            "true_grade": int(y_true[i]),
            "predicted_grade": int(y_pred[i]),
            "error": int(abs(int(y_true[i]) - int(y_pred[i]))),
            "is_correct": bool(y_true[i] == y_pred[i])
        }
        if y_probs is not None:
            for c in range(y_probs.shape[1]):
                rec[f"prob_grade_{c}"] = round(float(y_probs[i, c]), 4)
        records.append(rec)
        
    df = pd.DataFrame(records)
    df.to_csv(save_path, index=False)
    print(f"Saved per-sample predictions ({len(df)} rows) -> {save_path}")
    return df

class ModelTrainer:
    def __init__(
        self,
        model: nn.Module,
        output_dir: str,
        criterion: nn.Module = None,
        device: torch.device = None,
        max_grad_norm: float = 2.0
    ):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device
            
        self.model = model.to(self.device)
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self.max_grad_norm = max_grad_norm
        
        self.best_qwk = -1.0
        self.best_checkpoint_path = os.path.join(self.output_dir, "best_model.pth")
        self.history = []

    def train_epoch(self, dataloader, optimizer) -> tuple:
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch in tqdm(dataloader, desc="Training", leave=False):
            images, labels, _ = batch
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
            
        epoch_loss = running_loss / max(1, total)
        epoch_acc = correct / max(1, total)
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def evaluate(self, dataloader) -> tuple:
        self.model.eval()
        running_loss = 0.0
        total = 0
        
        all_true = []
        all_preds = []
        all_probs = []
        all_ids = []
        
        for batch in dataloader:
            images, labels, ids = batch
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            total += images.size(0)
            
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()
            
            all_true.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.tolist())
            all_probs.append(probs)
            all_ids.extend(ids)
            
        val_loss = running_loss / max(1, total)
        all_probs = np.vstack(all_probs)
        
        metrics, cm, clf_dict = compute_classification_metrics(all_true, all_preds)
        metrics["loss"] = float(val_loss)
        
        return metrics, cm, clf_dict, all_true, all_preds, all_probs, all_ids

    def fit(
        self,
        train_loader,
        val_loader,
        epochs: int = 10,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        early_stopping_patience: int = 5
    ) -> dict:
        optimizer, scheduler = get_optimizer_and_scheduler(
            self.model, lr=lr, weight_decay=weight_decay, epochs=epochs
        )
        
        start_time = time.time()
        no_improve_epochs = 0
        
        print(f"\n[Trainer] Starting training on device: {self.device} for {epochs} epochs...")
        
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss, train_acc = self.train_epoch(train_loader, optimizer)
            scheduler.step()
            
            val_metrics, val_cm, val_clf, y_true, y_pred, y_probs, y_ids = self.evaluate(val_loader)
            val_qwk = val_metrics["quadratic_weighted_kappa"]
            val_loss = val_metrics["loss"]
            val_acc = val_metrics["accuracy"]
            val_f1 = val_metrics["macro_f1"]
            val_mae = val_metrics["mean_absolute_error"]
            
            elapsed = time.time() - t0
            
            print(f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} QWK: {val_qwk:.4f} F1: {val_f1:.4f} MAE: {val_mae:.3f}")
                  
            is_best = val_qwk > self.best_qwk
            if is_best:
                self.best_qwk = val_qwk
                no_improve_epochs = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_qwk": val_qwk,
                    "val_loss": val_loss,
                    "val_metrics": val_metrics
                }, self.best_checkpoint_path)
                print(f"  --> Saved new best checkpoint to {self.best_checkpoint_path} (QWK: {val_qwk:.4f})")
            else:
                no_improve_epochs += 1
                
            self.history.append({
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_acc": round(train_acc, 4),
                "val_loss": round(val_loss, 4),
                "val_acc": round(val_acc, 4),
                "val_qwk": round(val_qwk, 4),
                "val_macro_f1": round(val_f1, 4),
                "val_mae": round(val_mae, 4),
                "lr": round(scheduler.get_last_lr()[0], 6),
                "epoch_time_sec": round(elapsed, 2)
            })
            
            if no_improve_epochs >= early_stopping_patience:
                print(f"[Trainer] Early stopping triggered after {epoch} epochs.")
                break
                
        total_time = time.time() - start_time
        
        history_path = os.path.join(self.output_dir, "training_history.csv")
        pd.DataFrame(self.history).to_csv(history_path, index=False)
        print(f"[Trainer] Saved training history -> {history_path}")
        
        if os.path.exists(self.best_checkpoint_path):
            ckpt = torch.load(self.best_checkpoint_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            print(f"[Trainer] Reloaded best checkpoint from epoch {ckpt['epoch']} (QWK: {ckpt['val_qwk']:.4f})")
            
        final_metrics, final_cm, final_clf, y_true, y_pred, y_probs, y_ids = self.evaluate(val_loader)
        final_metrics["total_training_time_seconds"] = round(total_time, 2)
        final_metrics["best_epoch"] = ckpt["epoch"] if os.path.exists(self.best_checkpoint_path) else epochs
        
        save_evaluation_artifacts(final_metrics, final_cm, final_clf, self.output_dir)
        
        preds_path = os.path.join(self.output_dir, "predictions.csv")
        save_predictions_csv(y_true, y_pred, y_probs, y_ids, preds_path)
        
        return final_metrics
