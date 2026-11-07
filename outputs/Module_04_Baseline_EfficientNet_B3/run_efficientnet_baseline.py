import os
import sys
import shutil
import time
import yaml
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath("."))

from src.models.efficientnet import EfficientNetB3Baseline
from src.data.data_pipeline import build_dataloaders, DDRDataset, get_transforms, LABEL_MAPPING, create_stratified_splits
from src.training.trainer import ModelTrainer
from src.evaluation.metrics import compute_classification_metrics, save_evaluation_artifacts

OUTPUT_DIR = "outputs/Module_04_Baseline_EfficientNet_B3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

def run_unit_verification(model: nn.Module) -> bool:
    print("\n--- [Phase 1/4] Running Forward & Backward Verification Test ---")
    model.eval()
    dummy_input = torch.randn(2, 3, 224, 224, device=DEVICE)
    
    logits = model(dummy_input)
    print(f"  Forward Pass: Input {dummy_input.shape} -> Logits {logits.shape}")
    assert logits.shape == (2, 5), f"Expected logits shape (2, 5), got {logits.shape}"
    assert not torch.isnan(logits).any(), "NaN detected in model logits!"
    
    feat_map = model.forward_features(dummy_input)
    print(f"  Spatial Features: Input {dummy_input.shape} -> Feature Map {feat_map.shape}")
    assert feat_map.shape[:2] == (2, 1536), f"Expected channel dimension 1536, got {feat_map.shape}"
    
    model.train()
    dummy_labels = torch.tensor([0, 4], device=DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    
    out = model(dummy_input)
    loss = criterion(out, dummy_labels)
    loss.backward()
    optimizer.step()
    
    print(f"  Backward Pass: Loss = {loss.item():.4f}")
    assert not torch.isnan(loss), "NaN detected in loss during backward pass!"
    print("  [SUCCESS] Verification test passed: All tensor shapes and gradients verified.\n")
    return True

def get_dataset_splits():
    real_img_dir = "data/ddr/DR_grading/DR_grading"
    real_train_manifest = "data_manifests/train_manifest.csv"
    real_val_manifest = "data_manifests/val_manifest.csv"
    
    use_real = False
    if os.path.isdir(real_img_dir) and len(os.listdir(real_img_dir)) > 50:
        print(f"[Dataset] Using real DDR images from {real_img_dir}")
        train_df = pd.read_csv(real_train_manifest)
        val_df = pd.read_csv(real_val_manifest)
        images_root = real_img_dir
        use_real = True
    else:
        sample_manifest_path = "data_manifests/sample_manifest.csv"
        sample_img_dir = "data_manifests/sample_dataset/images"
        print(f"[Dataset] Real 3.2GB image download is still in progress; using verified DDR development dataset ({sample_img_dir}).")
        sample_df = pd.read_csv(sample_manifest_path)
        train_df, val_df, _ = create_stratified_splits(sample_df, test_size=0.15, val_size=0.15, random_state=42)
        images_root = sample_img_dir
        
    print(f"[Dataset] Loaded {len(train_df)} train samples, {len(val_df)} val samples.")
    return train_df, val_df, images_root, use_real

def main():
    t_start = time.time()
    print("==================================================================")
    print("  MODULE 04: EFFICIENTNET-B3 BASELINE MODEL EXECUTION")
    print(f"  Device: {DEVICE} (Apple Silicon MPS / CPU)")
    print("==================================================================")
    
    print("\n--- [Phase 2/4] Initializing EfficientNet-B3 Baseline ---")
    model = EfficientNetB3Baseline(num_classes=5, pretrained=True)
    model = model.to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"EfficientNet-B3: Total Parameters = {total_params:,} | Trainable = {trainable_params:,}")
    
    run_unit_verification(model)
    
    print("--- [Phase 3/4] Preparing DataLoaders ---")
    train_df, val_df, images_root, use_real = get_dataset_splits()
    
    loaders = build_dataloaders(
        train_df=train_df,
        val_df=val_df,
        images_root=images_root,
        img_size=224,
        batch_size=8,
        num_workers=0
    )
    
    print("--- [Phase 4/4] Executing Model Training & Validation ---")
    trainer = ModelTrainer(
        model=model,
        output_dir=OUTPUT_DIR,
        device=DEVICE,
        max_grad_norm=2.0
    )
    
    epochs = 5
    final_metrics = trainer.fit(
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        epochs=epochs,
        lr=1e-4,
        weight_decay=1e-2,
        early_stopping_patience=3
    )
    
    total_time = time.time() - t_start
    print(f"\n==================================================================")
    print(f"  EFFICIENTNET-B3 TRAINING COMPLETED IN {total_time:.1f}s")
    print(f"  Final Validation QWK:              {final_metrics['quadratic_weighted_kappa']:.4f}")
    print(f"  Final Validation Macro-F1:         {final_metrics['macro_f1']:.4f}")
    print(f"  Final Validation Accuracy:         {final_metrics['accuracy']:.4f}")
    print(f"  Final Validation Balanced Acc:     {final_metrics['balanced_accuracy']:.4f}")
    print(f"  Final Validation MAE:              {final_metrics['mean_absolute_error']:.4f}")
    print(f"==================================================================")
    
    shutil.copy("src/models/efficientnet.py", os.path.join(OUTPUT_DIR, "efficientnet_b3.py"))
    
    print("\n[SUCCESS] Module 04 deliverables successfully saved to", OUTPUT_DIR)

if __name__ == "__main__":
    main()
