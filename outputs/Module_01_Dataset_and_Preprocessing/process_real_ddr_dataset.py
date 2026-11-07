import os
import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import torch
from PIL import Image

import sys
sys.path.append(os.path.abspath("."))

from src.data.data_pipeline import (
    LABEL_MAPPING,
    CLASS_NAMES,
    apply_clahe_lab,
    retina_circle_crop,
    get_transforms,
    DDRDataset,
    create_stratified_splits,
    audit_dataset,
    build_dataloaders
)

np.random.seed(42)
torch.manual_seed(42)

OUTPUT_DIR = "outputs/Module_01_Dataset_and_Preprocessing"
MANIFEST_DIR = "data_manifests"

def find_ddr_files():
    csv_candidates = glob.glob("data/ddr/**/DR_grading.csv", recursive=True) + glob.glob("data/**/DR_grading.csv", recursive=True)
    if not csv_candidates:
        csv_candidates = glob.glob("data/**/*.csv", recursive=True)
    
    if not csv_candidates:
        raise FileNotFoundError("Could not find DDR annotation CSV in data/")
    
    csv_path = csv_candidates[0]
    print(f"Found DDR annotation CSV: {csv_path}")
    
    img_candidates = glob.glob("data/ddr/**/DR_grading", recursive=True) + glob.glob("data/**/DR_grading", recursive=True)
    img_dir = None
    for cand in img_candidates:
        if os.path.isdir(cand) and len(glob.glob(os.path.join(cand, "*.jpg"))) > 0:
            img_dir = cand
            break
            
    if img_dir is None:
        for root, dirs, files in os.walk("data/ddr"):
            jpgs = [f for f in files if f.endswith(".jpg")]
            if len(jpgs) > 10:
                img_dir = root
                break
                
    print(f"Found DDR image directory: {img_dir}")
    return csv_path, img_dir

def main():
    print("[1/6] Locating downloaded DDR dataset files...")
    csv_path, img_dir = find_ddr_files()
    
    df_raw = pd.read_csv(csv_path)
    print(f"Raw CSV loaded: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
    print("Columns:", list(df_raw.columns))
    print("Sample:\n", df_raw.head(3))
    
    col_id = None
    col_label = None
    for c in df_raw.columns:
        if c.lower() in ["image_id", "image", "id", "name", "filename"]:
            col_id = c
        if c.lower() in ["numeric_label", "dr_level", "label", "grade", "level", "dr_grade"]:
            col_label = c
            
    if col_id is None:
        col_id = df_raw.columns[0]
    if col_label is None:
        col_label = df_raw.columns[1]
        
    print(f"Using image column: '{col_id}', label column: '{col_label}'")
    
    df_manifest = pd.DataFrame()
    df_manifest["image_id"] = df_raw[col_id].astype(str).apply(lambda x: x if x.endswith(".jpg") else x + ".jpg")
    df_manifest["numeric_label"] = df_raw[col_label].astype(int)
    df_manifest["filepath"] = df_manifest["image_id"].apply(lambda fname: os.path.join(img_dir, fname))
    
    df_manifest["file_exists"] = df_manifest["filepath"].apply(os.path.exists)
    existing_count = df_manifest["file_exists"].sum()
    print(f"Verified files on disk: {existing_count} / {len(df_manifest)}")
    
    df_clean = df_manifest[df_manifest["file_exists"]].copy().reset_index(drop=True)
    df_clean["grade_name"] = df_clean["numeric_label"].map(LABEL_MAPPING)
    
    master_manifest_path = os.path.join(MANIFEST_DIR, "ddr_master_manifest.csv")
    df_clean.to_csv(master_manifest_path, index=False)
    print(f"Saved master DDR manifest -> {master_manifest_path} ({len(df_clean)} samples)")
    
    print("\n[2/6] Running dataset audit on real DDR images...")
    audit_df, class_dist_df, summary = audit_dataset(df_clean, images_root=img_dir)
    
    audit_csv_path = os.path.join(OUTPUT_DIR, "dataset_audit.csv")
    class_dist_path = os.path.join(OUTPUT_DIR, "class_distribution.csv")
    audit_df.to_csv(audit_csv_path, index=False)
    class_dist_df.to_csv(class_dist_path, index=False)
    print(f"Saved audit records to {audit_csv_path}")
    print("\nClass Distribution:")
    print(class_dist_df.to_string(index=False))
    
    print("\n[3/6] Generating stratified Train (70%), Val (15%), Test (15%) splits...")
    train_df, val_df, test_df = create_stratified_splits(
        df_clean,
        test_size=0.15,
        val_size=0.15,
        random_state=42
    )
    
    train_path = os.path.join(OUTPUT_DIR, "split_manifest_train.csv")
    val_path = os.path.join(OUTPUT_DIR, "split_manifest_val.csv")
    test_path = os.path.join(OUTPUT_DIR, "split_manifest_test.csv")
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    train_df.to_csv(os.path.join(MANIFEST_DIR, "train_manifest.csv"), index=False)
    val_df.to_csv(os.path.join(MANIFEST_DIR, "val_manifest.csv"), index=False)
    test_df.to_csv(os.path.join(MANIFEST_DIR, "test_manifest.csv"), index=False)
    
    print(f"Split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    print("\n[4/6] Testing PyTorch DataLoaders on real DDR images...")
    loaders = build_dataloaders(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        images_root=img_dir,
        img_size=224,
        batch_size=8,
        num_workers=0
    )
    
    train_batch_img, train_batch_labels, train_batch_ids = next(iter(loaders["train"]))
    val_batch_img, val_batch_labels, val_batch_ids = next(iter(loaders["val"]))
    
    print(f"Train Batch: Image Shape={train_batch_img.shape}, Labels={train_batch_labels.tolist()}")
    print(f"Val Batch:   Image Shape={val_batch_img.shape}, Labels={val_batch_labels.tolist()}")
    assert train_batch_img.shape == (8, 3, 224, 224), f"Unexpected train batch shape: {train_batch_img.shape}"
    assert val_batch_img.shape == (8, 3, 224, 224), f"Unexpected val batch shape: {val_batch_img.shape}"
    print("DataLoader shape and tensor validation PASSED.")
    
    print("\n[5/6] Generating dataset inspection visualization from real DDR samples...")
    fig, axes = plt.subplots(5, 4, figsize=(16, 18))
    fig.suptitle("Real DDR Retinal Fundus Preprocessing & Augmentation Pipeline", fontsize=16, fontweight="bold", y=0.99)
    
    col_titles = ["Raw Fundus Image", "Retina Cropped", "CLAHE LAB Enhanced", "Augmented Train Sample"]
    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=13, fontweight="bold", pad=8)
        
    for grade in range(5):
        sample_subset = df_clean[df_clean["numeric_label"] == grade]
        if len(sample_subset) > 0:
            sample_row = sample_subset.iloc[0]
            raw_bgr = cv2.imread(sample_row["filepath"])
            raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
            
            cropped_rgb = retina_circle_crop(raw_rgb)
            clahe_rgb = apply_clahe_lab(cropped_rgb, clip_limit=2.0)
            
            train_tf = get_transforms(img_size=224, is_training=True)
            tensor_img = train_tf(Image.fromarray(raw_rgb))
            mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
            aug_np = (tensor_img.numpy() * std + mean).transpose(1, 2, 0)
            aug_np = np.clip(aug_np, 0, 1)
            
            axes[grade, 0].imshow(raw_rgb)
            axes[grade, 0].set_ylabel(f"Grade {grade}\n({LABEL_MAPPING[grade]})", fontsize=12, fontweight="bold", labelpad=10)
            axes[grade, 1].imshow(cropped_rgb)
            axes[grade, 2].imshow(clahe_rgb)
            axes[grade, 3].imshow(aug_np)
            
        for col_idx in range(4):
            axes[grade, col_idx].axis("off")
            
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "dataset_inspection.png")
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved dataset inspection plot -> {plot_path}")
    
    config = {
        "dataset_name": "DDR (Dataset for Diabetic Retinopathy)",
        "source": "https://www.kaggle.com/datasets/mariaherrerot/ddrdataset",
        "num_classes": 5,
        "classes": {i: LABEL_MAPPING[i] for i in range(5)},
        "split_ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
        "total_verified_samples": int(len(df_clean)),
        "split_counts": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df))
        },
        "random_seed": 42,
        "image_size": 224,
        "preprocessing": {
            "retina_crop": True,
            "crop_threshold": 10,
            "crop_margin": 5,
            "clahe_enabled": True,
            "clahe_clip_limit": 2.0,
            "clahe_tile_grid": [8, 8],
            "normalization_mean": [0.485, 0.456, 0.406],
            "normalization_std": [0.229, 0.224, 0.225]
        },
        "augmentation_train_only": {
            "resize_margin": 32,
            "random_crop_scale": [0.85, 1.0],
            "horizontal_flip_p": 0.5,
            "vertical_flip_p": 0.2,
            "rotation_degrees": 15,
            "color_jitter": {"brightness": 0.15, "contrast": 0.15, "saturation": 0.15, "hue": 0.02}
        },
        "paths": {
            "raw_annotation_csv": csv_path,
            "images_dir": img_dir,
            "master_manifest": master_manifest_path,
            "train_manifest": train_path,
            "val_manifest": val_path,
            "test_manifest": test_path
        }
    }
    
    cfg_path = os.path.join(OUTPUT_DIR, "preprocessing_config.yaml")
    with open(cfg_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    with open("configs/dataset_config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
    print(f"Saved configurations to {cfg_path} and configs/dataset_config.yaml")
    print("\n[SUCCESS] Real DDR dataset processing completed successfully!")

if __name__ == "__main__":
    main()
