import os
import sys
import time
import numpy as np
import pandas as pd
from PIL import Image

sys.path.append(os.path.abspath("."))

from src.features.radiomics import SpatialRadiomicsExtractor
from src.data.data_pipeline import retina_circle_crop, apply_clahe_lab

OUTPUT_DIR = "outputs/Module_06_Spatially_Localized_Radiomics"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def select_stratified_real_ddr(manifest_path: str, n_per_grade: int, seed: int = 42) -> pd.DataFrame:
    df = pd.read_csv(manifest_path)
    df["exists"] = df["filepath"].apply(os.path.exists)
    df = df[df["exists"]].reset_index(drop=True)
    
    samples = []
    for g in range(5):
        sub = df[df["numeric_label"] == g]
        n_take = min(n_per_grade, len(sub))
        sampled = sub.sample(n=n_take, random_state=seed)
        samples.append(sampled)
    res = pd.concat(samples).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return res

def extract_real_radiomics():
    print("==================================================================")
    print("  EXTRACTING RADIOMICS FROM REAL DDR CLINICAL FUNDUS DATASET")
    print("==================================================================")
    t0 = time.time()
    
    train_manifest = "data_manifests/train_manifest.csv"
    val_manifest = "data_manifests/val_manifest.csv"
    test_manifest = "data_manifests/test_manifest.csv"
    
    train_df = select_stratified_real_ddr(train_manifest, n_per_grade=100, seed=42)
    val_df = select_stratified_real_ddr(val_manifest, n_per_grade=30, seed=42)
    test_df = select_stratified_real_ddr(test_manifest, n_per_grade=30, seed=42)
    
    print(f"Sampled Real DDR: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    print(f"Train Grade Counts: {dict(train_df['numeric_label'].value_counts())}")
    
    extractor = SpatialRadiomicsExtractor(grid_size=(3, 3), target_size=(224, 224))
    
    def process_split(split_df, split_name):
        t_split = time.time()
        spatial_list, y_list, id_list = [], [], []
        print(f"\nProcessing {split_name} ({len(split_df)} images)...")
        
        for i, (_, row) in enumerate(split_df.iterrows()):
            fp = row["filepath"]
            img_id = row["image_id"]
            label = int(row["numeric_label"])
            
            img = Image.open(fp).convert("RGB")
            img_np = np.array(img)
            cropped_np = retina_circle_crop(img_np)
            enhanced_np = apply_clahe_lab(cropped_np)
            
            feats = extractor.extract_image(enhanced_np)
            spatial_list.append(feats)
            y_list.append(label)
            id_list.append(img_id)
            
            if (i + 1) % 50 == 0 or (i + 1) == len(split_df):
                print(f"  [{i+1}/{len(split_df)}] images extracted in {time.time() - t_split:.1f}s")
                
        return np.stack(spatial_list, axis=0), np.array(y_list, dtype=np.int64), id_list
        
    X_train, y_train, ids_train = process_split(train_df, "Train")
    X_val, y_val, ids_val = process_split(val_df, "Val")
    X_test, y_test, ids_test = process_split(test_df, "Test")
    
    print("\nFitting Normalizer on Real DDR Human Fundus Images...")
    extractor.fit_normalizer(X_train)
    
    X_train_norm = extractor.transform(X_train)
    X_val_norm = extractor.transform(X_val)
    X_test_norm = extractor.transform(X_test)
    
    print(f"Real DDR Normalizer: mean shape={extractor.mean_.shape}, scale shape={extractor.scale_.shape}")
    print(f"Normalized Train Min={X_train_norm.min():.2f}, Max={X_train_norm.max():.2f}, Mean={X_train_norm.mean():.4f}")
    
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, "radiomic_features_train.npz"),
        X_spatial=X_train,
        X_spatial_norm=X_train_norm,
        X_flat=X_train.reshape(len(X_train), -1),
        X_flat_norm=X_train_norm.reshape(len(X_train_norm), -1),
        y=y_train,
        image_ids=ids_train
    )
    
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, "radiomic_features_val.npz"),
        X_spatial=X_val,
        X_spatial_norm=X_val_norm,
        X_flat=X_val.reshape(len(X_val), -1),
        X_flat_norm=X_val_norm.reshape(len(X_val_norm), -1),
        y=y_val,
        image_ids=ids_val
    )
    
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, "radiomic_features_test.npz"),
        X_spatial=X_test,
        X_spatial_norm=X_test_norm,
        X_flat=X_test.reshape(len(X_test), -1),
        X_flat_norm=X_test_norm.reshape(len(X_test_norm), -1),
        y=y_test,
        image_ids=ids_test
    )
    
    np.save(os.path.join(OUTPUT_DIR, "normalizer_mean.npy"), extractor.mean_)
    np.save(os.path.join(OUTPUT_DIR, "normalizer_scale.npy"), extractor.scale_)
    
    train_df.to_csv(os.path.join(OUTPUT_DIR, "active_real_train_manifest.csv"), index=False)
    val_df.to_csv(os.path.join(OUTPUT_DIR, "active_real_val_manifest.csv"), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, "active_real_test_manifest.csv"), index=False)
    
    total_time = time.time() - t0
    print(f"\n[SUCCESS] Extracted and normalized {len(train_df) + len(val_df) + len(test_df)} real DDR images in {total_time:.1f}s.")
    print(f"Files saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    extract_real_radiomics()
