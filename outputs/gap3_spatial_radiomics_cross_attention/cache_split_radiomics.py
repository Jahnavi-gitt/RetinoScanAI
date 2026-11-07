import os
import sys
import time
import numpy as np
import pandas as pd
from PIL import Image

sys.path.append(os.path.abspath("."))
from src.features.radiomics import SpatialRadiomicsExtractor
from src.data.data_pipeline import retina_circle_crop, apply_clahe_lab

CACHE_DIR = "outputs/gap3_spatial_radiomics_cross_attention/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def extract_manifest_radiomics(manifest_csv: str, split_name: str, extractor: SpatialRadiomicsExtractor):
    df = pd.read_csv(manifest_csv)
    print(f"\n[{split_name.upper()}] Extracting localized radiomics for {len(df)} images from {manifest_csv}...")
    t0 = time.time()
    
    spatial_features = []
    labels = []
    image_ids = []
    
    total = len(df)
    for idx, row in df.iterrows():
        fp = row["filepath"]
        img_id = row["image_id"]
        label = int(row["numeric_label"])
        
        im = Image.open(fp).convert("RGB")
        im_np = np.array(im)
        cropped_np = retina_circle_crop(im_np)
        enhanced_np = apply_clahe_lab(cropped_np)
        
        feats = extractor.extract_image(enhanced_np)
        spatial_features.append(feats)
        labels.append(label)
        image_ids.append(img_id)
        
        if (idx + 1) % 1000 == 0 or (idx + 1) == total:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            print(f"  [{idx + 1}/{total}] extracted ({rate:.1f} img/s, {elapsed:.1f}s elapsed)")
            
    X_spatial = np.stack(spatial_features, axis=0)
    y = np.array(labels, dtype=np.int64)
    ids = np.array(image_ids)
    return X_spatial, y, ids

def main():
    print("=" * 70)
    print("  EXTRACTING AND CACHING REAL DDR SPATIALLY LOCALIZED RADIOMICS")
    print("=" * 70)
    t_start = time.time()
    
    train_manifest = "data_manifests/split_80_10_10/train_manifest.csv"
    val_manifest = "data_manifests/split_80_10_10/val_manifest.csv"
    test_manifest = "data_manifests/split_80_10_10/test_manifest.csv"
    
    extractor = SpatialRadiomicsExtractor(grid_size=(3, 3), target_size=(224, 224))
    
    X_train, y_train, ids_train = extract_manifest_radiomics(train_manifest, "train", extractor)
    
    X_val, y_val, ids_val = extract_manifest_radiomics(val_manifest, "val", extractor)
    
    X_test, y_test, ids_test = extract_manifest_radiomics(test_manifest, "test", extractor)
    
    print("\nFitting Normalizer on Train Split...")
    mean_ = np.mean(X_train, axis=0)
    scale_ = np.std(X_train, axis=0) + 1e-8
    
    X_train_norm = np.clip((X_train - mean_) / scale_, -3.0, 3.0).astype(np.float32)
    X_val_norm = np.clip((X_val - mean_) / scale_, -3.0, 3.0).astype(np.float32)
    X_test_norm = np.clip((X_test - mean_) / scale_, -3.0, 3.0).astype(np.float32)
    
    scaler_path = os.path.join(CACHE_DIR, "scaler.npz")
    np.savez_compressed(scaler_path, mean=mean_, scale=scale_)
    print(f"Saved normalizer -> {scaler_path}")
    
    train_cache = os.path.join(CACHE_DIR, "radiomics_train.npz")
    np.savez_compressed(train_cache, X_spatial=X_train, X_spatial_norm=X_train_norm, y=y_train, image_ids=ids_train)
    print(f"Saved train cache ({len(y_train)} records) -> {train_cache}")
    
    val_cache = os.path.join(CACHE_DIR, "radiomics_val.npz")
    np.savez_compressed(val_cache, X_spatial=X_val, X_spatial_norm=X_val_norm, y=y_val, image_ids=ids_val)
    print(f"Saved val cache ({len(y_val)} records) -> {val_cache}")
    
    test_cache = os.path.join(CACHE_DIR, "radiomics_test.npz")
    np.savez_compressed(test_cache, X_spatial=X_test, X_spatial_norm=X_test_norm, y=y_test, image_ids=ids_test)
    print(f"Saved test cache ({len(y_test)} records) -> {test_cache}")
    
    total_time = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"  RADIOMICS EXTRACTION COMPLETED SUCCESSFULLY IN {total_time:.1f}s ({total_time / 60:.1f} min)")
    print("=" * 70)

if __name__ == "__main__":
    main()
