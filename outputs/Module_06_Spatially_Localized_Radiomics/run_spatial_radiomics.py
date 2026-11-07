import os
import sys
import shutil
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

sys.path.append(os.path.abspath("."))

from src.features.radiomics import (
    SpatialRadiomicsExtractor,
    PATCH_FEATURE_NAMES,
    FEATURE_CATEGORIES,
    REGION_NAMES
)
from src.data.data_pipeline import create_stratified_splits

OUTPUT_DIR = "outputs/Module_06_Spatially_Localized_Radiomics"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_unit_verification(extractor: SpatialRadiomicsExtractor) -> bool:
    print("\n--- [Phase 1/4] Running Radiomics Verification Test ---")
    synthetic_img = np.random.randint(20, 220, (224, 224, 3), dtype=np.uint8)
    feats = extractor.extract_image(synthetic_img)
    
    print(f"  Extracted shape: {feats.shape} (Expected: (9, 32))")
    assert feats.shape == (9, 32), f"Expected shape (9, 32), got {feats.shape}"
    assert not np.isnan(feats).any(), "NaN detected in extracted radiomic features!"
    assert not np.isinf(feats).any(), "Inf detected in extracted radiomic features!"
    print("  [SUCCESS] Radiomics verification test passed: All 9 regions and 32 features verified finite.\n")
    return True

def get_dataset_splits():
    real_img_dir = "data/ddr/DR_grading/DR_grading"
    real_train_manifest = "data_manifests/train_manifest.csv"
    real_val_manifest = "data_manifests/val_manifest.csv"
    real_test_manifest = "data_manifests/test_manifest.csv"
    
    if os.path.isdir(real_img_dir) and len(os.listdir(real_img_dir)) > 50:
        print(f"[Dataset] Using real DDR images from {real_img_dir}")
        train_df = pd.read_csv(real_train_manifest)
        val_df = pd.read_csv(real_val_manifest)
        test_df = pd.read_csv(real_test_manifest)
        images_root = real_img_dir
    else:
        sample_manifest_path = "data_manifests/sample_manifest.csv"
        sample_img_dir = "data_manifests/sample_dataset/images"
        print(f"[Dataset] Using verified DDR development dataset ({sample_img_dir}).")
        sample_df = pd.read_csv(sample_manifest_path)
        train_df, val_df, test_df = create_stratified_splits(sample_df, test_size=0.15, val_size=0.15, random_state=42)
        images_root = sample_img_dir
        
    print(f"[Dataset] Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df, images_root

def create_feature_dictionary():
    descriptions = {
        "first_order_mean": ("First-Order", "Arithmetic mean of gray-level pixel intensities within the sub-region"),
        "first_order_variance": ("First-Order", "Spread of gray-level intensities around the mean"),
        "first_order_skewness": ("First-Order", "Asymmetry of intensity distribution (lesion asymmetry indicator)"),
        "first_order_kurtosis": ("First-Order", "Peakedness of intensity distribution (clustering of lesion intensities)"),
        "first_order_p10": ("First-Order", "10th percentile intensity value (background and hemorrhage floor)"),
        "first_order_p50": ("First-Order", "Median intensity value"),
        "first_order_p90": ("First-Order", "90th percentile intensity value (bright exudate ceiling)"),
        "first_order_energy": ("First-Order", "Sum of squared pixel intensities normalized by region area"),
        "first_order_entropy": ("First-Order", "Shannon entropy of intensity histogram (structural complexity)"),
        "first_order_rms": ("First-Order", "Root mean square intensity"),
        "glcm_contrast": ("GLCM", "Local intensity variation and edge contrast in spatial co-occurrence"),
        "glcm_dissimilarity": ("GLCM", "Mean difference in gray levels between neighboring pixel pairs"),
        "glcm_homogeneity": ("GLCM", "Inverse difference moment reflecting local structural uniformity"),
        "glcm_energy": ("GLCM", "Uniformity of GLCM matrix elements (texture regularity)"),
        "glcm_correlation": ("GLCM", "Linear dependency of gray levels between neighboring pixels"),
        "glcm_asm": ("GLCM", "Angular Second Moment measuring orderliness of texture"),
        "lbp_bin_0": ("LBP", "Uniform local binary pattern micro-texture histogram bin 0 (flat/smooth regions)"),
        "lbp_bin_1": ("LBP", "Uniform LBP micro-texture histogram bin 1 (fine edge transitions)"),
        "lbp_bin_2": ("LBP", "Uniform LBP micro-texture histogram bin 2 (micro-vascular patterns)"),
        "lbp_bin_3": ("LBP", "Uniform LBP micro-texture histogram bin 3 (texture irregularities)"),
        "lbp_bin_4": ("LBP", "Uniform LBP micro-texture histogram bin 4 (spot/microaneurysm candidates)"),
        "lbp_bin_5": ("LBP", "Uniform LBP micro-texture histogram bin 5 (intermediate texture changes)"),
        "lbp_bin_6": ("LBP", "Uniform LBP micro-texture histogram bin 6 (corner/lesion boundary patterns)"),
        "lbp_bin_7": ("LBP", "Uniform LBP micro-texture histogram bin 7 (vascular curvature patterns)"),
        "lbp_bin_8": ("LBP", "Uniform LBP micro-texture histogram bin 8 (dense texture transitions)"),
        "lbp_bin_9": ("LBP", "Uniform LBP micro-texture histogram bin 9 (non-uniform texture artifacts)"),
        "morph_gradient_mean": ("Morphology", "Average Sobel gradient magnitude reflecting microvascular boundary sharpness"),
        "morph_gradient_variance": ("Morphology", "Variance of gradient magnitude reflecting vascular branching heterogeneity"),
        "morph_canny_edge_density": ("Morphology", "Fraction of pixels identified as sharp anatomical or lesion edges"),
        "morph_exudate_bright_ratio": ("Morphology", "Fraction of bright pixels indicative of hard/soft exudates"),
        "morph_hemorrhage_dark_ratio": ("Morphology", "Fraction of dark non-background pixels indicative of retinal hemorrhages"),
        "morph_michelson_contrast": ("Morphology", "Michelson contrast ratio (Imax - Imin) / (Imax + Imin)")
    }
    
    rows = []
    idx = 0
    for r_idx, r_name in enumerate(REGION_NAMES):
        for f_name in PATCH_FEATURE_NAMES:
            category, desc = descriptions[f_name]
            full_col_name = f"{r_name}__{f_name}"
            rows.append({
                "feature_index": idx,
                "column_name": full_col_name,
                "region_index": r_idx,
                "region_name": r_name,
                "patch_feature": f_name,
                "category": category,
                "clinical_description": desc
            })
            idx += 1
            
    dict_df = pd.DataFrame(rows)
    dict_path = os.path.join(OUTPUT_DIR, "feature_dictionary.csv")
    dict_df.to_csv(dict_path, index=False)
    print(f"[Feature Dictionary] Saved 288-feature dictionary -> {dict_path}")
    return dict_df

def generate_visualizations(
    sample_img_path: str,
    extractor: SpatialRadiomicsExtractor,
    X_train_spatial: np.ndarray,
    dict_df: pd.DataFrame
):
    print("--- [Phase 4/4] Generating Publication Visualizations ---")
    
    img = cv2.imread(sample_img_path)
    if img is not None:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (300, 300))
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        ax = axes[0]
        ax.imshow(img_resized)
        for i in range(1, 3):
            ax.axvline(i * 100, color='yellow', linestyle='--', linewidth=2)
            ax.axhline(i * 100, color='yellow', linestyle='--', linewidth=2)
            
        region_labels = [
            ("Sup-Temporal", (50, 50)), ("Sup-Arcade", (150, 50)), ("Sup-Nasal", (250, 50)),
            ("Temporal-Mid", (50, 150)), ("Macula/Fovea", (150, 150)), ("Nasal-Mid", (250, 150)),
            ("Inf-Temporal", (50, 250)), ("Inf-Arcade", (150, 250)), ("Inf-Nasal", (250, 250))
        ]
        for name, (cx, cy) in region_labels:
            ax.text(cx, cy, name, color='white', fontsize=9, fontweight='bold',
                    ha='center', va='center', bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6))
        ax.set_title("Retina 3x3 Anatomical Sub-Region Grid", fontsize=12, fontweight='bold')
        ax.axis('off')
        
        sample_spatial = extractor.extract_image(img_rgb)
        contrast_idx = PATCH_FEATURE_NAMES.index("glcm_contrast")
        contrast_grid = sample_spatial[:, contrast_idx].reshape(3, 3)
        
        ax = axes[1]
        sns.heatmap(contrast_grid, annot=True, fmt=".2f", cmap="magma", ax=ax, cbar=True)
        ax.set_title("Localized GLCM Contrast (Texture Roughness)", fontsize=12, fontweight='bold')
        ax.set_xticklabels(["Temporal", "Center", "Nasal"])
        ax.set_yticklabels(["Superior", "Mid", "Inferior"])
        
        exudate_idx = PATCH_FEATURE_NAMES.index("morph_exudate_bright_ratio")
        exudate_grid = sample_spatial[:, exudate_idx].reshape(3, 3)
        
        ax = axes[2]
        sns.heatmap(exudate_grid, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax, cbar=True)
        ax.set_title("Localized Exudate Brightness Ratio (Lesions)", fontsize=12, fontweight='bold')
        ax.set_xticklabels(["Temporal", "Center", "Nasal"])
        ax.set_yticklabels(["Superior", "Mid", "Inferior"])
        
        plt.tight_layout()
        grid_vis_path = os.path.join(OUTPUT_DIR, "spatial_grid_visualization.png")
        plt.savefig(grid_vis_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved spatial grid visualization -> {grid_vis_path}")

    rep_features = [
        "first_order_mean", "first_order_variance", "first_order_skewness", "first_order_entropy",
        "glcm_contrast", "glcm_homogeneity", "glcm_energy", "glcm_correlation",
        "lbp_bin_0", "lbp_bin_2", "lbp_bin_6",
        "morph_gradient_mean", "morph_canny_edge_density", "morph_exudate_bright_ratio", "morph_michelson_contrast"
    ]
    rep_indices = [PATCH_FEATURE_NAMES.index(f) for f in rep_features]
    
    mean_patch_feats = np.mean(X_train_spatial, axis=1)
    rep_matrix = mean_patch_feats[:, rep_indices]
    corr_df = pd.DataFrame(rep_matrix, columns=[f.replace("first_order_", "FO_").replace("glcm_", "GLCM_").replace("morph_", "M_") for f in rep_features])
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_df.corr(), annot=True, fmt=".2f", cmap="vlag", center=0, vmin=-1, vmax=1, square=True)
    plt.title("Radiomic Feature Correlation Matrix (Cross-Category)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    corr_vis_path = os.path.join(OUTPUT_DIR, "feature_correlation_matrix.png")
    plt.savefig(corr_vis_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved feature correlation matrix -> {corr_vis_path}")

def main():
    t_start = time.time()
    print("==================================================================")
    print("  MODULE 06: SPATIALLY LOCALIZED RADIOMICS EXTRACTION")
    print("  Grid Configuration: 3x3 (9 regions) | Features per patch: 32")
    print("  Total Feature Output per Image: 9 x 32 = 288 descriptors")
    print("==================================================================")
    
    extractor = SpatialRadiomicsExtractor(grid_size=(3, 3), target_size=(224, 224))
    run_unit_verification(extractor)
    
    train_df, val_df, test_df, images_root = get_dataset_splits()
    
    print("\n--- [Phase 2/4] Extracting Spatially Localized Radiomics ---")
    print(f"Extracting Train Split ({len(train_df)} samples)...")
    X_train_spatial, y_train, train_ids = extractor.extract_manifest(train_df, images_root)
    print(f"  Train Tensor Shape: {X_train_spatial.shape}")
    
    print(f"Extracting Validation Split ({len(val_df)} samples)...")
    X_val_spatial, y_val, val_ids = extractor.extract_manifest(val_df, images_root)
    print(f"  Val Tensor Shape: {X_val_spatial.shape}")
    
    print(f"Extracting Test Split ({len(test_df)} samples)...")
    X_test_spatial, y_test, test_ids = extractor.extract_manifest(test_df, images_root)
    print(f"  Test Tensor Shape: {X_test_spatial.shape}")
    
    print("\n--- [Phase 3/4] Fitting Spatial Radiomic Normalizer ---")
    extractor.fit_normalizer(X_train_spatial)
    X_train_norm = extractor.transform(X_train_spatial)
    X_val_norm = extractor.transform(X_val_spatial)
    X_test_norm = extractor.transform(X_test_spatial)
    
    dict_df = create_feature_dictionary()
    flat_cols = dict_df["column_name"].tolist()
    
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, "radiomic_features_train.npz"),
        X_spatial=X_train_spatial,
        X_spatial_norm=X_train_norm,
        X_flat=X_train_spatial.reshape(len(X_train_spatial), -1),
        X_flat_norm=X_train_norm.reshape(len(X_train_norm), -1),
        y=y_train,
        image_ids=train_ids
    )
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, "radiomic_features_val.npz"),
        X_spatial=X_val_spatial,
        X_spatial_norm=X_val_norm,
        X_flat=X_val_spatial.reshape(len(X_val_spatial), -1),
        X_flat_norm=X_val_norm.reshape(len(X_val_norm), -1),
        y=y_val,
        image_ids=val_ids
    )
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, "radiomic_features_test.npz"),
        X_spatial=X_test_spatial,
        X_spatial_norm=X_test_norm,
        X_flat=X_test_spatial.reshape(len(X_test_spatial), -1),
        X_flat_norm=X_test_norm.reshape(len(X_test_norm), -1),
        y=y_test,
        image_ids=test_ids
    )
    print("Saved .npz feature archives for train, val, test.")
    
    train_flat = pd.DataFrame(X_train_spatial.reshape(len(X_train_spatial), -1), columns=flat_cols)
    train_flat.insert(0, "numeric_label", y_train)
    train_flat.insert(0, "image_id", train_ids)
    train_flat.to_csv(os.path.join(OUTPUT_DIR, "radiomic_features_train.csv"), index=False)
    
    val_flat = pd.DataFrame(X_val_spatial.reshape(len(X_val_spatial), -1), columns=flat_cols)
    val_flat.insert(0, "numeric_label", y_val)
    val_flat.insert(0, "image_id", val_ids)
    val_flat.to_csv(os.path.join(OUTPUT_DIR, "radiomic_features_val.csv"), index=False)
    print("Saved tabular .csv datasets.")
    
    sample_img_id = train_ids[0]
    sample_img_path = os.path.join(images_root, sample_img_id)
    if not os.path.exists(sample_img_path):
        sample_img_path = os.path.join(images_root, f"{sample_img_id}.jpg")
        
    generate_visualizations(sample_img_path, extractor, X_train_spatial, dict_df)
    
    shutil.copy("src/features/radiomics.py", os.path.join(OUTPUT_DIR, "spatial_radiomics_extractor.py"))
    
    total_time = time.time() - t_start
    print(f"\n==================================================================")
    print(f"  MODULE 06 COMPLETED IN {total_time:.1f}s")
    print(f"  Train samples processed:      {len(train_df)}")
    print(f"  Val samples processed:        {len(val_df)}")
    print(f"  Test samples processed:       {len(test_df)}")
    print(f"  Spatial tensor shape:         (N, 9, 32)")
    print(f"  Flattened features per image: 288")
    print(f"==================================================================")

if __name__ == "__main__":
    main()
