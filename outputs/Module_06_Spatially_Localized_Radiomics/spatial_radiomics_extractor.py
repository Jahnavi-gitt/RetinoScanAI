import os
from typing import List, Tuple, Dict, Any, Union
import numpy as np
import cv2
import pandas as pd
from scipy import stats
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

PATCH_FEATURE_NAMES = [
    "first_order_mean",
    "first_order_variance",
    "first_order_skewness",
    "first_order_kurtosis",
    "first_order_p10",
    "first_order_p50",
    "first_order_p90",
    "first_order_energy",
    "first_order_entropy",
    "first_order_rms",
    "glcm_contrast",
    "glcm_dissimilarity",
    "glcm_homogeneity",
    "glcm_energy",
    "glcm_correlation",
    "glcm_asm",
    "lbp_bin_0",
    "lbp_bin_1",
    "lbp_bin_2",
    "lbp_bin_3",
    "lbp_bin_4",
    "lbp_bin_5",
    "lbp_bin_6",
    "lbp_bin_7",
    "lbp_bin_8",
    "lbp_bin_9",
    "morph_gradient_mean",
    "morph_gradient_variance",
    "morph_canny_edge_density",
    "morph_exudate_bright_ratio",
    "morph_hemorrhage_dark_ratio",
    "morph_michelson_contrast"
]

FEATURE_CATEGORIES = {
    "first_order": PATCH_FEATURE_NAMES[0:10],
    "glcm": PATCH_FEATURE_NAMES[10:16],
    "lbp": PATCH_FEATURE_NAMES[16:26],
    "morphological": PATCH_FEATURE_NAMES[26:32]
}

REGION_NAMES = [
    "R00_Superior_Temporal",
    "R01_Superior_Arcade",
    "R02_Superior_Nasal",
    "R10_Temporal_Mid",
    "R11_Central_Macula_Fovea",
    "R12_Nasal_Mid",
    "R20_Inferior_Temporal",
    "R21_Inferior_Arcade",
    "R22_Inferior_Nasal"
]

def extract_patch_radiomics(patch: np.ndarray) -> np.ndarray:
    if patch.ndim == 3:
        patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        
    patch_f = patch.astype(np.float64)
    h, w = patch.shape
    total_pixels = h * w
    
    mean_val = float(np.mean(patch_f))
    var_val = float(np.var(patch_f))
    flat_patch = patch_f.ravel()
    
    if var_val < 1e-6:
        skew_val = 0.0
        kurt_val = 0.0
    else:
        skew_val = float(stats.skew(flat_patch))
        kurt_val = float(stats.kurtosis(flat_patch))
        
    p10 = float(np.percentile(patch_f, 10))
    p50 = float(np.median(patch_f))
    p90 = float(np.percentile(patch_f, 90))
    energy_val = float(np.sum(patch_f ** 2) / total_pixels)
    
    counts, _ = np.histogram(patch, bins=32, range=(0, 256), density=True)
    entropy_val = float(stats.entropy(counts + 1e-10))
    rms_val = float(np.sqrt(energy_val))
    
    quantized = (patch // 4).astype(np.uint8)
    glcm = graycomatrix(
        quantized,
        distances=[1, 2],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=64,
        symmetric=True,
        normed=True
    )
    glcm_contrast = float(np.mean(graycoprops(glcm, 'contrast')))
    glcm_dissimilarity = float(np.mean(graycoprops(glcm, 'dissimilarity')))
    glcm_homogeneity = float(np.mean(graycoprops(glcm, 'homogeneity')))
    glcm_energy = float(np.mean(graycoprops(glcm, 'energy')))
    glcm_correlation = float(np.mean(graycoprops(glcm, 'correlation')))
    if np.isnan(glcm_correlation):
        glcm_correlation = 0.0
    glcm_asm = float(np.mean(graycoprops(glcm, 'ASM')))
    
    lbp = local_binary_pattern(patch, P=8, R=1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)
    lbp_feats = [float(h_val) for h_val in hist]
    
    sobelx = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    grad_mean = float(np.mean(grad_mag))
    grad_var = float(np.var(grad_mag))
    
    canny = cv2.Canny(patch, 50, 150)
    edge_density = float(np.mean(canny > 0))
    
    exudate_ratio = float(np.mean(patch > 180))
    hemorrhage_ratio = float(np.mean((patch > 10) & (patch < 60)))
    
    min_val, max_val = float(np.min(patch_f)), float(np.max(patch_f))
    michelson = float((max_val - min_val) / (max_val + min_val + 1e-5))
    
    features = [
        mean_val, var_val, skew_val, kurt_val, p10, p50, p90, energy_val, entropy_val, rms_val,
        glcm_contrast, glcm_dissimilarity, glcm_homogeneity, glcm_energy, glcm_correlation, glcm_asm,
        *lbp_feats,
        grad_mean, grad_var, edge_density, exudate_ratio, hemorrhage_ratio, michelson
    ]
    return np.array(features, dtype=np.float32)

class SpatialRadiomicsExtractor:
    def __init__(self, grid_size: Tuple[int, int] = (3, 3), target_size: Tuple[int, int] = (224, 224)):
        self.grid_size = grid_size
        self.target_size = target_size
        self.num_regions = grid_size[0] * grid_size[1]
        self.features_per_patch = len(PATCH_FEATURE_NAMES)
        self.total_features = self.num_regions * self.features_per_patch
        self.mean_ = None
        self.scale_ = None

    def partition_grid(self, image: np.ndarray) -> List[np.ndarray]:
        if image.shape[:2] != self.target_size:
            image = cv2.resize(image, self.target_size)
            
        h, w = image.shape[:2]
        gh, gw = self.grid_size
        patch_h, patch_w = h // gh, w // gw
        
        patches = []
        for r in range(gh):
            for c in range(gw):
                y1 = r * patch_h
                y2 = (r + 1) * patch_h if r < gh - 1 else h
                x1 = c * patch_w
                x2 = (c + 1) * patch_w if c < gw - 1 else w
                patches.append(image[y1:y2, x1:x2])
        return patches

    def extract_image(self, image_input: Union[str, np.ndarray]) -> np.ndarray:
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            if image is None:
                raise FileNotFoundError(f"Image not found at {image_input}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image_input

        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        patches = self.partition_grid(gray)
        region_features = []
        for p in patches:
            feats = extract_patch_radiomics(p)
            region_features.append(feats)
            
        return np.stack(region_features, axis=0)

    def extract_manifest(
        self,
        manifest_df: pd.DataFrame,
        images_root: str
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        X_list = []
        y_list = []
        id_list = []
        
        for _, row in manifest_df.iterrows():
            img_id = str(row["image_id"])
            lbl = int(row["numeric_label"])
            
            img_path = os.path.join(images_root, img_id)
            if not os.path.exists(img_path):
                img_path = os.path.join(images_root, f"{img_id}.jpg")
                
            spatial_feats = self.extract_image(img_path)
            X_list.append(spatial_feats)
            y_list.append(lbl)
            id_list.append(img_id)
            
        return np.stack(X_list, axis=0), np.array(y_list, dtype=np.int64), id_list

    def fit_normalizer(self, X_spatial: np.ndarray):
        flat_feats = X_spatial.reshape(-1, self.features_per_patch)
        self.mean_ = np.mean(flat_feats, axis=0, keepdims=True)
        self.scale_ = np.std(flat_feats, axis=0, keepdims=True) + 1e-6

    def transform(self, X_spatial: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Normalizer has not been fitted yet! Call fit_normalizer() first.")
        return (X_spatial - self.mean_) / self.scale_
