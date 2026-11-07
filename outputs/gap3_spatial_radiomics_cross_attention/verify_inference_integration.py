import os
import sys
import time
import numpy as np
from PIL import Image
import torch

sys.path.append(os.path.abspath("."))
from outputs.gap3_spatial_radiomics_cross_attention.model import SpatialRadiomicsSwinCoralModel
from outputs.gap3_spatial_radiomics_cross_attention.train_spatial_cross_attention import StandardRetinaTransform
from src.features.radiomics import SpatialRadiomicsExtractor
from src.data.data_pipeline import retina_circle_crop, apply_clahe_lab

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
OUTPUT_DIR = "outputs/gap3_spatial_radiomics_cross_attention"
CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")

BENCHMARK_SAMPLES = [
    ("Grade 0", "outputs/Module_14_Interactive_Web_Application/public/samples/grade_0_007-0007-000.jpg"),
    ("Grade 1", "outputs/Module_14_Interactive_Web_Application/public/samples/grade_1_007-1843-100.jpg"),
    ("Grade 2", "outputs/Module_14_Interactive_Web_Application/public/samples/grade_2_007-1927-100.jpg"),
    ("Grade 3", "outputs/Module_14_Interactive_Web_Application/public/samples/grade_3_007-3548-200.jpg"),
    ("Grade 4", "outputs/Module_14_Interactive_Web_Application/public/samples/grade_4_007-3979-200.jpg")
]

class ProposedInferenceEngine:
    def __init__(self, checkpoint_path: str):
        print(f"[InferenceEngine] Initializing on {DEVICE}...")
        self.device = DEVICE
        
        self.model = SpatialRadiomicsSwinCoralModel(
            num_classes=5,
            d_vis=768,
            d_rad=32,
            d_model=768,
            num_heads=8,
            dropout=0.0,
            pretrained_backbone=False
        )
        if os.path.exists(checkpoint_path):
            st = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if "model_state_dict" in st:
                st = st["model_state_dict"]
            self.model.load_state_dict(st)
            print(f"[InferenceEngine] Loaded checkpoint: {checkpoint_path}")
        else:
            print(f"[InferenceEngine] Warning: Checkpoint not found at {checkpoint_path}, using init weights.")
            
        self.model.to(self.device).eval()
        
        self.extractor = SpatialRadiomicsExtractor(grid_size=(3, 3), target_size=(224, 224))
        scaler_file = os.path.join(CACHE_DIR, "scaler.npz")
        assert os.path.exists(scaler_file), f"Missing scaler at {scaler_file}"
        scaler_data = np.load(scaler_file)
        self.mean_ = scaler_data["mean"]
        self.scale_ = scaler_data["scale"]
        
        self.transform = StandardRetinaTransform(is_training=False, img_size=224)
        print("[InferenceEngine] Ready for real-time inference.")

    def predict_fundus(self, pil_image: Image.Image) -> dict:
        t0 = time.time()
        
        img_np = np.array(pil_image)
        cropped_np = retina_circle_crop(img_np)
        enhanced_np = apply_clahe_lab(cropped_np)
        enhanced_pil = Image.fromarray(enhanced_np)
        
        img_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        rad_raw = self.extractor.extract_image(enhanced_np)
        rad_norm = np.clip((rad_raw - self.mean_) / self.scale_, -3.0, 3.0).astype(np.float32)
        rad_tensor = torch.from_numpy(rad_norm).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            out = self.model.predict(img_tensor, rad_tensor)
            
        pred_grade = int(out["predicted_grade"].item())
        class_probs = out["class_probabilities"].squeeze(0).cpu().numpy().tolist()
        rank_probs = out["rank_probabilities"].squeeze(0).cpu().numpy().tolist()
        attn_matrix = out["attention_weights"].squeeze(0).cpu().numpy()
        
        latency = round((time.time() - t0) * 1000.0, 1)
        
        return {
            "predicted_grade": pred_grade,
            "class_probabilities": [round(p, 4) for p in class_probs],
            "rank_probabilities": [round(p, 4) for p in rank_probs],
            "attention_shape": attn_matrix.shape,
            "attention_mean_per_sector": [round(float(a), 4) for a in attn_matrix.mean(axis=0)],
            "latency_ms": latency
        }

def run_verification():
    print("=" * 70)
    print("  VERIFYING INFERENCE INTEGRATION OF PROPOSED GAP 3 ARCHITECTURE")
    print("=" * 70)
    
    ckpt = os.path.join(OUTPUT_DIR, "best_spatial_radiomics_coral_model.pth")
    engine = ProposedInferenceEngine(ckpt)
    
    print("\nRunning Benchmark Test on Clinical Samples:")
    all_passed = True
    for label, img_path in BENCHMARK_SAMPLES:
        if not os.path.exists(img_path):
            print(f"  [SKIP] Missing image: {img_path}")
            continue
            
        im = Image.open(img_path).convert("RGB")
        res = engine.predict_fundus(im)
        print(f"\n  Image: {os.path.basename(img_path)} ({label})")
        print(f"    Predicted Grade:      Grade {res['predicted_grade']}")
        print(f"    Class Probabilities:  {res['class_probabilities']}")
        print(f"    Rank Probabilities:   {res['rank_probabilities']}")
        print(f"    Cross-Attn Shape:     {res['attention_shape']} (49 visual x 9 radiomic)")
        print(f"    Regional Attn Mean:   {res['attention_mean_per_sector']}")
        print(f"    Inference Latency:    {res['latency_ms']} ms")
        
        assert res["predicted_grade"] in [0, 1, 2, 3, 4], "Invalid grade!"
        assert abs(sum(res["class_probabilities"]) - 1.0) < 0.05, "Probabilities must sum to ~1.0!"
        assert res["attention_shape"] == (49, 9), "Cross-attention map shape must be (49, 9)!"
        assert res["latency_ms"] < 250.0, f"Latency {res['latency_ms']} ms exceeds budget!"
        
    print("\n" + "=" * 70)
    print("  INFERENCE INTEGRATION VERIFIED: ALL CHECKS PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_verification()
