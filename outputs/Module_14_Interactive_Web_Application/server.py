import os
import sys
import io
import time
import json
import base64
from http.server import HTTPServer, SimpleHTTPRequestHandler
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.abspath("."))

from src.models.swin import SwinTinyBaseline
from src.models.integrated_model import IntegratedRetinopathyModel
from src.models.coral import coral_predict, coral_probabilities
from src.features.radiomics import SpatialRadiomicsExtractor
from src.data.data_pipeline import get_transforms, LABEL_MAPPING, retina_circle_crop, apply_clahe_lab
from outputs.Module_03_Baseline_Swin_Tiny.train_full_ddr_swin_coral import SwinTinyCoralModel, StandardRetinaTransform
from outputs.gap3_spatial_radiomics_cross_attention.model import SpatialRadiomicsSwinCoralModel

PUBLIC_DIR = os.path.abspath("outputs/Module_14_Interactive_Web_Application/public")
PORT = 8080

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

ANATOMICAL_ZONES = [
    "R0: Superotemporal",
    "R1: Sup. Vascular Arcade",
    "R2: Superonasal",
    "R3: Nasal / Optic Disc",
    "R4: Central Macula / Fovea",
    "R5: Temporal Periphery",
    "R6: Inferotemporal",
    "R7: Inf. Vascular Arcade",
    "R8: Inferonasal"
]

BENCHMARK_SAMPLES = {
    "grade_0": {
        "id": "grade_0",
        "grade": 0,
        "name": "Grade 0: No DR (Healthy Retina)",
        "file": "outputs/Module_14_Interactive_Web_Application/public/samples/grade_0_007-0007-000.jpg",
        "url": "/samples/grade_0_007-0007-000.jpg"
    },
    "grade_1": {
        "id": "grade_1",
        "grade": 1,
        "name": "Grade 1: Mild NPDR (Microaneurysms)",
        "file": "outputs/Module_14_Interactive_Web_Application/public/samples/grade_1_007-1843-100.jpg",
        "url": "/samples/grade_1_007-1843-100.jpg"
    },
    "grade_2": {
        "id": "grade_2",
        "grade": 2,
        "name": "Grade 2: Moderate NPDR (Hemorrhages & Exudates)",
        "file": "outputs/Module_14_Interactive_Web_Application/public/samples/grade_2_007-1927-100.jpg",
        "url": "/samples/grade_2_007-1927-100.jpg"
    },
    "grade_3": {
        "id": "grade_3",
        "grade": 3,
        "name": "Grade 3: Severe NPDR (Venous Beading & Cotton-Wool)",
        "file": "outputs/Module_14_Interactive_Web_Application/public/samples/grade_3_007-3548-200.jpg",
        "url": "/samples/grade_3_007-3548-200.jpg"
    },
    "grade_4": {
        "id": "grade_4",
        "grade": 4,
        "name": "Grade 4: Proliferative DR (Neovascularization)",
        "file": "outputs/Module_14_Interactive_Web_Application/public/samples/grade_4_007-3979-200.jpg",
        "url": "/samples/grade_4_007-3979-200.jpg"
    }
}

REFERRAL_GUIDELINES = {
    0: {
        "urgency": "Routine / Low Risk",
        "color": "#00e676",
        "follow_up": "12 to 24 Months",
        "recommendation": "Normal retinal examination with no apparent diabetic retinopathy. Maintain strict glycemic and blood pressure control. Schedule annual routine ophthalmic screening.",
        "icd_code": "E11.9 / E10.9 (Uncomplicated Diabetes)"
    },
    1: {
        "urgency": "Mild / Low-Moderate Risk",
        "color": "#00b0ff",
        "follow_up": "6 to 12 Months",
        "recommendation": "Microaneurysms present without macular edema. Reinforce glycemic targets (HbA1c < 7.0%). Routine dilated fundus examination recommended in 6-12 months.",
        "icd_code": "E11.319 (Mild Nonproliferative Retinopathy)"
    },
    2: {
        "urgency": "Moderate / Actionable Risk",
        "color": "#ffab00",
        "follow_up": "3 to 6 Months",
        "recommendation": "Moderate non-proliferative retinopathy with scattered microaneurysms, intraretinal hemorrhages, and lipid exudates. Comprehensive ophthalmic evaluation recommended within 3 to 6 months.",
        "icd_code": "E11.329 (Moderate Nonproliferative Retinopathy)"
    },
    3: {
        "urgency": "Severe / High Risk",
        "color": "#ff6d00",
        "follow_up": "2 to 4 Weeks",
        "recommendation": "Severe non-proliferative retinopathy meeting the 4-2-1 international criteria (blot hemorrhages in 4 quadrants, venous beading in 2+ quadrants, or IRMA in 1+ quadrant). High risk of progression to proliferative stage. Refer to retina specialist within 2-4 weeks.",
        "icd_code": "E11.339 (Severe Nonproliferative Retinopathy)"
    },
    4: {
        "urgency": "Critical / Emergent Referral",
        "color": "#ff1744",
        "follow_up": "24 to 48 Hours",
        "recommendation": "High-risk Proliferative Diabetic Retinopathy characterized by neovascularization of the disc (NVD) or retina (NVE), preretinal/vitreous hemorrhage, or fibrovascular proliferation. Immediate emergent referral for prompt panretinal photocoagulation (PRP) or anti-VEGF injection to prevent severe permanent vision loss.",
        "icd_code": "E11.359 (Proliferative Diabetic Retinopathy)"
    }
}

class ModelEngine:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        print(f"[ModelEngine] Initializing on {DEVICE}...")
        t0 = time.time()
        
        self.model = SwinTinyCoralModel(num_classes=5)
        clean_ckpt = "outputs/Module_03_Baseline_Swin_Tiny/clean_run/best_model_clean.pth"
        st = torch.load(clean_ckpt, map_location="cpu", weights_only=False)
        if "model_state_dict" in st:
            st = st["model_state_dict"]
        self.model.load_state_dict(st)
        self.model.to(DEVICE).eval()
        
        self.features = None
        self.gradients = None
        target_layer = self.model.backbone.features[-1]
        target_layer.register_forward_hook(self._hook_fwd)
        target_layer.register_full_backward_hook(self._hook_bwd)
        
        self.extractor = SpatialRadiomicsExtractor()
        train_data = np.load("outputs/Module_06_Spatially_Localized_Radiomics/radiomic_features_train.npz")
        self.extractor.fit_normalizer(train_data["X_spatial"])
        
        self.clean_transform = StandardRetinaTransform(is_training=False, img_size=224)
        
        self.gap3_model = None
        self.gap3_scaler = None
        self.gap3_checkpoint = "outputs/gap3_spatial_radiomics_cross_attention/best_spatial_radiomics_coral_model.pth"
        self.gap3_scaler_path = "outputs/gap3_spatial_radiomics_cross_attention/cache/scaler.npz"
        print(f"[ModelEngine] Ready with best_model_clean.pth in {time.time() - t0:.2f}s")

    def _get_gap3_model(self):
        if self.gap3_model is None:
            if not os.path.exists(self.gap3_checkpoint):
                raise FileNotFoundError(f"Gap 3 checkpoint not found at: {self.gap3_checkpoint}")
            if not os.path.exists(self.gap3_scaler_path):
                raise FileNotFoundError(f"Gap 3 scaler not found at: {self.gap3_scaler_path}")
                
            print(f"[ModelEngine] Loading Gap 3 model from {self.gap3_checkpoint}...")
            m = SpatialRadiomicsSwinCoralModel(
                num_classes=5,
                d_vis=768,
                d_rad=32,
                d_model=768,
                num_heads=8,
                dropout=0.0,
                pretrained_backbone=False
            )
            st = torch.load(self.gap3_checkpoint, map_location="cpu", weights_only=False)
            if "model_state_dict" in st:
                st = st["model_state_dict"]
            m.load_state_dict(st)
            self.gap3_model = m.to(DEVICE).eval()
            
            sc = np.load(self.gap3_scaler_path)
            self.gap3_scaler = {
                "mean": sc["mean"],
                "scale": sc["scale"]
            }
            print("[ModelEngine] Gap 3 model and training-fitted scaler ready.")
        return self.gap3_model, self.gap3_scaler

    def _hook_fwd(self, module, inp, outp):
        self.features = outp

    def _hook_bwd(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def compute_gradcam(self, img_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        self.model.zero_grad()
        img_tensor = img_tensor.clone().detach().requires_grad_(True)
        logits = self.model(img_tensor)
        target_task = min(max(0, target_class - 1), 3) if target_class > 0 else 0
        logits[0, target_task].backward()
        
        weights = self.gradients.mean(dim=(1, 2), keepdim=True)
        cam = (weights * self.features).sum(dim=-1, keepdim=True)
        cam = F.relu(cam).squeeze(-1)
        
        cam_resized = F.interpolate(
            cam.unsqueeze(1),
            size=(224, 224),
            mode="bicubic",
            align_corners=False
        ).squeeze().cpu().detach().numpy()
        
        cam_min, cam_max = cam_resized.min(), cam_resized.max()
        if cam_max > cam_min:
            cam_norm = (cam_resized - cam_min) / (cam_max - cam_min)
        else:
            cam_norm = np.zeros_like(cam_resized)
        return cam_norm

    def predict(self, pil_image: Image.Image, model_type: str = "baseline") -> dict:
        t0 = time.time()
        img_np = np.array(pil_image)
        cropped_np = retina_circle_crop(img_np)
        enhanced_np = apply_clahe_lab(cropped_np)
        enhanced_pil = Image.fromarray(enhanced_np)
        
        img_tensor = self.clean_transform(pil_image).unsqueeze(0).to(DEVICE)
        
        rad_raw = self.extractor.extract_image(enhanced_np)
        
        if model_type == "gap3":
            model_gap3, scaler_gap3 = self._get_gap3_model()
            rad_norm = np.clip((rad_raw - scaler_gap3["mean"]) / scaler_gap3["scale"], -3.0, 3.0).astype(np.float32)
            rad_tensor = torch.from_numpy(rad_norm).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                out = model_gap3.predict(img_tensor, rad_tensor)
                
            pred_grade = int(out["predicted_grade"].item())
            coral_probs = out["class_probabilities"].squeeze(0).cpu().numpy().tolist()
            attn_matrix = out["attention_weights"].squeeze(0).cpu().numpy()
            
            reg_weights = (attn_matrix.mean(axis=0) / (attn_matrix.mean(axis=0).sum() + 1e-8)).tolist()
            
            spatial_att = attn_matrix.mean(axis=-1).reshape(7, 7)
            spatial_att_resized = cv2.resize(spatial_att, (224, 224), interpolation=cv2.INTER_CUBIC)
            att_min, att_max = spatial_att_resized.min(), spatial_att_resized.max()
            if att_max > att_min:
                att_norm = (spatial_att_resized - att_min) / (att_max - att_min)
            else:
                att_norm = np.zeros_like(spatial_att_resized)
                
            color_cam = plt.colormaps["jet"](att_norm)[:, :, :3]
            orig_norm = np.array(enhanced_pil.resize((224, 224))) / 255.0
            overlay = 0.55 * orig_norm + 0.45 * color_cam
            overlay = (np.clip(overlay, 0.0, 1.0) * 255).astype(np.uint8)
            checkpoint_used = self.gap3_checkpoint
        else:
            with torch.no_grad():
                rank_logits = self.model(img_tensor)
                pred_grade = coral_predict(rank_logits).item()
                coral_probs = coral_probabilities(rank_logits).squeeze().cpu().numpy().tolist()
                
            local_contrast = np.array([float(rad_raw[i, 10]) for i in range(9)])
            reg_weights = (local_contrast / (local_contrast.sum() + 1e-8)).tolist()
            
            cam_map = self.compute_gradcam(img_tensor, target_class=pred_grade)
            color_cam = plt.colormaps["jet"](cam_map)[:, :, :3]
            orig_norm = np.array(enhanced_pil.resize((224, 224))) / 255.0
            overlay = 0.55 * orig_norm + 0.45 * color_cam
            overlay = (np.clip(overlay, 0.0, 1.0) * 255).astype(np.uint8)
            checkpoint_used = "outputs/Module_03_Baseline_Swin_Tiny/clean_run/best_model_clean.pth"
            
        overlay_pil = Image.fromarray(overlay)
        buf = io.BytesIO()
        overlay_pil.save(buf, format="PNG")
        overlay_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
        
        latency = round((time.time() - t0) * 1000.0, 1)
        
        macula_features = {
            "first_order_mean": round(float(rad_raw[4, 0]), 2),
            "first_order_variance": round(float(rad_raw[4, 1]), 2),
            "first_order_entropy": round(float(rad_raw[4, 8]), 2),
            "glcm_contrast": round(float(rad_raw[4, 10]), 2),
            "glcm_dissimilarity": round(float(rad_raw[4, 11]), 2),
            "glcm_homogeneity": round(float(rad_raw[4, 12]), 2),
            "canny_edge_density": round(float(rad_raw[4, 28]), 4),
            "michelson_contrast": round(float(rad_raw[4, 31]), 4)
        }
        
        guidance = REFERRAL_GUIDELINES.get(pred_grade)
        
        return {
            "model_type": model_type,
            "checkpoint": checkpoint_used,
            "predicted_grade": pred_grade,
            "predicted_label": LABEL_MAPPING.get(pred_grade, "Unknown"),
            "coral_probabilities": [round(p, 4) for p in coral_probs],
            "auxiliary_probabilities": [round(p, 4) for p in coral_probs],
            "referral_guidance": guidance,
            "regional_attention": [round(w, 4) for w in reg_weights],
            "peak_attended_zone": ANATOMICAL_ZONES[int(np.argmax(reg_weights))],
            "gradcam_overlay_base64": overlay_b64,
            "macula_radiomics": macula_features,
            "latency_ms": latency
        }

class DiagnosticDashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=PUBLIC_DIR, **kwargs)

    def do_GET(self):
        if self.path == "/api/benchmark_samples":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            samples_list = list(BENCHMARK_SAMPLES.values())
            self.wfile.write(json.dumps(samples_list).encode("utf-8"))
            return
            
        elif self.path == "/api/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "healthy", "device": str(DEVICE)}).encode("utf-8"))
            return
            
        return super().do_GET()

    def do_POST(self):
        if self.path == "/api/predict":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            
            try:
                data = json.loads(body.decode("utf-8"))
            except Exception as e:
                self.send_error(400, f"Invalid JSON payload: {e}")
                return
                
            model_type = data.get("model_type", "baseline")
            if model_type not in ["baseline", "gap3"]:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "error": f"Invalid model_type '{model_type}'. Supported options are 'baseline' and 'gap3'."
                }).encode("utf-8"))
                return
                
            engine = ModelEngine.get_instance()
            
            pil_img = None
            if "sample_id" in data and data["sample_id"] in BENCHMARK_SAMPLES:
                sample_file = BENCHMARK_SAMPLES[data["sample_id"]]["file"]
                if os.path.exists(sample_file):
                    pil_img = Image.open(sample_file).convert("RGB")
            elif "image_base64" in data:
                b64_str = data["image_base64"]
                if "," in b64_str:
                    b64_str = b64_str.split(",")[1]
                img_bytes = base64.b64decode(b64_str)
                pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                
            if pil_img is None:
                self.send_error(400, "Could not load image from sample_id or image_base64")
                return
                
            result = engine.predict(pil_img, model_type=model_type)
            
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))
            return
            
        self.send_error(404, "Endpoint not found")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

def start_server(port=PORT):
    server_address = ("", port)
    httpd = HTTPServer(server_address, DiagnosticDashboardHandler)
    print(f"\n==================================================================")
    print(f"  RETINOSCAN AI - CLINICAL DIAGNOSTIC DASHBOARD")
    print(f"  Serving at: http://localhost:{port}")
    print(f"  Device:     {DEVICE}")
    print(f"==================================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        httpd.server_close()

if __name__ == "__main__":
    start_server(PORT)
