# Module 14: Interactive Web Application (Clinical Diagnostic Dashboard)
**RetinoScan AI: Spatial Visual–Radiomic Diagnostic & Explainability Platform**
**Diabetic Retinopathy Multimodal Spatial Cross-Attention Project**

---

## 1. Executive Summary & Clinical Platform Overview

To translate our scientific advancements into clinical reality, we designed and implemented **RetinoScan AI**: a full-stack, glassmorphic medical web application that empowers clinicians, ophthalmologists, and researchers to interactively evaluate retinal fundus images with real-time explainability.

### Key Capabilities:
1. **Real-Time Multimodal Inference**:
   - Runs the end-to-end **Integrated Proposed Architecture** (Swin-Tiny + $3 \times 3$ Spatially Localized Radiomics + 8-Head Spatial Cross-Attention + CORAL Ordinal Head).
   - Extracts 32 IBSI-compliant radiomic descriptors across the 9 anatomical zones on-the-fly and standardizes them using our pre-fitted normalizer.
   - Computes CORAL monotonic ordinal rank probabilities, eliminating catastrophic grading omissions.
2. **Interactive Visual & Spatial Saliency Grounding**:
   - **Original Fundus View**: Pristine preprocessed fundus photography.
   - **$3 \times 3$ Anatomical Retinal Grid Overlay**: Interactive boundary overlays delineating the 9 retinal regions (Macula, Superior/Inferior Vascular Arcades, Nasal/Temporal Periphery).
   - **Dynamic Grad-CAM Overlay**: Continuous gradient saliency heatmap from the Swin-Tiny backbone with an interactive real-time opacity slider (0% to 100%).
3. **Spatial Cross-Attention Interpretability**:
   - Interactive $3 \times 3$ attention mini-grid displaying the percentage of attention allocated by the 49 visual tokens to each of the 9 anatomical radiomic zones.
   - Highlights the peak attended zone (e.g. Central Macula vs Vascular Arcades) in real-time.
4. **Clinical Actionability & Referral Guidance**:
   - Automated ICD-10 diagnostic coding (e.g. `E11.319`, `E11.359`).
   - Standardized ophthalmic referral timelines:
     - *Grade 0 (No DR)*: Routine follow-up in 12–24 months.
     - *Grade 1 (Mild NPDR)*: Routine screening in 6–12 months.
     - *Grade 2 (Moderate NPDR)*: Comprehensive ophthalmologist evaluation within 3–6 months.
     - *Grade 3 (Severe NPDR)*: Urgent retina specialist referral within 2–4 weeks.
     - *Grade 4 (Proliferative DR)*: Emergent vitreoretinal referral within 24–48 hours for immediate panretinal photocoagulation (PRP) or anti-VEGF injection.
5. **One-Click Benchmark Gallery & Custom Upload**:
   - Includes 5 pre-loaded benchmark cases covering all DR grades (0 to 4) for instant one-click demonstration.
   - Full drag-and-drop and file-browser upload for custom patient fundus photographs.

---

## 2. System Architecture & Technology Stack

```mermaid
graph TD
    Client["Client Browser (HTML5 + Vanilla CSS + JS)"] -->|HTTP GET / POST| Server["Server (server.py - Python HTTP / REST API)"]
    Server --> Engine["ModelEngine Singleton (Apple Silicon MPS Accelerated)"]
    Engine --> Radiomics["SpatialRadiomicsExtractor (9 zones x 32 IBSI features = 288-d)"]
    Engine --> Swin["Swin-Tiny Backbone (49 Visual Spatial Tokens, d=768)"]
    Engine --> CrossAttn["8-Head Spatial Cross-Attention (49 visual tokens query 9 radiomic zones)"]
    CrossAttn --> CORAL["CORAL Ordinal Head (4 binary rank thresholds)"]
    CORAL --> Response["JSON Response (Grade, Probs, Referral, Grad-CAM Base64, Attention Grid)"]
    Response --> Client
```

- **Backend**: Python 3.9 HTTP server (`server.py`) with zero extraneous dependencies.
- **Frontend**: Pure semantic HTML5, Vanilla CSS3 (glassmorphic dark mode, radial gradients, micro-animations), and modern vanilla JavaScript (`app.js`).
- **Hardware Acceleration**: Apple Silicon MPS (`torch.device("mps")`) delivering ~80–120ms single-image inference latency.

---

## 3. REST API Specification

### Endpoint 1: `GET /api/benchmark_samples`
Returns the list of 5 pre-loaded DDR benchmark cases across Grades 0 to 4.
- **Response Format**:
  ```json
  [
    {
      "id": "grade_0",
      "grade": 0,
      "name": "Grade 0: No DR (Healthy Retina)",
      "url": "/samples/grade_0_ddr_sample_grade0_003.jpg"
    }, ...
  ]
  ```

### Endpoint 2: `POST /api/predict`
Executes real-time inference on an uploaded image or benchmark sample.
- **Request Body**:
  ```json
  { "sample_id": "grade_2" }
  // OR
  { "image_base64": "data:image/jpeg;base64,..." }
  ```
- **Response Body**:
  ```json
  {
    "predicted_grade": 2,
    "predicted_label": "Moderate NPDR",
    "coral_probabilities": [0.05, 0.15, 0.70, 0.08, 0.02],
    "referral_guidance": {
      "urgency": "Moderate / Actionable Risk",
      "follow_up": "3 to 6 Months",
      "recommendation": "Moderate non-proliferative retinopathy with scattered microaneurysms...",
      "icd_code": "E11.329 (Moderate Nonproliferative Retinopathy)"
    },
    "regional_attention": [0.04, 0.05, 0.04, 0.05, 0.55, 0.06, 0.04, 0.05, 0.04],
    "peak_attended_zone": "R4: Central Macula / Fovea",
    "gradcam_overlay_base64": "data:image/png;base64,...",
    "macula_radiomics": {
      "first_order_mean": 124.5,
      "first_order_variance": 412.3,
      "glcm_contrast": 18.4,
      "canny_edge_density": 0.0412
    },
    "latency_ms": 83.7
  }
  ```

---

## 4. How to Run the Web Application

To launch the dashboard locally:
```bash
/Users/jahnavi/.venv/bin/python3 outputs/Module_14_Interactive_Web_Application/server.py
```
Open your browser and navigate to:
```
http://localhost:8080
```

---
*Report certified by Module 14 Interactive Web Application Pipeline.*
