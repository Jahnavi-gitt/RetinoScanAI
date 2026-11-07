import os
import sys
import time
import json
import threading
import urllib.request
import urllib.error
from PIL import Image

sys.path.append(os.path.abspath("."))

from outputs.Module_14_Interactive_Web_Application.server import (
    ModelEngine,
    HTTPServer,
    DiagnosticDashboardHandler,
    BENCHMARK_SAMPLES
)

TEST_PORT = 8085

def run_server_validation_test():
    print("==================================================================")
    print("  MODULE 14: INTERACTIVE WEB APPLICATION VALIDATION TEST")
    print("==================================================================")
    
    print("\n--- [Step 1/3] Verifying Static Frontend Deliverables ---")
    public_dir = "outputs/Module_14_Interactive_Web_Application/public"
    required_files = ["index.html", "style.css", "app.js"]
    for rf in required_files:
        path = os.path.join(public_dir, rf)
        assert os.path.exists(path), f"Missing required frontend file: {path}"
        size_kb = os.path.getsize(path) / 1024
        print(f"  {rf}: OK ({size_kb:.1f} KB)")
        
    for k, v in BENCHMARK_SAMPLES.items():
        assert os.path.exists(v["file"]), f"Missing sample file: {v['file']}"
        print(f"  {k} image: OK ({v['name']})")
        
    print("\n--- [Step 2/3] Verifying ModelEngine Prediction Pipeline ---")
    engine = ModelEngine.get_instance()
    sample_img = Image.open(BENCHMARK_SAMPLES["grade_2"]["file"]).convert("RGB")
    res = engine.predict(sample_img)
    
    print(f"  Predicted Grade: {res['predicted_grade']} ({res['predicted_label']})")
    print(f"  CORAL Probs:     {res['coral_probabilities']}")
    print(f"  Peak Zone:       {res['peak_attended_zone']}")
    print(f"  Latency:         {res['latency_ms']} ms")
    print(f"  Grad-CAM B64:    Length = {len(res['gradcam_overlay_base64'])} chars")
    
    assert res["predicted_grade"] in range(5)
    assert len(res["coral_probabilities"]) == 5
    assert len(res["regional_attention"]) == 9
    assert res["gradcam_overlay_base64"].startswith("data:image/png;base64,")
    print("  [SUCCESS] ModelEngine direct prediction verified.")
    
    print(f"\n--- [Step 3/3] Testing HTTP Server on Port {TEST_PORT} ---")
    server_address = ("", TEST_PORT)
    httpd = HTTPServer(server_address, DiagnosticDashboardHandler)
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()
    time.sleep(0.5)
    
    base_url = f"http://localhost:{TEST_PORT}"
    
    req_health = urllib.request.Request(f"{base_url}/api/health")
    with urllib.request.urlopen(req_health) as resp:
        assert resp.status == 200
        health_data = json.loads(resp.read().decode())
        print(f"  GET /api/health: OK (Status: {health_data['status']}, Device: {health_data['device']})")
        
    req_samples = urllib.request.Request(f"{base_url}/api/benchmark_samples")
    with urllib.request.urlopen(req_samples) as resp:
        assert resp.status == 200
        samples_data = json.loads(resp.read().decode())
        print(f"  GET /api/benchmark_samples: OK ({len(samples_data)} samples returned)")
        assert len(samples_data) == 5
        
    pred_payload = json.dumps({"sample_id": "grade_4"}).encode("utf-8")
    req_pred = urllib.request.Request(
        f"{base_url}/api/predict",
        data=pred_payload,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req_pred) as resp:
        assert resp.status == 200
        pred_data = json.loads(resp.read().decode())
        print(f"  POST /api/predict (Grade 4): OK -> Predicted: {pred_data['predicted_label']} (Latency: {pred_data['latency_ms']} ms)")
        assert "referral_guidance" in pred_data
        assert pred_data["referral_guidance"]["urgency"] == "Critical / Emergent Referral"
        
    httpd.shutdown()
    httpd.server_close()
    print("\n==================================================================")
    print("  [ALL TESTS PASSED] Module 14 Web Application Verified.")
    print("==================================================================")

if __name__ == "__main__":
    run_server_validation_test()
