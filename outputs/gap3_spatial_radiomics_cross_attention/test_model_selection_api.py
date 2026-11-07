import urllib.request
import urllib.error
import json
import os
import sys

BASE_URL = "http://localhost:8080/api/predict"

def post_request(payload):
    req = urllib.request.Request(
        BASE_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req) as resp:
            status_code = resp.getcode()
            body = json.loads(resp.read().decode("utf-8"))
            return status_code, body
    except urllib.error.HTTPError as e:
        status_code = e.code
        body = json.loads(e.read().decode("utf-8")) if e.headers.get("Content-Type") == "application/json" else e.read().decode("utf-8")
        return status_code, body

def run_tests():
    print("=" * 75)
    print("  RUNNING MODEL SELECTION API TESTS ON LIVE SERVER (PORT 8080)")
    print("=" * 75)
    
    print("\n[Test 1] Testing Default Selection (no model_type supplied)...")
    payload_default = {"sample_id": "grade_0"}
    status, res_default = post_request(payload_default)
    print(f"  HTTP Status: {status}")
    print(f"  Model Type:  {res_default.get('model_type')}")
    print(f"  Checkpoint:  {res_default.get('checkpoint')}")
    print(f"  Predicted:   Grade {res_default.get('predicted_grade')} ({res_default.get('predicted_label')})")
    assert status == 200, f"Expected 200, got {status}"
    assert res_default.get("model_type") == "baseline", "Default must be 'baseline'!"
    assert "best_model_clean.pth" in res_default.get("checkpoint"), "Must load clean baseline checkpoint!"
    print("  --> PASSED: No parameter successfully selects baseline as default.")

    print("\n[Test 2] Testing Explicit model_type='baseline'...")
    payload_base = {"sample_id": "grade_2", "model_type": "baseline"}
    status, res_base = post_request(payload_base)
    print(f"  HTTP Status: {status}")
    print(f"  Model Type:  {res_base.get('model_type')}")
    print(f"  Checkpoint:  {res_base.get('checkpoint')}")
    print(f"  Predicted:   Grade {res_base.get('predicted_grade')} ({res_base.get('predicted_label')})")
    assert status == 200, f"Expected 200, got {status}"
    assert res_base.get("model_type") == "baseline", "Must return model_type='baseline'!"
    assert "best_model_clean.pth" in res_base.get("checkpoint")
    print("  --> PASSED: Explicit 'baseline' selects baseline.")

    print("\n[Test 3] Testing Explicit model_type='gap3'...")
    payload_gap3 = {"sample_id": "grade_2", "model_type": "gap3"}
    status, res_gap3 = post_request(payload_gap3)
    print(f"  HTTP Status: {status}")
    print(f"  Model Type:  {res_gap3.get('model_type')}")
    print(f"  Checkpoint:  {res_gap3.get('checkpoint')}")
    print(f"  Predicted:   Grade {res_gap3.get('predicted_grade')} ({res_gap3.get('predicted_label')})")
    print(f"  Latency:     {res_gap3.get('latency_ms')} ms")
    assert status == 200, f"Expected 200, got {status}"
    assert res_gap3.get("model_type") == "gap3", "Must return model_type='gap3'!"
    assert "best_spatial_radiomics_coral_model.pth" in res_gap3.get("checkpoint")
    print("  --> PASSED: Explicit 'gap3' selects Gap 3 Spatial Radiomics + Cross-Attention model.")

    print("\n[Test 4] Testing Invalid model_type ('unsupported_net')...")
    payload_inv = {"sample_id": "grade_0", "model_type": "unsupported_net"}
    status, res_inv = post_request(payload_inv)
    print(f"  HTTP Status: {status}")
    print(f"  Error Body:  {res_inv}")
    assert status == 400, f"Expected 400 for invalid model, got {status}"
    assert "error" in res_inv and "Invalid model_type" in res_inv["error"]
    print("  --> PASSED: Invalid model_type returns clear HTTP 400 error.")

    print("\n[Test 5] Running Multi-Grade Inference Verification Test on Both Models...")
    samples = ["grade_0", "grade_1", "grade_2", "grade_3", "grade_4"]
    for sid in samples:
        _, b_out = post_request({"sample_id": sid, "model_type": "baseline"})
        _, g_out = post_request({"sample_id": sid, "model_type": "gap3"})
        
        b_sum = sum(b_out["coral_probabilities"])
        g_sum = sum(g_out["coral_probabilities"])
        assert abs(b_sum - 1.0) < 0.05, f"Baseline probabilities sum to {b_sum} != 1.0"
        assert abs(g_sum - 1.0) < 0.05, f"Gap3 probabilities sum to {g_sum} != 1.0"
        
        print(f"  Sample: {sid.upper():<7} | Baseline Pred: Grade {b_out['predicted_grade']} ({b_out['latency_ms']}ms) | "
              f"Gap3 Pred: Grade {g_out['predicted_grade']} ({g_out['latency_ms']}ms)")

    print("\n" + "=" * 75)
    print("  ALL 5 API MODEL SELECTION TESTS PASSED SUCCESSFULLY!")
    print("=" * 75)

if __name__ == "__main__":
    run_tests()
