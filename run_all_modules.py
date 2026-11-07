import os
import sys
import time
import argparse
import subprocess
import pandas as pd

MODULE_SCRIPTS = {
    "00": {
        "name": "Project & Notebook Audit",
        "script": "outputs/Module_00_Project_Audit/generate_cell_audit.py"
    },
    "01": {
        "name": "Dataset Validation & Reproducible Preprocessing",
        "script": "outputs/Module_01_Dataset_and_Preprocessing/verify_preprocessing.py"
    },
    "02": {
        "name": "Baseline ConvNeXt-Small",
        "script": "outputs/Module_02_Baseline_ConvNeXt_Small/run_convnext_baseline.py"
    },
    "03": {
        "name": "Baseline Swin-Tiny",
        "script": "outputs/Module_03_Baseline_Swin_Tiny/run_swin_baseline.py"
    },
    "04": {
        "name": "Baseline EfficientNet-B3",
        "script": "outputs/Module_04_Baseline_EfficientNet_B3/run_efficientnet_baseline.py"
    },
    "05": {
        "name": "Baseline Comparison & Model Selection",
        "script": "outputs/Module_05_Baseline_Comparison_and_Model_Selection/generate_comparison.py"
    },
    "06": {
        "name": "Spatially Localized Radiomics (3x3 Grid)",
        "script": "outputs/Module_06_Spatially_Localized_Radiomics/extract_features.py"
    },
    "07": {
        "name": "Spatial Visual-Radiomic Cross-Attention Fusion",
        "script": "outputs/Module_07_Visual_Radiomic_Cross_Attention_Fusion/run_fusion_experiment.py"
    },
    "08": {
        "name": "Class Imbalance Strategies & Loss Benchmark",
        "script": "outputs/Module_08_Class_Imbalance_Strategies/run_imbalance_experiments.py"
    },
    "09": {
        "name": "CORAL Ordinal Classification Head",
        "script": "outputs/Module_09_CORAL_Ordinal_Grading/run_coral_experiments.py"
    },
    "10": {
        "name": "Integrated Proposed Architecture",
        "script": "outputs/Module_10_Integrated_Proposed_Model/run_integrated_model.py"
    },
    "11": {
        "name": "Systematic Ablation Studies (A0 to A5)",
        "script": "outputs/Module_11_Ablation_Studies/run_ablation_experiments.py"
    },
    "12": {
        "name": "Final Evaluation on Held-Out Test Split",
        "script": "outputs/Module_12_Final_Evaluation_on_Held_Out_Test_Split/run_test_evaluation.py"
    },
    "13": {
        "name": "Explainability & Interpretability",
        "script": "outputs/Module_13_Explainability_and_Interpretability/run_explainability.py"
    },
    "14": {
        "name": "Interactive Web Application Validation Test",
        "script": "outputs/Module_14_Interactive_Web_Application/test_server.py"
    }
}

def run_module(mod_id: str, py_exec: str) -> dict:
    info = MODULE_SCRIPTS[mod_id]
    script_path = info["script"]
    name = info["name"]
    
    print(f"\n==================================================================")
    print(f"  [RUNNING] Module {mod_id}: {name}")
    print(f"  Script: {script_path}")
    print(f"==================================================================")
    
    t0 = time.time()
    try:
        res = subprocess.run([py_exec, script_path], check=True, text=True, capture_output=True)
        elapsed = time.time() - t0
        print(f"  --> Completed successfully in {elapsed:.1f}s")
        return {"module": mod_id, "name": name, "status": "SUCCESS", "time_sec": round(elapsed, 1), "error": ""}
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - t0
        err_msg = e.stderr[-300:] if e.stderr else str(e)
        print(f"  --> FAILED in {elapsed:.1f}s: {err_msg}")
        return {"module": mod_id, "name": name, "status": "FAILED", "time_sec": round(elapsed, 1), "error": err_msg}

def main():
    parser = argparse.ArgumentParser(description="Master Pipeline Runner for Diabetic Retinopathy Project")
    parser.add_argument("--module", type=str, default="all", help="Module ID to run (e.g. 00, 06, 10, 12) or 'all'")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable path")
    args = parser.parse_args()
    
    py_exec = args.python
    print("==================================================================")
    print("  DIABETIC RETINOPATHY SPATIAL CROSS-ATTENTION MASTER RUNNER")
    print(f"  Python: {py_exec}")
    print("==================================================================")
    
    records = []
    if args.module.lower() == "all":
        for m_id in sorted(MODULE_SCRIPTS.keys()):
            r = run_module(m_id, py_exec)
            records.append(r)
    else:
        m_id = args.module.zfill(2)
        if m_id not in MODULE_SCRIPTS:
            print(f"Error: Module '{args.module}' not found. Available modules: {list(MODULE_SCRIPTS.keys())}")
            sys.exit(1)
        r = run_module(m_id, py_exec)
        records.append(r)
        
    df = pd.DataFrame(records)
    print("\n==================================================================")
    print("  PIPELINE EXECUTION SUMMARY")
    print("==================================================================")
    print(df[["module", "name", "status", "time_sec"]].to_string(index=False))
    print("==================================================================")

if __name__ == "__main__":
    main()
