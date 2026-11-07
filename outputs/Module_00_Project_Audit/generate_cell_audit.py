import json
import csv
import os

with open("dr-ansh (11).ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb.get("cells", [])

def classify_cell(idx, ctype, src, ec, outs):
    has_err = any(o.get("output_type") == "error" for o in outs)
    err_msg = ""
    if has_err:
        for o in outs:
            if o.get("output_type") == "error":
                ename = o.get("ename", "")
                evalue = o.get("evalue", "")
                err_msg = f"{ename}: {evalue}"[:80]
                break

    if idx <= 15:
        group = "Kaggle Setup & Dataset Upload"
        mapped_module = "Module 00 / External"
    elif idx <= 44:
        group = "EyePACS Exploration & Pretraining (Legacy)"
        mapped_module = "Module 01 (Pretraining / Legacy)"
    elif idx <= 93:
        group = "Attention Maps / ResNet50 CAMs (Legacy)"
        mapped_module = "Module 13 (Legacy CAM)"
    elif idx <= 133:
        group = "Radiomics, Vessel Extraction & CLAHE"
        mapped_module = "Module 01 / Module 06 (Radiomics)"
    elif idx <= 163:
        group = "Feature Fusion MLP & SMOTE Experiments"
        mapped_module = "Module 08 (SMOTE) / Module 10 (MLP Fusion)"
    elif idx <= 205:
        group = "Multi-Backbone Exploration & Embedding Extraction"
        mapped_module = "Module 02/03/04 (Embeddings)"
    elif idx <= 219:
        group = "Dataset Realignment Notice (EyePACS -> DDR)"
        mapped_module = "Module 01 (Dataset Validation)"
    elif idx <= 229:
        group = "Baseline: EfficientNet-B3 Fine-Tuning on DDR"
        mapped_module = "Module 04 (EfficientNet-B3)"
    elif idx <= 235:
        group = "Baseline: ConvNeXt-Small Fine-Tuning on DDR"
        mapped_module = "Module 02 (ConvNeXt-Small)"
    elif idx <= 243:
        group = "Baseline: Swin-Tiny Fine-Tuning on DDR"
        mapped_module = "Module 03 (Swin-Tiny)"
    elif idx <= 259:
        group = "Multi-Backbone + Global Radiomics + CORAL"
        mapped_module = "Module 09 (CORAL) / Module 10 (Fusion)"
    elif idx <= 310:
        group = "Grad-CAM++ & Lesion Patch Extraction"
        mapped_module = "Module 06 (Local Patches) / Module 13 (GradCAM)"
    elif idx <= 336:
        group = "Fusion Ablations & Imbalance Weighted Loss"
        mapped_module = "Module 08 (Class Imbalance) / Module 11 (Ablations)"
    elif idx <= 358:
        group = "Patch-Level Radiomics (pyradiomics)"
        mapped_module = "Module 06 (Local Radiomics)"
    elif idx <= 390:
        group = "Lesion Graph Construction & GATv2 Training"
        mapped_module = "Module 07 (Alternative Graph Fusion)"
    elif idx <= 418:
        group = "Exploratory Data Analysis, UMAP & Embeddings"
        mapped_module = "Module 01 / Module 12 (EDA)"
    elif idx <= 455:
        group = "Backbone Comparisons & CORAL Evaluation"
        mapped_module = "Module 05 (Selection) / Module 09 (CORAL)"
    else:
        group = "Clinical Decision Rules, Error Analysis & Visuals"
        mapped_module = "Module 12 (Evaluation) / Module 14 (Demo Logic)"

    if has_err:
        status = f"Failed ({err_msg})"
    elif ec is not None:
        status = f"Executed (Run #{ec})"
    elif ctype == "markdown":
        status = "Markdown Documentation"
    else:
        status = "Unexecuted Code"

    return group, mapped_module, status, has_err, err_msg

rows = []
for i, c in enumerate(cells):
    ctype = c.get("cell_type")
    src = "".join(c.get("source", []))
    lines = [l.strip() for l in src.split("\n") if l.strip()]
    first_line = lines[0] if lines else "<empty>"
    ec = c.get("execution_count")
    outs = c.get("outputs", [])
    
    group, mapped_mod, status, has_err, err_msg = classify_cell(i, ctype, src, ec, outs)
    rows.append({
        "cell_idx": i,
        "type": ctype,
        "exec_count": ec if ec is not None else "",
        "group": group,
        "mapped_module": mapped_mod,
        "status": status,
        "first_line": first_line[:100],
        "line_count": len(lines)
    })

os.makedirs("outputs/Module_00_Project_Audit", exist_ok=True)
csv_path = "outputs/Module_00_Project_Audit/notebook_cells_classified.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["cell_idx", "type", "exec_count", "group", "mapped_module", "status", "first_line", "line_count"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Successfully classified {len(rows)} cells into {csv_path}")

md_lines = [
    "# Notebook Cell-by-Cell Audit: `dr-ansh (11).ipynb`\n",
    "## Executive Summary\n",
    f"- **Total Cells**: {len(cells)} (Markdown: {sum(1 for c in cells if c.get('cell_type')=='markdown')}, Code: {sum(1 for c in cells if c.get('cell_type')=='code')})\n",
    f"- **Executed Code Cells**: {sum(1 for c in cells if c.get('execution_count') is not None)}\n",
    f"- **Unexecuted Code Cells**: {sum(1 for c in cells if c.get('cell_type')=='code' and c.get('execution_count') is None)}\n",
    "- **Execution Errors**: 3 (Cells 419, 422, 445 due to interactive session variable state)\n",
    "- **Origin Environment**: Kaggle Notebook (Linux, CUDA GPU, `/kaggle/input`, `/kaggle/working`)\n\n",
    "## Functional Section Breakdown\n\n",
    "| Cell Range | Primary Topic / Functional Group | Mapped Project Module | Status & Usability |\n",
    "|---|---|---|---|\n",
    "| 0–15 | Kaggle Environment Setup & Dataset Upload | Module 00 / External | Environment-specific scripts; not runnable locally without modification. |\n",
    "| 16–44 | EyePACS Pretraining & Data Loader (Legacy) | Module 01 (Pretraining) | Implements EyePACS loader, FocalLoss, cosine warmup scheduler. Superseded when dataset shifted to DDR. |\n",
    "| 45–93 | Attention Maps & ResNet-50 CAMs (Legacy) | Module 13 (Explainability) | Circular crop, CLAHE, mean CAM calculation. Useful reference for Grad-CAM. |\n",
    "| 94–133 | Radiomics, Vessel Extraction, CLAHE & SMOTE | Module 01 & Module 06 | Vessel segmentation, skeletonization, PyRadiomics (107 features), SMOTE. Reusable logic for radiomics. |\n",
    "| 134–163 | MLP Feature Fusion & 5-Fold Cross-Validation | Module 08 & Module 10 | Early tabular MLP fusion on static features; t-SNE evaluation. Incomplete end-to-end integration. |\n",
    "| 164–205 | Multi-Backbone Exploration & Embedding Extraction | Modules 02, 03, 04 | Extracts penultimate embeddings from backbones into `.npz` files. |\n",
    "| 206–219 | Dataset Realignment Notice (EyePACS to DDR) | Module 01 (Dataset Audit) | Explicitly documents the misalignment between EyePACS models and DDR radiomics. Transition to DDR. |\n",
    "| 220–229 | Baseline: EfficientNet-B3 Fine-Tuning on DDR | Module 04 | Complete fine-tuning pipeline: Config, DDR dataset, timm model init, QWK metric, train/val loops. |\n",
    "| 230–235 | Baseline: ConvNeXt-Small Fine-Tuning on DDR | Module 02 | Complete fine-tuning pipeline for ConvNeXt-Small on DDR with QWK optimization. |\n",
    "| 236–243 | Baseline: Swin-Tiny Fine-Tuning on DDR | Module 03 | Complete fine-tuning pipeline for Swin-Tiny on DDR with QWK optimization. |\n",
    "| 244–259 | Multi-Backbone Embeddings + Radiomics + CORAL | Module 09 & Module 10 | FusionDataset concatenating 3 deep embeddings (1536+768+768) + 107 radiomics, passed to CORAL head. |\n",
    "| 260–310 | Full Ensemble Grad-CAM++ & Lesion Patch Extraction | Module 06 & Module 13 | Extracts bounding boxes/patches around high-attention retinal regions using Grad-CAM++. |\n",
    "| 311–336 | Fusion Ablations & Imbalance Weighted Loss | Module 08 & Module 11 | Explores class weighting and removes mean centering from embeddings. |\n",
    "| 337–358 | Patch-Level Radiomics | Module 06 (Local Radiomics) | Computes radiomic descriptors specifically on extracted lesion patches. Highly relevant to spatial radiomics. |\n",
    "| 359–390 | Lesion Graph Construction & GATv2 GNN | Module 07 (Alternative) | Builds lesion graphs with torch_geometric; trains GATv2 model. Complex and fragile; cross-attention is cleaner. |\n",
    "| 391–418 | Exploratory Data Analysis & UMAP Embeddings | Module 01 & Module 12 | 3D UMAP projections and feature distributions across DR grades. |\n",
    "| 419–455 | Backbone Comparison, Logistic Regression, CORAL | Module 05 & Module 09 | Evaluates backbone representations via Logistic Regression and CORAL; contains runtime errors in interactive state. |\n",
    "| 456–473 | Clinical Decision Rules, Error Analysis & Plots | Module 12 & Module 14 | Decision rule: >0.85 Auto-accept, 0.65-0.85 Review, <0.65 Specialist; Plotly ordinal error gauge visualizer. |\n\n",
    "## Detailed Cell Table\n\n",
    "| Cell # | Type | Exec # | Functional Group | Mapped Module | Status | Line Count | First Line Preview |\n",
    "|---|---|---|---|---|---|---|---|\n"
]

for r in rows:
    first_line_clean = r["first_line"].replace("|", "\\|")
    md_lines.append(f"| {r['cell_idx']:03d} | {r['type']} | {r['exec_count']} | {r['group']} | {r['mapped_module']} | {r['status']} | {r['line_count']} | `{first_line_clean}` |\n")

with open("outputs/Module_00_Project_Audit/NOTEBOOK_CELL_AUDIT.md", "w", encoding="utf-8") as f:
    f.writelines(md_lines)

print("Saved outputs/Module_00_Project_Audit/NOTEBOOK_CELL_AUDIT.md")
