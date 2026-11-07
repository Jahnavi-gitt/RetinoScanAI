import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

LABEL_MAPPING = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR"
}

def build_clean_splits():
    raw_csv = "data/ddr/DR_grading.csv"
    img_dir = os.path.abspath("data/ddr/DR_grading/DR_grading")
    assert os.path.exists(raw_csv), f"Missing {raw_csv}"
    assert os.path.exists(img_dir), f"Missing {img_dir}"

    df = pd.read_csv(raw_csv).rename(columns={"id_code": "image_id", "diagnosis": "numeric_label"})
    df["grade_name"] = df["numeric_label"].map(LABEL_MAPPING)
    df["filepath"] = df["image_id"].apply(lambda fname: os.path.join(img_dir, fname))

    missing = [fp for fp in df["filepath"] if not os.path.exists(fp)]
    assert len(missing) == 0, f"Found {len(missing)} missing files!"

    def assign_group(fname):
        if "-" in fname:
            parts = fname.split("-")
            return f"{parts[0]}-{parts[1]}"
        else:
            return f"TS_{fname[:14]}"

    df["group_id"] = df["image_id"].apply(assign_group)

    group_max_label = df.groupby("group_id")["numeric_label"].max()
    df["strat_label"] = df["group_id"].map(group_max_label)

    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
    splits = list(sgkf.split(df, df["strat_label"], groups=df["group_id"]))

    test_idx = splits[0][1]
    val_idx = splits[1][1]
    train_idx = np.concatenate([splits[i][1] for i in range(2, 10)])

    train_df = df.iloc[train_idx].sort_values("image_id").reset_index(drop=True)
    val_df = df.iloc[val_idx].sort_values("image_id").reset_index(drop=True)
    test_df = df.iloc[test_idx].sort_values("image_id").reset_index(drop=True)

    tr_groups = set(train_df["group_id"])
    va_groups = set(val_df["group_id"])
    te_groups = set(test_df["group_id"])

    assert len(tr_groups.intersection(va_groups)) == 0, "Train-Val group leakage!"
    assert len(tr_groups.intersection(te_groups)) == 0, "Train-Test group leakage!"
    assert len(va_groups.intersection(te_groups)) == 0, "Val-Test group leakage!"

    tr_images = set(train_df["image_id"])
    va_images = set(val_df["image_id"])
    te_images = set(test_df["image_id"])

    assert len(tr_images.intersection(va_images)) == 0, "Train-Val image leakage!"
    assert len(tr_images.intersection(te_images)) == 0, "Train-Test image leakage!"
    assert len(va_images.intersection(te_images)) == 0, "Val-Test image leakage!"

    out_dir = "data_manifests/split_80_10_10"
    os.makedirs(out_dir, exist_ok=True)

    cols = ["image_id", "numeric_label", "grade_name", "filepath", "group_id"]
    train_df[cols].to_csv(os.path.join(out_dir, "train_manifest.csv"), index=False)
    val_df[cols].to_csv(os.path.join(out_dir, "val_manifest.csv"), index=False)
    test_df[cols].to_csv(os.path.join(out_dir, "test_manifest.csv"), index=False)

    summary = {
        "total_images": len(df),
        "total_groups": df["group_id"].nunique(),
        "train": {
            "count": len(train_df),
            "percentage": round(len(train_df) / len(df) * 100, 2),
            "per_grade": train_df["numeric_label"].value_counts().sort_index().to_dict()
        },
        "val": {
            "count": len(val_df),
            "percentage": round(len(val_df) / len(df) * 100, 2),
            "per_grade": val_df["numeric_label"].value_counts().sort_index().to_dict()
        },
        "test": {
            "count": len(test_df),
            "percentage": round(len(test_df) / len(df) * 100, 2),
            "per_grade": test_df["numeric_label"].value_counts().sort_index().to_dict()
        },
        "overlap_checks": {
            "train_val_group_overlap": len(tr_groups.intersection(va_groups)),
            "train_test_group_overlap": len(tr_groups.intersection(te_groups)),
            "val_test_group_overlap": len(va_groups.intersection(te_groups)),
            "train_val_image_overlap": len(tr_images.intersection(va_images)),
            "train_test_image_overlap": len(tr_images.intersection(te_images)),
            "val_test_image_overlap": len(va_images.intersection(te_images)),
        }
    }

    with open(os.path.join(out_dir, "manifest_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("=== MANIFEST GENERATION COMPLETE ===")
    print(f"Saved clean splits to {out_dir}/")
    print(f"Train: {summary['train']['count']} ({summary['train']['percentage']}%)")
    print(f"Val:   {summary['val']['count']} ({summary['val']['percentage']}%)")
    print(f"Test:  {summary['test']['count']} ({summary['test']['percentage']}%)")
    print("Overlap checks passed: ALL 0 (ZERO).")

if __name__ == "__main__":
    build_clean_splits()
