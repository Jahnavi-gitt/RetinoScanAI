import os
import pandas as pd
from sklearn.model_selection import train_test_split

LABEL_MAPPING = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR"
}

raw_csv_path = "data/ddr/DR_grading.csv"
assert os.path.exists(raw_csv_path), f"File not found: {raw_csv_path}"
df = pd.read_csv(raw_csv_path)

df = df.rename(columns={"id_code": "image_id", "diagnosis": "numeric_label"})
df["grade_name"] = df["numeric_label"].map(LABEL_MAPPING)

img_dir = os.path.abspath("data/ddr/DR_grading/DR_grading")
df["filepath"] = df["image_id"].apply(lambda fname: os.path.join(img_dir, fname))

df = df.sort_values("image_id").reset_index(drop=True)

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["numeric_label"],
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["numeric_label"],
    random_state=42
)

train_df = train_df.sort_values("image_id").reset_index(drop=True)
val_df = val_df.sort_values("image_id").reset_index(drop=True)
test_df = test_df.sort_values("image_id").reset_index(drop=True)

print(f"Master manifest: {len(df)} samples")
print(f"Train split:     {len(train_df)} samples (70.0%)")
print(f"Val split:       {len(val_df)} samples (15.0%)")
print(f"Test split:      {len(test_df)} samples (15.0%)")

os.makedirs("data_manifests", exist_ok=True)
df.to_csv("data_manifests/ddr_master_manifest.csv", index=False)
train_df.to_csv("data_manifests/train_manifest.csv", index=False)
val_df.to_csv("data_manifests/val_manifest.csv", index=False)
test_df.to_csv("data_manifests/test_manifest.csv", index=False)

out_dir = "outputs/Module_01_Dataset_and_Preprocessing"
os.makedirs(out_dir, exist_ok=True)
train_df.to_csv(os.path.join(out_dir, "split_manifest_train.csv"), index=False)
val_df.to_csv(os.path.join(out_dir, "split_manifest_val.csv"), index=False)
test_df.to_csv(os.path.join(out_dir, "split_manifest_test.csv"), index=False)

class_counts = df["numeric_label"].value_counts().sort_index()
train_counts = train_df["numeric_label"].value_counts().sort_index()
val_counts = val_df["numeric_label"].value_counts().sort_index()
test_counts = test_df["numeric_label"].value_counts().sort_index()

dist_rows = []
total_all = len(df)
for c in range(5):
    dist_rows.append({
        "grade": c,
        "grade_name": LABEL_MAPPING[c],
        "total_count": int(class_counts[c]),
        "total_percentage": round(class_counts[c] / total_all * 100, 2),
        "train_count": int(train_counts[c]),
        "val_count": int(val_counts[c]),
        "test_count": int(test_counts[c])
    })

dist_df = pd.DataFrame(dist_rows)
dist_df.to_csv(os.path.join(out_dir, "class_distribution.csv"), index=False)
print("\nSaved full class distribution:")
print(dist_df.to_string(index=False))

audit_rows = []
for idx, r in df.iterrows():
    audit_rows.append({
        "image_id": r["image_id"],
        "filepath": r["filepath"],
        "numeric_label": r["numeric_label"],
        "label_name": r["grade_name"],
        "file_exists": os.path.exists(r["filepath"]),
        "audit_status": "OK" if r["numeric_label"] in range(5) else "INVALID_LABEL"
    })
audit_df = pd.DataFrame(audit_rows)
audit_df.to_csv(os.path.join(out_dir, "dataset_audit.csv"), index=False)
print(f"\nSaved dataset audit records ({len(audit_df)} rows) -> {out_dir}/dataset_audit.csv")
