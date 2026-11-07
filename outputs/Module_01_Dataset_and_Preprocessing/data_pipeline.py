import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

LABEL_MAPPING = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    
    3: "Severe NPDR",
    4: "Proliferative DR"
}

CLASS_NAMES = [
    "No DR",
    "Mild NPDR",
    "Moderate NPDR",
    "Severe NPDR",
    "Proliferative DR"
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def retina_circle_crop(image: np.ndarray, thresh: int = 10, margin: int = 5) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    mask = gray > thresh
    if not np.any(mask):
        return image
    
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    
    h, w = image.shape[:2]
    y0 = max(0, y0 - margin)
    x0 = max(0, x0 - margin)
    y1 = min(h, y1 + margin)
    x1 = min(w, x1 + margin)
    
    return image[y0:y1, x0:x1]

def apply_clahe_lab(image_rgb: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    enhanced_rgb = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
    return enhanced_rgb

class PreprocessedRetinaTransform:
    def __init__(self, apply_clahe: bool = True, clip_limit: float = 2.0):
        self.apply_clahe = apply_clahe
        self.clip_limit = clip_limit

    def __call__(self, img_pil: Image.Image) -> Image.Image:
        img_np = np.array(img_pil)
        cropped_np = retina_circle_crop(img_np)
        if self.apply_clahe:
            enhanced_np = apply_clahe_lab(cropped_np, clip_limit=self.clip_limit)
        else:
            enhanced_np = cropped_np
        return Image.fromarray(enhanced_np)

def get_transforms(img_size: int = 224, is_training: bool = True, use_clahe: bool = True):
    base_preprocessing = [
        PreprocessedRetinaTransform(apply_clahe=use_clahe)
    ]
    
    if is_training:
        tf_list = base_preprocessing + [
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ]
    else:
        tf_list = base_preprocessing + [
            transforms.Resize((img_size, img_size)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ]
        
    return transforms.Compose(tf_list)

class DDRDataset(Dataset):
    def __init__(self, df: pd.DataFrame, images_root: str = None, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_root = images_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_path(self, row: pd.Series) -> str:
        if "filepath" in row and pd.notna(row["filepath"]) and os.path.exists(str(row["filepath"])):
            return str(row["filepath"])
        
        img_id = str(row["image_id"])
        if self.images_root:
            p1 = os.path.join(self.images_root, img_id)
            if os.path.exists(p1):
                return p1
            for ext in [".jpg", ".png", ".jpeg"]:
                p2 = p1 + ext
                if os.path.exists(p2):
                    return p2
        return p1

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self._resolve_path(row)
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
            
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise IOError(f"Failed to load image at {img_path}: {e}")
            
        if self.transform is not None:
            img_tensor = self.transform(img)
        else:
            to_tensor = transforms.ToTensor()
            img_tensor = to_tensor(img)
            
        label = int(row["numeric_label"])
        image_id = str(row["image_id"])
        
        return img_tensor, label, image_id

def create_stratified_splits(
    manifest_df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42
) -> tuple:
    df = manifest_df.copy().sort_values("image_id").reset_index(drop=True)
    
    assert "numeric_label" in df.columns, "Manifest must contain 'numeric_label'"
    assert "image_id" in df.columns, "Manifest must contain 'image_id'"
    
    total_test_val = test_size + val_size
    train_df, temp_df = train_test_split(
        df,
        test_size=total_test_val,
        stratify=df["numeric_label"],
        random_state=random_state
    )
    
    val_ratio_of_temp = val_size / total_test_val
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_ratio_of_temp),
        stratify=temp_df["numeric_label"],
        random_state=random_state
    )
    
    train_df = train_df.sort_values("image_id").reset_index(drop=True)
    val_df = val_df.sort_values("image_id").reset_index(drop=True)
    test_df = test_df.sort_values("image_id").reset_index(drop=True)
    
    return train_df, val_df, test_df

def audit_dataset(manifest_df: pd.DataFrame, images_root: str = None) -> tuple:
    audit_rows = []
    
    missing_images = 0
    corrupt_images = 0
    valid_images = 0
    channels_set = set()
    sizes_set = set()
    
    for idx, row in manifest_df.iterrows():
        img_id = str(row["image_id"])
        label = row.get("numeric_label", None)
        
        valid_label = (label is not None) and (label in [0, 1, 2, 3, 4])
        
        if "filepath" in row and pd.notna(row["filepath"]) and os.path.exists(str(row["filepath"])):
            path = str(row["filepath"])
        elif images_root:
            path = os.path.join(images_root, img_id)
            if not os.path.exists(path):
                for ext in [".jpg", ".png", ".jpeg"]:
                    if os.path.exists(path + ext):
                        path = path + ext
                        break
        else:
            path = img_id
            
        file_exists = os.path.exists(path)
        is_readable = False
        img_w, img_h, channels = 0, 0, 0
        status = "OK"
        
        if not file_exists:
            missing_images += 1
            status = "MISSING_FILE"
        else:
            try:
                with Image.open(path) as img:
                    img_w, img_h = img.size
                    channels = len(img.getbands())
                    channels_set.add(channels)
                    sizes_set.add((img_w, img_h))
                    is_readable = True
                    valid_images += 1
            except Exception as e:
                corrupt_images += 1
                status = f"CORRUPT_FILE: {e}"
                
        if not valid_label:
            status = f"INVALID_LABEL: {label}" if status == "OK" else status + f"_AND_INVALID_LABEL: {label}"
            
        audit_rows.append({
            "image_id": img_id,
            "filepath": path,
            "numeric_label": label,
            "label_name": LABEL_MAPPING.get(label, "Unknown"),
            "file_exists": file_exists,
            "is_readable": is_readable,
            "width": img_w,
            "height": img_h,
            "channels": channels,
            "audit_status": status
        })
        
    audit_df = pd.DataFrame(audit_rows)
    
    class_counts = manifest_df["numeric_label"].value_counts().sort_index().to_dict()
    class_dist_rows = []
    total_samples = len(manifest_df)
    for c in range(5):
        count = class_counts.get(c, 0)
        class_dist_rows.append({
            "grade": c,
            "grade_name": LABEL_MAPPING[c],
            "count": count,
            "percentage": round((count / total_samples) * 100, 2) if total_samples > 0 else 0.0
        })
    class_dist_df = pd.DataFrame(class_dist_rows)
    
    summary = {
        "total_records": total_samples,
        "valid_images": valid_images,
        "missing_images": missing_images,
        "corrupt_images": corrupt_images,
        "channels_observed": list(channels_set),
        "class_distribution": class_counts
    }
    
    return audit_df, class_dist_df, summary

def build_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame = None,
    images_root: str = None,
    img_size: int = 224,
    batch_size: int = 16,
    num_workers: int = 2
) -> dict:
    train_tf = get_transforms(img_size=img_size, is_training=True)
    val_tf = get_transforms(img_size=img_size, is_training=False)
    
    train_ds = DDRDataset(train_df, images_root=images_root, transform=train_tf)
    val_ds = DDRDataset(val_df, images_root=images_root, transform=val_tf)
    
    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    }
    
    if test_df is not None:
        test_ds = DDRDataset(test_df, images_root=images_root, transform=val_tf)
        loaders["test"] = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
    return loaders
