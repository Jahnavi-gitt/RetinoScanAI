import os
import torch
import torch.nn as nn
import torchvision.models as tv_models
import timm

class SwinTinyBaseline(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        drop_rate: float = 0.2
    ):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = 768
        
        cache_dir = os.path.expanduser("~/.cache/torch/hub/checkpoints")
        cached_files = [f for f in os.listdir(cache_dir) if f.startswith("swin_t") and f.endswith(".pth")] if os.path.exists(cache_dir) else []
        
        loaded = False
        if pretrained and cached_files:
            try:
                ckpt_path = os.path.join(cache_dir, cached_files[0])
                print(f"[Swin-Tiny] Loading cached torchvision weights: {ckpt_path}")
                base = tv_models.swin_t(weights=None)
                state = torch.load(ckpt_path, map_location="cpu")
                base.load_state_dict(state)
                in_feats = base.head.in_features
                base.head = nn.Linear(in_feats, num_classes)
                self.model = base
                self.is_torchvision = True
                loaded = True
            except Exception as e:
                print(f"[Swin-Tiny] Could not load cached weights ({e}), falling back to timm/init.")
                
        if not loaded:
            try:
                self.model = timm.create_model(
                    "swin_tiny_patch4_window7_224",
                    pretrained=pretrained,
                    num_classes=num_classes,
                    drop_rate=drop_rate
                )
                self.is_torchvision = False
                loaded = True
            except Exception:
                base = tv_models.swin_t(weights=None)
                in_feats = base.head.in_features
                base.head = nn.Linear(in_feats, num_classes)
                self.model = base
                self.is_torchvision = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_torchvision:
            feat = self.model.features(x)
            if feat.dim() == 4 and feat.shape[3] == self.feature_dim:
                feat = feat.permute(0, 3, 1, 2)
            return feat
        else:
            feat = self.model.forward_features(x)
            if feat.dim() == 4 and feat.shape[3] == self.feature_dim:
                feat = feat.permute(0, 3, 1, 2)
            return feat
