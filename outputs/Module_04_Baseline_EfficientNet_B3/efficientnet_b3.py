import os
import torch
import torch.nn as nn
import torchvision.models as tv_models
import timm

class EfficientNetB3Baseline(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        drop_rate: float = 0.3
    ):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = 1536
        
        cache_dir = os.path.expanduser("~/.cache/torch/hub/checkpoints")
        cached_files = [
            f for f in os.listdir(cache_dir)
            if f.startswith("efficientnet_b3") and f.endswith(".pth")
        ] if os.path.exists(cache_dir) else []
        
        loaded = False
        if pretrained and cached_files:
            try:
                ckpt_path = os.path.join(cache_dir, cached_files[0])
                print(f"[EfficientNet-B3] Loading cached torchvision weights: {ckpt_path}")
                base = tv_models.efficientnet_b3(weights=None)
                state = torch.load(ckpt_path, map_location="cpu")
                base.load_state_dict(state)
                in_feats = base.classifier[1].in_features
                base.classifier = nn.Sequential(
                    nn.Dropout(p=drop_rate, inplace=True),
                    nn.Linear(in_feats, num_classes)
                )
                self.model = base
                self.is_torchvision = True
                loaded = True
            except Exception as e:
                print(f"[EfficientNet-B3] Could not load cached weights ({e}), falling back to timm/tv_models.")

        if not loaded:
            try:
                base = tv_models.efficientnet_b3(weights="DEFAULT" if pretrained else None)
                in_feats = base.classifier[1].in_features
                base.classifier = nn.Sequential(
                    nn.Dropout(p=drop_rate, inplace=True),
                    nn.Linear(in_feats, num_classes)
                )
                self.model = base
                self.is_torchvision = True
                loaded = True
            except Exception:
                try:
                    self.model = timm.create_model(
                        "efficientnet_b3",
                        pretrained=pretrained,
                        num_classes=num_classes,
                        drop_rate=drop_rate
                    )
                    self.is_torchvision = False
                    loaded = True
                except Exception as e:
                    print(f"[EfficientNet-B3] Error creating model: {e}")
                    base = tv_models.efficientnet_b3(weights=None)
                    in_feats = base.classifier[1].in_features
                    base.classifier = nn.Sequential(
                        nn.Dropout(p=drop_rate, inplace=True),
                        nn.Linear(in_feats, num_classes)
                    )
                    self.model = base
                    self.is_torchvision = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_torchvision:
            return self.model.features(x)
        else:
            return self.model.forward_features(x)
