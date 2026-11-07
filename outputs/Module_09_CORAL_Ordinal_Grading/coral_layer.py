import os
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.fusion import VisualRadiomicFusionModel, SpatialCrossAttention

class CoralLayer(nn.Module):
    def __init__(self, in_features: int, num_classes: int = 5):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.num_tasks = num_classes - 1
        
        self.weight = nn.Parameter(torch.empty(in_features, 1))
        self.bias = nn.Parameter(torch.empty(self.num_tasks))
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight, mode='fan_in', nonlinearity='linear')
        with torch.no_grad():
            self.bias.copy_(torch.linspace(2.0, -2.0, self.num_tasks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        linear = torch.matmul(x, self.weight)
        return linear + self.bias

def coral_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 5,
    reduction: str = "mean"
) -> torch.Tensor:
    num_tasks = num_classes - 1
    device = targets.device
    
    ranks = torch.arange(num_tasks, device=device).unsqueeze(0)
    binary_targets = (targets.unsqueeze(1) > ranks).float()
    
    loss = F.binary_cross_entropy_with_logits(logits, binary_targets, reduction="none")
    task_loss = loss.sum(dim=-1)
    
    if reduction == "mean":
        return task_loss.mean()
    elif reduction == "sum":
        return task_loss.sum()
    else:
        return task_loss

def coral_predict(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    return (probs > threshold).sum(dim=-1)

def coral_probabilities(logits: torch.Tensor) -> torch.Tensor:
    sig = torch.sigmoid(logits)
    B = sig.size(0)
    
    p0 = 1.0 - sig[:, 0:1]
    p1 = sig[:, 0:1] - sig[:, 1:2]
    p2 = sig[:, 1:2] - sig[:, 2:3]
    p3 = sig[:, 2:3] - sig[:, 3:4]
    p4 = sig[:, 3:4]
    
    probs = torch.cat([p0, p1, p2, p3, p4], dim=-1)
    probs = torch.clamp(probs, min=0.0)
    probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)
    return probs

class CoralFusionModel(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        fusion_checkpoint: Optional[str] = "outputs/Module_07_Visual_Radiomic_Cross_Attention_Fusion/best_model.pth",
        d_rad: int = 32,
        d_model: int = 768,
        num_heads: int = 8,
        dropout: float = 0.2
    ):
        super().__init__()
        self.num_classes = num_classes
        self.d_model = d_model
        
        base = VisualRadiomicFusionModel(
            num_classes=num_classes,
            swin_checkpoint="outputs/Module_03_Baseline_Swin_Tiny/best_model.pth",
            d_rad=d_rad,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )
        if fusion_checkpoint and os.path.exists(fusion_checkpoint):
            try:
                print(f"[CORAL] Loading pre-trained fusion weights: {fusion_checkpoint}")
                state = torch.load(fusion_checkpoint, map_location="cpu")
                if "model_state_dict" in state:
                    state = state["model_state_dict"]
                base.load_state_dict(state, strict=False)
                print("[CORAL] Successfully loaded fusion feature extractor weights.")
            except Exception as e:
                print(f"[CORAL] Warning: Could not load fusion checkpoint ({e}).")
                
        self.backbone = base.backbone
        self.cross_attention = base.cross_attention
        
        self.dropout = nn.Dropout(dropout)
        self.coral_head = CoralLayer(in_features=d_model, num_classes=num_classes)

    def forward(self, images: torch.Tensor, radiomics: torch.Tensor) -> torch.Tensor:
        feat_map = self.backbone.forward_features(images)
        vis_tokens = feat_map.flatten(2).permute(0, 2, 1)
        
        fused_tokens, _ = self.cross_attention(vis_tokens, radiomics)
        
        pooled = fused_tokens.mean(dim=1)
        pooled = self.dropout(pooled)
        
        rank_logits = self.coral_head(pooled)
        return rank_logits

    def forward_with_attention(
        self,
        images: torch.Tensor,
        radiomics: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat_map = self.backbone.forward_features(images)
        vis_tokens = feat_map.flatten(2).permute(0, 2, 1)
        fused_tokens, attn_weights = self.cross_attention(vis_tokens, radiomics)
        pooled = fused_tokens.mean(dim=1)
        pooled = self.dropout(pooled)
        rank_logits = self.coral_head(pooled)
        probs = coral_probabilities(rank_logits)
        return rank_logits, probs, attn_weights
