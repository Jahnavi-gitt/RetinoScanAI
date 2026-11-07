import os
from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

from src.models.coral import CoralLayer, coral_predict, coral_probabilities, coral_loss

class SpatialCrossAttention(nn.Module):
    def __init__(
        self,
        d_vis: int = 768,
        d_rad: int = 32,
        d_model: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_vis = d_vis
        self.d_rad = d_rad
        self.d_model = d_model
        self.num_heads = num_heads
        
        self.proj_q = nn.Linear(d_vis, d_model)
        self.proj_k = nn.Linear(d_rad, d_model)
        self.proj_v = nn.Linear(d_rad, d_model)
        
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        vis_tokens: torch.Tensor,
        rad_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.proj_q(vis_tokens)
        k = self.proj_k(rad_tokens)
        v = self.proj_v(rad_tokens)
        
        attn_out, attn_weights = self.mha(q, k, v, need_weights=True)
        
        x = self.norm1(vis_tokens + self.dropout(attn_out))
        
        out = self.norm2(x + self.dropout(self.ffn(x)))
        
        return out, attn_weights

class SpatialRadiomicsSwinCoralModel(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        d_vis: int = 768,
        d_rad: int = 32,
        d_model: int = 768,
        num_heads: int = 8,
        dropout: float = 0.2,
        pretrained_backbone: bool = True,
        backbone_checkpoint: Optional[str] = None
    ):
        super().__init__()
        self.num_classes = num_classes
        self.d_model = d_model
        
        self.backbone = tv_models.swin_t(weights=None)
        
        if backbone_checkpoint and os.path.exists(backbone_checkpoint):
            print(f"[Model] Loading backbone weights from: {backbone_checkpoint}")
            ckpt = torch.load(backbone_checkpoint, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                ckpt = ckpt["model_state_dict"]
            bb_state = {}
            for k, v in ckpt.items():
                if k.startswith("backbone."):
                    bb_state[k[len("backbone."):]] = v
                elif not k.startswith("coral_head.") and not k.startswith("classifier."):
                    bb_state[k] = v
            msg = self.backbone.load_state_dict(bb_state, strict=False)
            print(f"[Model] Loaded backbone checkpoint: {msg}")
        elif pretrained_backbone:
            cache_dir = os.path.expanduser("~/.cache/torch/hub/checkpoints")
            ckpt_path = os.path.join(cache_dir, "swin_t-704ceda3.pth")
            if os.path.exists(ckpt_path):
                print(f"[Model] Loading ImageNet-1k Swin-Tiny weights: {ckpt_path}")
                st = torch.load(ckpt_path, map_location="cpu")
                self.backbone.load_state_dict(st, strict=False)
            else:
                print("[Model] Using default Swin-Tiny initialization.")
                
        self.backbone.head = nn.Identity()
        
        self.cross_attention = SpatialCrossAttention(
            d_vis=d_vis,
            d_rad=d_rad,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.dropout = nn.Dropout(dropout)
        self.coral_head = CoralLayer(in_features=d_model, num_classes=num_classes)

    def extract_visual_tokens(self, images: torch.Tensor) -> torch.Tensor:
        feat = self.backbone.features(images)
        norm_feat = self.backbone.norm(feat)
        vis_tokens = norm_feat.flatten(1, 2)
        return vis_tokens

    def forward(
        self,
        images: torch.Tensor,
        radiomics: torch.Tensor
    ) -> torch.Tensor:
        vis_tokens = self.extract_visual_tokens(images)
        
        fused_tokens, _ = self.cross_attention(vis_tokens, radiomics)
        
        pooled = fused_tokens.mean(dim=1)
        pooled = self.dropout(pooled)
        
        rank_logits = self.coral_head(pooled)
        return rank_logits

    def forward_with_attention(
        self,
        images: torch.Tensor,
        radiomics: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        vis_tokens = self.extract_visual_tokens(images)
        fused_tokens, attn_weights = self.cross_attention(vis_tokens, radiomics)
        pooled = fused_tokens.mean(dim=1)
        pooled = self.dropout(pooled)
        rank_logits = self.coral_head(pooled)
        return rank_logits, attn_weights

    def predict(
        self,
        images: torch.Tensor,
        radiomics: torch.Tensor,
        threshold: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        rank_logits, attn_weights = self.forward_with_attention(images, radiomics)
        predicted_grade = coral_predict(rank_logits, threshold=threshold)
        class_probabilities = coral_probabilities(rank_logits)
        rank_probabilities = torch.sigmoid(rank_logits)
        
        return {
            "predicted_grade": predicted_grade,
            "class_probabilities": class_probabilities,
            "rank_probabilities": rank_probabilities,
            "attention_weights": attn_weights,
            "rank_logits": rank_logits
        }
