import os
from typing import Tuple, Optional
import torch
import torch.nn as nn
from src.models.swin import SwinTinyBaseline

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

class VisualRadiomicFusionModel(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        swin_checkpoint: Optional[str] = "outputs/Module_03_Baseline_Swin_Tiny/best_model.pth",
        d_rad: int = 32,
        d_model: int = 768,
        num_heads: int = 8,
        dropout: float = 0.2
    ):
        super().__init__()
        self.num_classes = num_classes
        self.d_model = d_model
        
        self.backbone = SwinTinyBaseline(num_classes=num_classes, pretrained=True)
        if swin_checkpoint and os.path.exists(swin_checkpoint):
            try:
                print(f"[FusionModel] Loading best Swin-Tiny weights: {swin_checkpoint}")
                state = torch.load(swin_checkpoint, map_location="cpu")
                if isinstance(state, dict) and "model_state_dict" in state:
                    state = state["model_state_dict"]
                self.backbone.load_state_dict(state)
                print("[FusionModel] Successfully loaded fine-tuned Swin-Tiny weights.")
            except Exception as e:
                print(f"[FusionModel] Warning: Could not load checkpoint ({e}), using default pretrained.")

        self.cross_attention = SpatialCrossAttention(
            d_vis=768,
            d_rad=d_rad,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def extract_visual_tokens(self, images: torch.Tensor) -> torch.Tensor:
        feat_map = self.backbone.forward_features(images)
        B, C, H, W = feat_map.shape
        vis_tokens = feat_map.flatten(2).permute(0, 2, 1)
        return vis_tokens

    def forward(
        self,
        images: torch.Tensor,
        radiomics: torch.Tensor
    ) -> torch.Tensor:
        vis_tokens = self.extract_visual_tokens(images)
        
        fused_tokens, _ = self.cross_attention(vis_tokens, radiomics)
        
        pooled = fused_tokens.mean(dim=1)
        
        logits = self.classifier(pooled)
        return logits

    def forward_with_attention(
        self,
        images: torch.Tensor,
        radiomics: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        vis_tokens = self.extract_visual_tokens(images)
        fused_tokens, attn_weights = self.cross_attention(vis_tokens, radiomics)
        pooled = fused_tokens.mean(dim=1)
        logits = self.classifier(pooled)
        return logits, attn_weights
