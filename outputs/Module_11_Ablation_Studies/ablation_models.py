import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional, Union

from src.models.swin import SwinTinyBaseline
from src.models.fusion import SpatialCrossAttention, VisualRadiomicFusionModel
from src.models.coral import CoralLayer, coral_loss, coral_predict, coral_probabilities
from src.models.integrated_model import IntegratedRetinopathyModel

class AblationA0_VisionOnly(nn.Module):
    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super().__init__()
        self.model = SwinTinyBaseline(num_classes=num_classes, pretrained=pretrained)
        self.name = "A0_VisionOnly"

    def forward(self, images: torch.Tensor, radiomics: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.model(images)

class AblationA1_RadiomicsOnly(nn.Module):
    def __init__(self, in_features: int = 288, hidden_dim: int = 128, num_classes: int = 5, dropout: float = 0.2):
        super().__init__()
        self.name = "A1_RadiomicsOnly"
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, images: Optional[torch.Tensor], radiomics: torch.Tensor) -> torch.Tensor:
        if radiomics.dim() == 3:
            radiomics = radiomics.reshape(radiomics.size(0), -1)
        return self.mlp(radiomics)

class AblationA2_NaiveConcatenation(nn.Module):
    def __init__(self, num_classes: int = 5, rad_features: int = 288, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        self.name = "A2_NaiveConcatenation"
        self.backbone = SwinTinyBaseline(num_classes=num_classes, pretrained=pretrained)
        total_in = 768 + rad_features
        self.fusion_head = nn.Sequential(
            nn.Linear(total_in, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, images: torch.Tensor, radiomics: torch.Tensor) -> torch.Tensor:
        feat_map = self.backbone.forward_features(images)
        vis_pooled = F.adaptive_avg_pool2d(feat_map, (1, 1)).flatten(1)
        
        if radiomics.dim() == 3:
            rad_flat = radiomics.reshape(radiomics.size(0), -1)
        else:
            rad_flat = radiomics
            
        fused = torch.cat([vis_pooled, rad_flat], dim=1)
        return self.fusion_head(fused)

class AblationA3_CrossAttentionNoRadiomics(nn.Module):
    def __init__(self, num_classes: int = 5, d_rad: int = 32, d_model: int = 768, num_heads: int = 8, dropout: float = 0.2, pretrained: bool = True):
        super().__init__()
        self.name = "A3_CrossAttentionNoRadiomics"
        self.backbone = SwinTinyBaseline(num_classes=num_classes, pretrained=pretrained)
        
        self.spatial_anchors = nn.Parameter(torch.randn(1, 9, d_rad) * 0.02)
        
        self.cross_attention = SpatialCrossAttention(
            d_vis=768,
            d_rad=d_rad,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, images: torch.Tensor, radiomics: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = images.size(0)
        feat_map = self.backbone.forward_features(images)
        vis_tokens = feat_map.flatten(2).permute(0, 2, 1)
        
        dummy_rad = self.spatial_anchors.repeat(B, 1, 1).contiguous()
        
        fused_tokens, _ = self.cross_attention(vis_tokens, dummy_rad)
        
        pooled = fused_tokens.mean(dim=1)
        return self.classifier(pooled)

class AblationA4_MultimodalNoCORAL(nn.Module):
    def __init__(self, num_classes: int = 5, d_rad: int = 32, d_model: int = 768, num_heads: int = 8, dropout: float = 0.2):
        super().__init__()
        self.name = "A4_MultimodalNoCORAL"
        self.model = VisualRadiomicFusionModel(
            num_classes=num_classes,
            d_rad=d_rad,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )

    def forward(self, images: torch.Tensor, radiomics: torch.Tensor) -> torch.Tensor:
        return self.model(images, radiomics)

class AblationA5_FullProposed(nn.Module):
    def __init__(self, num_classes: int = 5, d_rad: int = 32, d_model: int = 768, num_heads: int = 8, dropout: float = 0.2):
        super().__init__()
        self.name = "A5_FullProposed"
        self.model = IntegratedRetinopathyModel(
            num_classes=num_classes,
            d_rad=d_rad,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            pretrained_checkpoint="outputs/Module_10_Integrated_Proposed_Model/best_model.pth"
        )

    def forward(self, images: torch.Tensor, radiomics: torch.Tensor, return_aux: bool = False):
        return self.model(images, radiomics, return_aux=return_aux)

    def compute_hybrid_loss(self, rank_logits, aux_logits, targets, lambda_aux=0.3):
        return self.model.compute_hybrid_loss(rank_logits, aux_logits, targets, lambda_aux=lambda_aux)
