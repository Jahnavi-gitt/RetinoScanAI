import os
from typing import Tuple, Dict, Any, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.swin import SwinTinyBaseline
from src.models.fusion import SpatialCrossAttention, VisualRadiomicFusionModel
from src.models.coral import CoralLayer, coral_loss, coral_predict, coral_probabilities

class IntegratedRetinopathyModel(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        d_rad: int = 32,
        d_model: int = 768,
        num_heads: int = 8,
        dropout: float = 0.2,
        pretrained_checkpoint: Optional[str] = "outputs/Module_07_Visual_Radiomic_Cross_Attention_Fusion/best_model.pth"
    ):
        super().__init__()
        self.num_classes = num_classes
        self.d_model = d_model
        self.num_tasks = num_classes - 1
        
        self.backbone = SwinTinyBaseline(num_classes=num_classes, pretrained=True)
        
        self.cross_attention = SpatialCrossAttention(
            d_vis=768,
            d_rad=d_rad,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )
        
        if pretrained_checkpoint and os.path.exists(pretrained_checkpoint):
            try:
                print(f"[IntegratedModel] Loading pre-trained fusion weights: {pretrained_checkpoint}")
                state = torch.load(pretrained_checkpoint, map_location="cpu")
                if "model_state_dict" in state:
                    state = state["model_state_dict"]
                model_state = self.state_dict()
                filtered_state = {k: v for k, v in state.items() if k in model_state and v.shape == model_state[k].shape}
                self.load_state_dict(filtered_state, strict=False)
                print(f"[IntegratedModel] Successfully loaded {len(filtered_state)} pre-trained parameter tensors.")
            except Exception as e:
                print(f"[IntegratedModel] Notice: Checkpoint loading ({e}), proceeding with default initializations.")
                
        self.dropout = nn.Dropout(dropout)
        
        self.coral_head = CoralLayer(in_features=d_model, num_classes=num_classes)
        
        self.aux_head = nn.Linear(d_model, num_classes)

    def extract_visual_tokens(self, images: torch.Tensor) -> torch.Tensor:
        feat_map = self.backbone.forward_features(images)
        return feat_map.flatten(2).permute(0, 2, 1)

    def forward(
        self,
        images: torch.Tensor,
        radiomics: torch.Tensor,
        return_aux: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        vis_tokens = self.extract_visual_tokens(images)
        fused_tokens, _ = self.cross_attention(vis_tokens, radiomics)
        
        pooled = fused_tokens.mean(dim=1)
        pooled = self.dropout(pooled)
        
        rank_logits = self.coral_head(pooled)
        
        if return_aux:
            aux_logits = self.aux_head(pooled)
            return rank_logits, aux_logits
            
        return rank_logits

    def forward_with_attention(
        self,
        images: torch.Tensor,
        radiomics: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        vis_tokens = self.extract_visual_tokens(images)
        fused_tokens, attn_weights = self.cross_attention(vis_tokens, radiomics)
        pooled = fused_tokens.mean(dim=1)
        pooled = self.dropout(pooled)
        
        rank_logits = self.coral_head(pooled)
        aux_logits = self.aux_head(pooled)
        return rank_logits, aux_logits, attn_weights

    def predict(
        self,
        images: torch.Tensor,
        radiomics: torch.Tensor,
        threshold: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        rank_logits, aux_logits, attn_weights = self.forward_with_attention(images, radiomics)
        preds = coral_predict(rank_logits, threshold=threshold)
        class_probs = coral_probabilities(rank_logits)
        rank_probs = torch.sigmoid(rank_logits)
        
        return {
            "predicted_grade": preds,
            "class_probabilities": class_probs,
            "rank_probabilities": rank_probs,
            "aux_logits": aux_logits,
            "attention_weights": attn_weights
        }

    def compute_hybrid_loss(
        self,
        rank_logits: torch.Tensor,
        aux_logits: Optional[torch.Tensor],
        targets: torch.Tensor,
        lambda_aux: float = 0.3
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        l_coral = coral_loss(rank_logits, targets, num_classes=self.num_classes)
        if aux_logits is not None and lambda_aux > 0:
            l_aux = F.cross_entropy(aux_logits, targets)
            total = l_coral + lambda_aux * l_aux
            return total, l_coral, l_aux
        return l_coral, l_coral, torch.tensor(0.0, device=targets.device)
