import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union, Optional

def compute_effective_number_weights(
    class_counts: Union[List[int], np.ndarray, torch.Tensor],
    beta: float = 0.9999
) -> torch.Tensor:
    counts = np.array(class_counts, dtype=np.float64)
    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / (effective_num + 1e-8)
    num_classes = len(counts)
    weights = weights / np.sum(weights) * float(num_classes)
    return torch.tensor(weights, dtype=torch.float32)

class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean"
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", alpha.clone().detach().float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1.0 - pt) ** self.gamma
        
        if self.alpha is not None:
            at = self.alpha.to(logits.device).gather(0, targets)
            loss = at * focal_weight * ce_loss
        else:
            loss = focal_weight * ce_loss
            
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

class ClassBalancedFocalLoss(nn.Module):
    def __init__(
        self,
        class_counts: Union[List[int], np.ndarray],
        beta: float = 0.9999,
        gamma: float = 2.0,
        reduction: str = "mean"
    ):
        super().__init__()
        weights = compute_effective_number_weights(class_counts, beta=beta)
        self.focal_loss = FocalLoss(alpha=weights, gamma=gamma, reduction=reduction)
        self.class_weights = weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.focal_loss(logits, targets)

class CostSensitiveLoss(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        cost_matrix: Optional[torch.Tensor] = None,
        reduction: str = "mean"
    ):
        super().__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        
        if cost_matrix is None:
            grid = np.arange(num_classes)
            mat = (grid[:, None] - grid[None, :]) ** 2
            mat = mat / float((num_classes - 1) ** 2)
            cost_matrix = torch.tensor(mat, dtype=torch.float32)
            
        self.register_buffer("cost_matrix", cost_matrix)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        
        cost_mat = self.cost_matrix.to(logits.device)
        sample_costs = cost_mat[targets]
        expected_cost = (sample_costs * probs).sum(dim=-1)
        
        total_loss = ce_loss * (1.0 + expected_cost)
        
        if self.reduction == "mean":
            return total_loss.mean()
        elif self.reduction == "sum":
            return total_loss.sum()
        else:
            return total_loss

def build_loss_function(
    loss_name: str,
    class_counts: Union[List[int], np.ndarray],
    **kwargs
) -> nn.Module:
    loss_name = loss_name.lower()
    if loss_name in ["ce", "cross_entropy"]:
        return nn.CrossEntropyLoss()
    elif loss_name in ["weighted_ce", "inverse_freq"]:
        weights = 1.0 / (np.array(class_counts, dtype=np.float32) + 1e-5)
        weights = weights / np.sum(weights) * len(class_counts)
        return nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
    elif loss_name == "focal":
        gamma = kwargs.get("gamma", 2.0)
        return FocalLoss(gamma=gamma)
    elif loss_name in ["cb_focal", "class_balanced_focal"]:
        beta = kwargs.get("beta", 0.9999)
        gamma = kwargs.get("gamma", 2.0)
        return ClassBalancedFocalLoss(class_counts, beta=beta, gamma=gamma)
    elif loss_name in ["cost_sensitive", "cost"]:
        return CostSensitiveLoss(num_classes=len(class_counts))
    else:
        raise ValueError(f"Unknown loss function name: {loss_name}")
