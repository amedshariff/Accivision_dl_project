"""
Loss Functions Module (Training)
Provides Focal / Weighted BCE Loss to prioritize accident recall.
"""
import torch
import torch.nn as nn

class WeightedAccidentLoss(nn.Module):
    def __init__(self, accident_weight: float = 2.0):
        super(WeightedAccidentLoss, self).__init__()
        self.pos_weight = torch.tensor([accident_weight])

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(logits.device))
        return loss_fn(logits, targets)
