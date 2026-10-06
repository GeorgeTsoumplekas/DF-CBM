import torch
import torch.nn as nn
from losses.registry import register_loss


@register_loss("NoisyConceptBCELoss")
class NoisyConceptBCELoss(nn.Module):
    def __init__(self, default_pos_weight=1.0, neg_weight=1.0, noise_discount=0.2):
        super().__init__()
        self.pos_weight = default_pos_weight
        self.neg_weight = neg_weight
        self.noise_discount = noise_discount
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, targets, is_real_sample):
        pos_tensor = torch.tensor(
            self.pos_weight, dtype=torch.float32, device=logits.device
        )

        if pos_tensor.dim() == 1:
            pos_tensor = pos_tensor.unsqueeze(0)

        if is_real_sample.dim() == 1:
            is_real_sample = is_real_sample.unsqueeze(1)

        base_loss = self.bce(logits, targets)

        weight_mask = torch.where(
            targets == 1.0,
            pos_tensor,
            torch.where(
                is_real_sample == 1.0,
                self.neg_weight,
                self.neg_weight * self.noise_discount,
            ),
        )

        weighted_loss = base_loss * weight_mask
        return weighted_loss.mean()
