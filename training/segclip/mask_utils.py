"""Region patch mask stacking utilities (no detector imports)."""

from __future__ import annotations

import torch
from torch import Tensor


def stack_region_patch_masks(
    facial_patch_masks: Tensor,
    boundary_patch_masks: Tensor,
) -> Tensor:
    """Concatenate facial and boundary patch masks into [B, R, H, W] or [R, H, W]."""
    if facial_patch_masks.ndim != boundary_patch_masks.ndim:
        raise ValueError(
            "facial_patch_masks and boundary_patch_masks must have the same rank."
        )
    if facial_patch_masks.ndim not in {3, 4}:
        raise ValueError(
            "Expected facial_patch_masks and boundary_patch_masks with shape "
            "[R, H, W] or [B, R, H, W], got "
            f"{tuple(facial_patch_masks.shape)} and {tuple(boundary_patch_masks.shape)}."
        )
    return torch.cat([facial_patch_masks, boundary_patch_masks], dim=-3)


def merge_region_patch_masks(
    facial_patch_masks: Tensor,
    boundary_patch_masks: Tensor,
) -> Tensor:
    """Merge facial and boundary masks and flatten the patch grid to [R, N] or [B, R, N]."""
    stacked = stack_region_patch_masks(facial_patch_masks, boundary_patch_masks)
    if stacked.ndim == 3:
        return stacked.flatten(start_dim=1)
    return stacked.flatten(start_dim=2)
