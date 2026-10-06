"""GPU SegFace + CLIP encoder for on-the-fly concept CBL training."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from segclip.model import load_segclip_models
from segclip.segclip_inference import compute_segclip_features_from_rgb_batch


class ConceptFeatureEncoder(nn.Module):
    """Frozen SegFace + CLIP backbone used to build CBL inputs on the GPU."""

    def __init__(self, config: Dict[str, Any], device: torch.device) -> None:
        super().__init__()
        self.config = config
        self.device = device
        segface_model, clip_model = load_segclip_models(config, device)
        self.segface_model = segface_model
        self.clip_model = clip_model
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.eval()

    @torch.no_grad()
    def encode(self, image_rgb: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode uint8 RGB images with shape [B, H, W, 3] on ``self.device``."""
        if image_rgb.ndim != 4 or image_rgb.shape[-1] != 3:
            raise ValueError(
                f"Expected image_rgb with shape [B, H, W, 3], got {tuple(image_rgb.shape)}."
            )
        image_bchw = image_rgb.permute(0, 3, 1, 2).contiguous()
        features = compute_segclip_features_from_rgb_batch(
            image_bchw,
            segface_model=self.segface_model,
            clip_model=self.clip_model,
            config=self.config,
        )
        return {
            "clip_patch_tokens": features["clip_patch_tokens"],
            "region_masks": features["region_masks"],
        }
