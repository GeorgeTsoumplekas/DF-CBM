"""Shared SegFace + CLIP feature extraction for concept training."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import cv2
import numpy as np
import torch
from torchvision.transforms.functional import normalize as tv_normalize

from segclip.extract_clip_features import (
    BASE_CLASS_NAMES,
    BOUNDARY_CLASS_NAMES,
    boundary_masks_from_inference,
    class_masks_from_segmentation,
    standardize_patch_masks,
)
from segclip.inference import (
    BOUNDARY_DEFINITIONS,
    MERGE_TO_BACKGROUND_INDICES,
    MERGE_TO_SKIN_INDICES,
    SKIN_CLASS_INDEX,
    compute_boundary_masks,
    predict_segmentation,
)
from segclip.mask_utils import merge_region_patch_masks


def postprocess_segmentation(mask: np.ndarray) -> np.ndarray:

    processed = mask.copy()
    for class_index in MERGE_TO_BACKGROUND_INDICES:
        processed[processed == class_index] = 0
    for class_index in MERGE_TO_SKIN_INDICES:
        processed[processed == class_index] = SKIN_CLASS_INDEX
    return processed


def build_patch_region_masks(
    pred_mask: np.ndarray,
    *,
    clip_size: int,
    boundary_defs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if pred_mask.shape != (clip_size, clip_size):
        pred_mask = cv2.resize(
            pred_mask,
            (clip_size, clip_size),
            interpolation=cv2.INTER_NEAREST,
        )

    class_masks = class_masks_from_segmentation(pred_mask)
    boundaries = compute_boundary_masks(pred_mask, definitions=boundary_defs)
    boundary_masks = boundary_masks_from_inference(boundaries)
    facial_patch_masks, boundary_patch_masks = standardize_patch_masks(
        class_masks,
        boundary_masks,
        torch.device("cpu"),
    )
    return facial_patch_masks, boundary_patch_masks


def verify_undetected_regions_are_zero(
    facial_patch_masks: torch.Tensor,
    boundary_patch_masks: torch.Tensor,
    class_masks: dict,
    boundary_masks: dict,
) -> None:
    """Ensure regions absent from SegFace output have all-zero patch masks."""
    for idx, name in enumerate(BASE_CLASS_NAMES):
        if (
            name not in class_masks
            and facial_patch_masks[idx].abs().sum().item() != 0.0
        ):
            raise ValueError(
                f"Facial region '{name}' was not detected but has non-zero patch mask."
            )
    for idx, name in enumerate(BOUNDARY_CLASS_NAMES):
        if (
            name not in boundary_masks
            and boundary_patch_masks[idx].abs().sum().item() != 0.0
        ):
            raise ValueError(
                f"Boundary region '{name}' was not detected but has non-zero patch mask."
            )


def build_merged_region_patch_masks(
    pred_mask: np.ndarray,
    *,
    clip_size: int,
    boundary_defs,
) -> torch.Tensor:
    """Build merged facial+boundary patch masks with shape [R, P]."""

    if pred_mask.shape != (clip_size, clip_size):
        pred_mask = cv2.resize(
            pred_mask,
            (clip_size, clip_size),
            interpolation=cv2.INTER_NEAREST,
        )

    facial_patch_masks, boundary_patch_masks = build_patch_region_masks(
        pred_mask,
        clip_size=clip_size,
        boundary_defs=boundary_defs,
    )
    class_masks = class_masks_from_segmentation(pred_mask)
    boundaries = compute_boundary_masks(pred_mask, definitions=boundary_defs)
    boundary_masks = boundary_masks_from_inference(boundaries)
    verify_undetected_regions_are_zero(
        facial_patch_masks,
        boundary_patch_masks,
        class_masks,
        boundary_masks,
    )
    return merge_region_patch_masks(facial_patch_masks, boundary_patch_masks)


@torch.no_grad()
def encode_clip_patch_tokens_from_rgb_batch(
    clip_model,
    image_rgb: torch.Tensor,
    *,
    clip_mean,
    clip_std,
) -> torch.Tensor:
    """Encode CLIP patch tokens for a batch already on the model device."""
    pixel_values = tv_normalize(image_rgb, mean=clip_mean, std=clip_std)
    vision_outputs = clip_model.vision_model(pixel_values=pixel_values)
    patch_tokens = vision_outputs.last_hidden_state[:, 1:, :]
    patch_tokens = clip_model.vision_model.post_layernorm(patch_tokens)
    patch_tokens = clip_model.visual_projection(patch_tokens)
    return patch_tokens


@torch.no_grad()
def infer_segface_segmentation_from_rgb_batch(
    segface_model,
    image_rgb: torch.Tensor,
    *,
    segface_size: int,
    segface_mean,
    segface_std,
) -> torch.Tensor:
    """Run SegFace on a batch of RGB images already on the model device."""

    if image_rgb.dtype == torch.uint8:
        image_tensor = image_rgb.float() / 255.0
    else:
        image_tensor = image_rgb
    image_tensor = tv_normalize(image_tensor, mean=segface_mean, std=segface_std)
    device = image_tensor.device
    batch_size = image_tensor.shape[0]
    labels = {
        "lnm_seg": torch.zeros(batch_size, 5, 2, device=device),
        "segmentation": torch.zeros(
            batch_size,
            segface_size,
            segface_size,
            device=device,
        ),
    }
    dataset_tensor = torch.zeros(batch_size, dtype=torch.long, device=device)
    pred = predict_segmentation(
        segface_model,
        image_tensor,
        labels,
        dataset_tensor,
        segface_size,
    )
    return pred.to(dtype=torch.uint8)


@torch.no_grad()
def compute_segclip_features_from_rgb_batch(
    image_rgb: torch.Tensor,
    *,
    segface_model,
    clip_model,
    config: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    """Compute CLIP patch tokens and merged region masks on the GPU."""

    device = image_rgb.device
    clip_size = int(config.get("clip_input_resolution", config["resolution"]))
    segface_size = int(config.get("segface_input_resolution", config["resolution"]))
    segface_mean = config.get("segface_mean", [0.485, 0.456, 0.406])
    segface_std = config.get("segface_std", [0.229, 0.224, 0.225])
    clip_mean = config.get("clip_mean", [0.48145466, 0.4578275, 0.40821073])
    clip_std = config.get("clip_std", [0.26862954, 0.26130258, 0.27577711])

    if image_rgb.shape[-2] != segface_size or image_rgb.shape[-1] != segface_size:
        resized = image_rgb.float() if image_rgb.dtype == torch.uint8 else image_rgb
        image_rgb = torch.nn.functional.interpolate(
            resized,
            size=(segface_size, segface_size),
            mode="bilinear",
            align_corners=False,
        )
        if image_rgb.dtype != torch.uint8:
            image_rgb = image_rgb.clamp(0, 255).to(torch.uint8)

    pred_masks = infer_segface_segmentation_from_rgb_batch(
        segface_model,
        image_rgb,
        segface_size=segface_size,
        segface_mean=segface_mean,
        segface_std=segface_std,
    )

    clip_input = (
        image_rgb.float() / 255.0 if image_rgb.dtype == torch.uint8 else image_rgb
    )
    if clip_input.shape[-2] != clip_size or clip_input.shape[-1] != clip_size:
        clip_input = torch.nn.functional.interpolate(
            clip_input,
            size=(clip_size, clip_size),
            mode="bilinear",
            align_corners=False,
        )
    patch_tokens = encode_clip_patch_tokens_from_rgb_batch(
        clip_model,
        clip_input,
        clip_mean=clip_mean,
        clip_std=clip_std,
    )

    region_masks = []
    for pred_mask in pred_masks.detach().cpu().numpy():
        processed = postprocess_segmentation(pred_mask.astype("uint8"))
        region_masks.append(
            build_merged_region_patch_masks(
                processed,
                clip_size=clip_size,
                boundary_defs=BOUNDARY_DEFINITIONS,
            )
        )
    region_patch_masks = torch.stack(region_masks, dim=0).to(
        device=device,
        dtype=torch.float32,
    )
    return {
        "clip_patch_tokens": patch_tokens.to(dtype=torch.float32),
        "region_masks": region_patch_masks,
    }
