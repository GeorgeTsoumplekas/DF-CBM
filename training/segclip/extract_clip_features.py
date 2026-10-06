import numpy as np
import torch
import torch.nn.functional as F

from segclip.inference import (
    BOUNDARY_DEFINITIONS,
    CELEBAMASK_LABELS,
    MERGE_TO_BACKGROUND_INDICES,
    MERGE_TO_SKIN_INDICES,
)

CLIP_SIZE = 224
CLIP_PATCH_SIZE = 14
CLIP_GRID_SIZE = CLIP_SIZE // CLIP_PATCH_SIZE

BASE_CLASS_NAMES = [
    CELEBAMASK_LABELS[index]
    for index in range(len(CELEBAMASK_LABELS))
    if index not in MERGE_TO_BACKGROUND_INDICES and index not in MERGE_TO_SKIN_INDICES
]
BOUNDARY_CLASS_NAMES = [name for name, _, _ in BOUNDARY_DEFINITIONS]


def class_masks_from_segmentation(pred_mask):
    """Binary mask per base class present in the postprocessed segmentation."""
    masks = {}
    for class_index in np.unique(pred_mask):
        name = CELEBAMASK_LABELS[int(class_index)]
        if name not in BASE_CLASS_NAMES:
            continue
        region = pred_mask == class_index
        if region.any():
            masks[name] = region
    return masks


def boundary_masks_from_inference(boundaries):
    """Boundary masks from compute_boundary_masks."""
    return {name: region.astype(bool) for name, region in boundaries.items()}


def mask_to_tensor(mask_hw, device):
    if mask_hw.shape != (CLIP_SIZE, CLIP_SIZE):
        raise ValueError(
            f"Expected mask shape ({CLIP_SIZE}, {CLIP_SIZE}), got {mask_hw.shape}"
        )
    return torch.from_numpy(mask_hw.astype(np.float32))[None, None].to(device)


def downsample_mask_to_patch_grid(mask_hw, device):
    """
    Avg-pool a pixel mask to the CLIP patch grid.

    mask_hw: [224, 224] bool/float
    returns: [Gh, Gw] float32 patch weights in [0, 1]
    """
    mask = mask_to_tensor(mask_hw, device)
    mask_patch = F.avg_pool2d(
        mask,
        kernel_size=CLIP_PATCH_SIZE,
        stride=CLIP_PATCH_SIZE,
    )
    return mask_patch.squeeze(0).squeeze(0)


def standardize_patch_masks(
    class_masks,
    boundary_masks,
    device,
    base_class_names=BASE_CLASS_NAMES,
    boundary_class_names=BOUNDARY_CLASS_NAMES,
):
    """
    Fixed-size patch-grid masks aligned with BASE_CLASS_NAMES / BOUNDARY_CLASS_NAMES.
    Missing regions are all-zero [Gh, Gw] masks.
    """
    facial = torch.zeros(
        len(base_class_names), CLIP_GRID_SIZE, CLIP_GRID_SIZE, device=device
    )
    boundary = torch.zeros(
        len(boundary_class_names), CLIP_GRID_SIZE, CLIP_GRID_SIZE, device=device
    )
    for idx, name in enumerate(base_class_names):
        if name in class_masks:
            facial[idx] = downsample_mask_to_patch_grid(class_masks[name], device)
    for idx, name in enumerate(boundary_class_names):
        if name in boundary_masks:
            boundary[idx] = downsample_mask_to_patch_grid(boundary_masks[name], device)
    return facial, boundary
