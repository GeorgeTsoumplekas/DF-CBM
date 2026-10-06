"""Plot concept-specific attention maps for a RegionAwareCBM checkpoint.

For each image, reads the concept attention weights

    alpha_{k,i} = softmax_i( q_k^T W f_i / sqrt(d) + log(p_{k,i} + eps) )

and saves a three-panel figure per concept: the input image, the upsampled
attention heatmap, and the heatmap overlaid on the image.

Usage (from any working directory):

    python training/plot_concept_attention_maps.py \\
        --image-list data/examples/qualitative_images.txt \\
        --checkpoint checkpoints/df_cbm_best_video_auc.ckpt \\
        --output-dir outputs/attention_maps
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

_TRAINING_ROOT = Path(__file__).resolve().parent
if str(_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAINING_ROOT))

_REPO_ROOT = _TRAINING_ROOT.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(1, str(_REPO_ROOT))

from config.parse_config import load_and_parse_config, parse_segclip_config
from evaluation.eval_runner import load_eval_module, resolve_device
from plot_concept_contribution_scores import (
    output_stem_from_image_path,
    read_image_paths,
    resolve_image_path,
    resolve_training_concept_names,
    validate_concept_names,
)
from utils.dataloader_utils import configure_torch_dataloader_env
from utils.dsutils import read_rgb_uint8_square


def grid_size_from_num_patches(num_patches: int) -> int:
    grid_size = int(round(num_patches**0.5))
    if grid_size * grid_size != num_patches:
        raise ValueError(
            f"Attention has {num_patches} patches, which is not a square grid."
        )
    return grid_size


def upsample_attention(patch_values: np.ndarray, image_size: int) -> np.ndarray:
    grid_size = grid_size_from_num_patches(patch_values.shape[0])
    grid = patch_values.reshape(grid_size, grid_size).astype(np.float32)
    return cv2.resize(
        grid,
        (image_size, image_size),
        interpolation=cv2.INTER_CUBIC,
    )


def attention_overlay(
    rgb: np.ndarray, spatial: np.ndarray, vmax: float, alpha: float
) -> np.ndarray:
    heat = np.clip(spatial / vmax, 0.0, 1.0)
    colored = plt.get_cmap("jet")(heat)[..., :3]
    image = rgb.astype(np.float32) / 255.0
    blended = alpha * colored + (1.0 - alpha) * image
    return np.clip(blended, 0.0, 1.0)


@torch.no_grad()
def compute_attention(module, image_rgb: torch.Tensor):
    device = next(module.parameters()).device
    image_rgb = image_rgb.unsqueeze(0).to(device)
    encoded = module.feature_encoder.encode(image_rgb)
    tokens = encoded["clip_patch_tokens"]
    masks = encoded["region_masks"]
    output = module.model.concept_bottleneck(
        tokens,
        masks,
        return_attention=True,
        return_probs=True,
    )
    attention = output["attention"][0].detach().cpu().numpy()
    concept_probs = output["concept_probs"][0].detach().cpu().numpy()
    return attention, concept_probs


def save_concept_panel(
    output_path: Path,
    *,
    rgb: np.ndarray,
    spatial: np.ndarray,
    concept_name: str,
    overlay_alpha: float,
) -> None:
    vmax = float(max(spatial.max(), 1e-8))
    overlay = attention_overlay(rgb, spatial, vmax, overlay_alpha)

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.2))
    axes[0].imshow(rgb)
    axes[0].set_title("Original Image")

    heat = axes[1].imshow(spatial, cmap="jet", vmin=0.0, vmax=vmax)
    axes[1].set_title(f"Attention for: {concept_name}")
    fig.colorbar(heat, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay ({concept_name})")

    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def select_concept_indices(
    concept_names: Sequence[str],
    concept_probs: np.ndarray,
    requested_names: Sequence[str] | None,
    min_prob: float,
) -> List[int]:
    if requested_names:
        name_to_index = {name: idx for idx, name in enumerate(concept_names)}
        missing = [name for name in requested_names if name not in name_to_index]
        if missing:
            raise ValueError(
                f"Unknown concept name(s): {missing}. "
                f"Available concepts: {list(concept_names)}"
            )
        indices = [name_to_index[name] for name in requested_names]
    else:
        indices = list(range(len(concept_names)))

    indices = [idx for idx in indices if float(concept_probs[idx]) >= min_prob]
    indices.sort(key=lambda idx: float(concept_probs[idx]), reverse=True)
    return indices


def main() -> None:
    configure_torch_dataloader_env()

    parser = argparse.ArgumentParser(
        description="Save concept-specific attention maps for a RegionAwareCBM checkpoint."
    )
    parser.add_argument("--image-list", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent / "configs" / "config.yaml"),
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Root used to resolve relative image paths. Defaults to dataset_config.rootpath.",
    )
    parser.add_argument(
        "--concepts",
        type=str,
        default=None,
        help="Comma-separated concept names. Defaults to every concept.",
    )
    parser.add_argument(
        "--min-prob",
        type=float,
        default=0.0,
        help="Skip concepts whose predicted probability is below this value.",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.5,
        help="Heatmap opacity in the overlay panel.",
    )
    args = parser.parse_args()

    cfg = load_and_parse_config(config_path=args.config)
    cfg = parse_segclip_config(cfg)
    ds_cfg = cfg.get("dataset_config", {})
    device = resolve_device(cfg.get("training_config", {}).get("device", "cuda"))
    resolution = int(ds_cfg.get("resolution", 224))

    dataset_root = Path(args.dataset_root or ds_cfg.get("rootpath", "")).expanduser()
    if not dataset_root.is_dir() and args.dataset_root is None:
        dataset_root = None

    module = load_eval_module(args.checkpoint, device)
    concept_names = resolve_training_concept_names(ds_cfg)
    validate_concept_names(
        concept_names,
        module.concept_names,
        num_model_outputs=module.model.num_concepts,
    )
    requested = None
    if args.concepts:
        requested = [name.strip() for name in args.concepts.split(",") if name.strip()]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    for image_path in tqdm(
        read_image_paths(Path(args.image_list)), desc="Plotting attention maps"
    ):
        resolved_path = resolve_image_path(image_path, dataset_root)
        rgb = read_rgb_uint8_square(str(resolved_path), resolution)
        attention, concept_probs = compute_attention(module, torch.from_numpy(rgb))
        concept_indices = select_concept_indices(
            concept_names,
            concept_probs,
            requested,
            args.min_prob,
        )
        sample_dir = output_dir / output_stem_from_image_path(resolved_path)
        for rank, concept_idx in enumerate(concept_indices, start=1):
            spatial = upsample_attention(attention[concept_idx], resolution)
            concept_name = concept_names[concept_idx]
            filename = f"{rank:02d}_{concept_name.replace(' ', '_')}.png"
            save_concept_panel(
                sample_dir / filename,
                rgb=rgb,
                spatial=spatial,
                concept_name=concept_name,
                overlay_alpha=args.overlay_alpha,
            )
            saved += 1

    print(f"Saved {saved} attention panel(s) to {output_dir}")


if __name__ == "__main__":
    main()
