"""Plot per-image concept contribution bar charts for a RegionAwareCBM checkpoint.

For each image path listed in an input text file, computes

    contrib_{n,k} = w_{cls,k} * c_hat_{n,k}

where w_{cls,k} is the linear classifier weight for concept k (for the chosen
class) and c_hat_{n,k} is the predicted concept probability (sigmoid of the
concept logit). Saves one bar chart PNG per image.

Concept names are resolved from dataset_config using the same merge map and
min_samples_per_concept filtering as training (6 concepts by default).

Usage (from the repository root):

    python training/plot_concept_contribution_scores.py \\
        --image-list data/examples/qualitative_images.txt \\
        --checkpoint checkpoints/df_cbm_best_video_auc.ckpt \\
        --output-dir outputs/concept_contributions
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Sequence

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
from utils.concept_utils import get_filepath_to_concept_indices, ordered_concept_names
from utils.dataloader_utils import configure_torch_dataloader_env
from utils.dsutils import read_rgb_uint8_square

EXPECTED_NUM_CONCEPTS = 6


def read_image_paths(list_path: Path) -> List[Path]:
    paths: List[Path] = []
    with open(list_path, encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            paths.append(Path(stripped))
    if not paths:
        raise ValueError(f"No image paths found in {list_path}")
    return paths


def resolve_image_path(image_path: Path, dataset_root: Path | None) -> Path:
    if image_path.is_file():
        return image_path.resolve()
    if dataset_root is not None:
        joined = dataset_root / image_path
        if joined.is_file():
            return joined.resolve()
    raise FileNotFoundError(f"Could not resolve image path: {image_path}")


def resolve_training_concept_names(ds_cfg: dict) -> List[str]:
    """Same concept selection as FFDataModule / FFWithConcepts training.

    Applies CONCEPT_MERGE_MAP (e.g. mouth blurring -> mouth artifacts) and keeps
    only concepts with at least min_samples_per_concept appearances in both train
    and test splits (default 750).
    """
    _, concept_name_to_id = get_filepath_to_concept_indices(
        datarootpath=ds_cfg["rootpath"],
        concept_json_label=ds_cfg["concept_label_filepath"],
        min_appearances=int(ds_cfg.get("min_samples_per_concept", 750)),
        train_splits=ds_cfg.get("train_splits", ["train", "val"]),
        test_splits=ds_cfg.get("test_splits", ["test"]),
        fake_subsets=ds_cfg.get("fake_subsets"),
    )
    return ordered_concept_names(concept_name_to_id)


def validate_concept_names(
    dataset_concept_names: Sequence[str],
    checkpoint_concept_names: Sequence[str] | None,
    num_model_outputs: int,
) -> None:
    if len(dataset_concept_names) != EXPECTED_NUM_CONCEPTS:
        raise ValueError(
            f"Expected {EXPECTED_NUM_CONCEPTS} concepts after merge/filter, "
            f"got {len(dataset_concept_names)}: {list(dataset_concept_names)}. "
            "Check dataset_config.min_samples_per_concept and concept_labels.json."
        )
    if num_model_outputs != len(dataset_concept_names):
        raise ValueError(
            f"Model has {num_model_outputs} concept outputs but dataset resolves "
            f"{len(dataset_concept_names)} concepts."
        )
    if checkpoint_concept_names and list(checkpoint_concept_names) != list(
        dataset_concept_names
    ):
        raise ValueError(
            "Checkpoint concept_names do not match dataset-resolved concepts.\n"
            f"  checkpoint: {list(checkpoint_concept_names)}\n"
            f"  dataset:    {list(dataset_concept_names)}"
        )


def output_stem_from_image_path(image_path: Path) -> str:
    """Use path segment after ``frames/`` and before ``.png`` as the output stem."""
    path_str = str(image_path).replace("\\", "/")
    match = re.search(r"/frames/(.+?)\.png$", path_str, flags=re.IGNORECASE)
    if not match:
        raise ValueError(
            f"Could not extract output name from path (expected .../frames/.../*.png): "
            f"{image_path}"
        )
    return slugify(match.group(1).replace("/", "_"))


def slugify(name: str, max_len: int = 120) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    slug = slug.strip("._")
    return slug[:max_len] or "image"


def get_classifier_weights(
    classifier: torch.nn.Module, class_index: int
) -> torch.Tensor:
    if not isinstance(classifier, torch.nn.Linear):
        raise ValueError("Model classifier must be nn.Linear.")
    weight = classifier.weight

    num_classes = weight.shape[0]
    if class_index < 0 or class_index >= num_classes:
        raise ValueError(
            f"class_index={class_index} is out of range for {num_classes} classes."
        )
    return weight[class_index]


@torch.no_grad()
def compute_contribution_scores(
    module,
    image_rgb: torch.Tensor,
    *,
    class_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return (contributions, concept_probs, classifier_weights, fake_prob) for one image."""
    device = next(module.parameters()).device
    image_rgb = image_rgb.unsqueeze(0).to(device)

    encoded = module.feature_encoder.encode(image_rgb)
    tokens = encoded["clip_patch_tokens"]
    masks = encoded["region_masks"]

    class_logits, concept_logits = module(tokens, masks)
    concept_probs = torch.sigmoid(concept_logits)[0]
    classifier_weights = get_classifier_weights(module.model.classifier, class_index)

    contributions = classifier_weights * concept_probs
    fake_probability = float(torch.softmax(class_logits, dim=1)[0, 1].cpu())
    return (
        contributions.detach().cpu().numpy(),
        concept_probs.detach().cpu().numpy(),
        classifier_weights.detach().cpu().numpy(),
        fake_probability,
    )


def save_contribution_bar_chart(
    output_path: Path,
    *,
    concept_names: Sequence[str],
    contributions: np.ndarray,
) -> None:
    order = np.argsort(contributions)
    sorted_names = [concept_names[idx] for idx in order]
    sorted_contrib = contributions[order]

    num_concepts = len(concept_names)
    axis_label_fontsize = 14
    concept_label_fontsize = 12
    value_label_fontsize = 11

    fig_height = max(4.0, 0.32 * num_concepts + 1.5)
    fig, ax = plt.subplots(figsize=(10.0, fig_height))

    bar_color = "#4C78A8"
    y_positions = np.arange(num_concepts)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#D3D3D3", linestyle="-", linewidth=0.8, zorder=0)
    ax.barh(y_positions, sorted_contrib, color=bar_color, edgecolor="none", zorder=3)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(sorted_names, fontsize=concept_label_fontsize)
    ax.axvline(0.0, color="black", linewidth=0.8, zorder=2)
    ax.set_xlabel("Concept contribution scores", fontsize=axis_label_fontsize)
    ax.set_ylabel("Concept", fontsize=axis_label_fontsize)

    x_offset = 0.01 * (np.max(np.abs(sorted_contrib)) + 1e-8)
    value_texts = []
    for idx, contrib in enumerate(sorted_contrib):
        ha = "left" if contrib >= 0 else "right"
        x_pos = contrib + (x_offset if contrib >= 0 else -x_offset)
        value_texts.append(
            ax.text(
                x_pos,
                idx,
                f"{contrib:.3f}",
                va="center",
                ha=ha,
                fontsize=value_label_fontsize,
                color="#333333",
                zorder=4,
            )
        )

    x_min = min(0.0, float(sorted_contrib.min()))
    x_max = max(0.0, float(sorted_contrib.max()))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    transform = ax.transData.inverted()
    for text in value_texts:
        bbox = text.get_window_extent(renderer=renderer)
        bbox_data = transform.transform_bbox(bbox)
        x_min = min(x_min, bbox_data.x0)
        x_max = max(x_max, bbox_data.x1)

    span = max(x_max - x_min, 1e-8)
    margin = span * 0.04
    ax.set_xlim(x_min - margin, x_max + margin)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_torch_dataloader_env()

    parser = argparse.ArgumentParser(
        description=(
            "Compute concept contribution scores and save per-image bar chart PNGs "
            "for a RegionAwareCBM checkpoint."
        )
    )
    parser.add_argument(
        "--image-list",
        type=str,
        required=True,
        help="Text file with one image path per line (FaceForensics++ paths).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a SegClipCBMLightningModule checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent / "configs" / "config.yaml"),
        help="Path to the configuration YAML file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where contribution bar chart PNGs are saved.",
    )
    parser.add_argument(
        "--class-index",
        type=int,
        default=1,
        help="Classifier output index used for w_{cls,k} (default: 1 = fake).",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help=(
            "Optional root for resolving relative image paths "
            "(defaults to dataset_config.rootpath from config)."
        ),
    )
    args = parser.parse_args()

    cfg = load_and_parse_config(config_path=args.config)
    cfg = parse_segclip_config(cfg)

    train_cfg = cfg.get("training_config", {})
    ds_cfg = cfg.get("dataset_config", {})
    device = resolve_device(train_cfg.get("device", "cuda"))
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
    print(
        f"Using {len(concept_names)} merged/filtered concepts: "
        + ", ".join(concept_names)
    )

    image_paths = read_image_paths(Path(args.image_list))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for image_path in tqdm(image_paths, desc="Plotting contribution scores"):
        resolved_path = resolve_image_path(image_path, dataset_root)
        image_rgb_np = read_rgb_uint8_square(str(resolved_path), resolution)
        image_rgb = torch.from_numpy(image_rgb_np)

        contributions, _, _, _ = compute_contribution_scores(
            module,
            image_rgb,
            class_index=args.class_index,
        )

        output_stem = output_stem_from_image_path(resolved_path)
        output_path = output_dir / f"{output_stem}.png"
        save_contribution_bar_chart(
            output_path,
            concept_names=concept_names,
            contributions=contributions,
        )
        saved += 1

    print(f"Saved {saved} bar chart PNG(s) to {output_dir}")


if __name__ == "__main__":
    main()
