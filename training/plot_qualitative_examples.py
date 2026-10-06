"""Plot concept attention maps and contribution scores for the example frames.

The image list is ``data/examples/qualitative_images.txt``. Frames are read
from ``data/rgb/FaceForensics++/``. The checkpoint is
``checkpoints/df_cbm_best_video_auc.ckpt``.

Usage (from the repository root):

    python training/plot_qualitative_examples.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import torch
from tqdm import tqdm

_TRAINING_ROOT = Path(__file__).resolve().parent
if str(_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAINING_ROOT))

_REPO_ROOT = _TRAINING_ROOT.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(1, str(_REPO_ROOT))

from evaluation.eval_runner import load_eval_module, resolve_device
from plot_concept_attention_maps import (
    compute_attention,
    save_concept_panel,
    select_concept_indices,
    upsample_attention,
)
from plot_concept_contribution_scores import (
    compute_contribution_scores,
    output_stem_from_image_path,
    read_image_paths,
    save_contribution_bar_chart,
)
from utils.dataloader_utils import configure_torch_dataloader_env
from utils.dsutils import read_rgb_uint8_square

IMAGE_LIST = _REPO_ROOT / "data" / "examples" / "qualitative_images.txt"
DATASET_ROOT = _REPO_ROOT / "data" / "rgb" / "FaceForensics++"
SOURCE_ROOTS = (
    Path("/home/user/segface/datasets/rgb/FaceForensics++"),
    Path("/home/user/effort/datasets/rgb/FaceForensics++"),
)
CHECKPOINT = _REPO_ROOT / "checkpoints" / "df_cbm_best_video_auc.ckpt"
RESOLUTION = 224


def find_source_frame(relative: Path) -> Path | None:
    for root in SOURCE_ROOTS:
        candidate = root / relative
        if candidate.is_file():
            return candidate
    return None


def copy_example_frames(image_paths: list[Path]) -> None:
    missing: list[str] = []
    copied = 0
    for relative in image_paths:
        destination = DATASET_ROOT / relative
        if destination.is_file():
            continue
        source = find_source_frame(relative)
        if source is None:
            missing.append(relative.as_posix())
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied += 1
    if missing:
        searched = "\n".join(str(root) for root in SOURCE_ROOTS)
        raise FileNotFoundError(
            "Example frames were not found. Searched:\n"
            f"{searched}\nMissing:\n" + "\n".join(missing)
        )
    print(f"Copied {copied} example frame(s) into {DATASET_ROOT}")


def checkpoint_path() -> Path:
    if CHECKPOINT.is_file():
        return CHECKPOINT
    raise FileNotFoundError(
        "DF-CBM checkpoint not found at "
        f"{CHECKPOINT}. Download df_cbm_best_video_auc.ckpt into checkpoints/."
    )


def plot_examples(module, concept_names: list[str], image_paths: list[Path]) -> None:
    attention_count = 0
    contribution_count = 0

    for index, relative in enumerate(
        tqdm(image_paths, desc="Qualitative examples"), start=1
    ):
        example_dir = _REPO_ROOT / "outputs" / f"example_{index}"
        attention_dir = example_dir / "attention_maps"
        contribution_dir = example_dir / "concept_contributions"
        attention_dir.mkdir(parents=True, exist_ok=True)
        contribution_dir.mkdir(parents=True, exist_ok=True)

        image_path = DATASET_ROOT / relative
        rgb = read_rgb_uint8_square(str(image_path), RESOLUTION)
        image_rgb = torch.from_numpy(rgb)
        stem = output_stem_from_image_path(image_path)

        attention, concept_probs = compute_attention(module, image_rgb)
        concept_indices = select_concept_indices(
            concept_names,
            concept_probs,
            None,
            0.0,
        )
        for rank, concept_idx in enumerate(concept_indices, start=1):
            spatial = upsample_attention(attention[concept_idx], RESOLUTION)
            concept_name = concept_names[concept_idx]
            filename = f"{rank:02d}_{concept_name.replace(' ', '_')}.png"
            save_concept_panel(
                attention_dir / filename,
                rgb=rgb,
                spatial=spatial,
                concept_name=concept_name,
                overlay_alpha=0.5,
            )
            attention_count += 1

        contributions, _, _, _ = compute_contribution_scores(
            module,
            image_rgb,
            class_index=1,
        )
        save_contribution_bar_chart(
            contribution_dir / f"{stem}.png",
            concept_names=concept_names,
            contributions=contributions,
        )
        contribution_count += 1
        print(f"example_{index}: {attention_dir} and {contribution_dir}")

    print(
        f"Saved {attention_count} attention panel(s) and {contribution_count} contribution chart(s)."
    )


def main() -> None:
    configure_torch_dataloader_env()
    image_paths = read_image_paths(IMAGE_LIST)
    copy_example_frames(image_paths)

    device = resolve_device("cuda")
    ckpt = checkpoint_path()
    module = load_eval_module(str(ckpt), device)
    print(f"Checkpoint: {ckpt}")
    print(f"SegFace weights: {module.hparams.enc_cfg['segface_model_path']}")
    print(
        "Concept matrix: "
        f"{module.hparams.model_cfg['params']['concept_region_matrix_path']}"
    )
    concept_names = list(module.concept_names)
    if len(concept_names) != module.model.num_concepts:
        raise ValueError(
            f"Checkpoint lists {len(concept_names)} concepts but the model has "
            f"{module.model.num_concepts} outputs."
        )
    print("Concepts: " + ", ".join(concept_names))
    plot_examples(module, concept_names, image_paths)


if __name__ == "__main__":
    main()
