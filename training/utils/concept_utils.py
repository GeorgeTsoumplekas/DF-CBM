import csv
import os
from itertools import chain
from collections import Counter
from typing import List, Sequence, Tuple

import torch
from torch import Tensor

from utils.dsutils import get_ff_filepath_video_id, get_ff_split_ids
from utils.ioutils import load_json, get_filepaths


EXCLUDED_FAKE_SUBSETS = ["DeepFakeDetection"]


_FACIAL_REGION_ALIASES = {
    "l_ear": "left ear",
    "r_ear": "right ear",
    "l_brow": "left eyebrow",
    "r_brow": "right eyebrow",
    "l_eye": "left eye",
    "r_eye": "right eye",
    "l_lip": "lower lip",
    "u_lip": "upper lip",
}


CONCEPT_MERGE_MAP = {
    "nose distortion": "distorted nose",
    "mouth blurring": "mouth artifacts",
    "severe mouth distortion": "mouth artifacts",
}


def merge_concept_names(concept_names):
    merged = {CONCEPT_MERGE_MAP.get(name, name) for name in concept_names}
    return merged


def canonical_region_name(name: str) -> str:
    underscored = name.strip().lower().replace(" ", "_")
    if underscored in _FACIAL_REGION_ALIASES:
        return _FACIAL_REGION_ALIASES[underscored]
    return name.strip().lower()


def get_segface_region_names() -> List[str]:
    from segclip.inference import (
        BOUNDARY_DEFINITIONS,
        CELEBAMASK_LABELS,
        MERGE_TO_BACKGROUND_INDICES,
        MERGE_TO_SKIN_INDICES,
    )

    base_class_names = [
        CELEBAMASK_LABELS[index]
        for index in range(len(CELEBAMASK_LABELS))
        if index not in MERGE_TO_BACKGROUND_INDICES
        and index not in MERGE_TO_SKIN_INDICES
    ]
    boundary_class_names = [name for name, _, _ in BOUNDARY_DEFINITIONS]
    class_names = list(base_class_names) + list(boundary_class_names)
    return [canonical_region_name(name) for name in class_names]


def _align_concept_region_matrix(
    region_names: Sequence[str],
    rows: Sequence[Sequence[float]],
) -> Tuple[List[str], List[List[float]]]:
    expected_names = get_segface_region_names()
    csv_index = {
        canonical_region_name(name): idx for idx, name in enumerate(region_names)
    }

    missing = [name for name in expected_names if name not in csv_index]
    if missing:
        raise ValueError(
            "concept_region_mapping_matrix.csv is missing regions required by SegFace: "
            + ", ".join(missing)
        )

    reorder = [csv_index[name] for name in expected_names]
    aligned_rows = [[row[idx] for idx in reorder] for row in rows]
    return expected_names, aligned_rows


def load_concept_region_assets(
    concept_region_matrix_path: str,
) -> Tuple[Tensor, List[str], List[str]]:
    path = os.path.abspath(concept_region_matrix_path)
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")

        region_names = [
            name
            for name in reader.fieldnames
            if name not in {"concept_id", "concept_name"}
        ]
        concept_names: List[str] = []
        rows: List[List[float]] = []

        for row in reader:
            concept_names.append(row["concept_name"])
            rows.append([float(row[name]) for name in region_names])

    region_names, rows = _align_concept_region_matrix(region_names, rows)
    concept_region_matrix = torch.tensor(rows, dtype=torch.float32)
    return concept_region_matrix, concept_names, region_names


def encode_clip_text_embeddings(
    concept_names: Sequence[str],
    clip_model_id: str,
    device: torch.device,
) -> Tensor:
    from transformers import CLIPModel, CLIPTokenizer

    clip_model = CLIPModel.from_pretrained(clip_model_id).to(device).eval()
    tokenizer = CLIPTokenizer.from_pretrained(clip_model_id)

    batch_size = 64
    embeddings = []
    for start in range(0, len(concept_names), batch_size):
        batch_names = list(concept_names[start : start + batch_size])
        tokens = tokenizer(
            batch_names,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        text_features = clip_model.get_text_features(**tokens)
        embeddings.append(text_features.detach().cpu())

    return torch.cat(embeddings, dim=0)


def normalize_rel_frame_path(path):
    path = path.replace("\\", "/")
    if path.startswith("FaceForensics++/"):
        path = path[len("FaceForensics++/") :]
    if path.startswith("ff++/fake/"):
        path = path.replace("ff++/fake/", "manipulated_sequences/")
    elif path.startswith("ff++/"):
        path = path.replace("ff++/", "")
    return path


def get_concept_per_filepath(concept_json_file, valid_fake_frames):
    json_data = load_json(concept_json_file)
    concept_frames = dict()
    for entry in json_data:
        entry_cluster_names = merge_concept_names(entry.get("concept_name", []))
        entry_video = normalize_rel_frame_path(entry.get("ff_video"))
        entry_frames = entry.get("frames")
        entry_rel_frame_paths = [
            normalize_rel_frame_path(os.path.join(entry_video, frame + ".png"))
            for frame in entry_frames
        ]

        for each in entry_rel_frame_paths:
            if each in valid_fake_frames:
                if each not in concept_frames:
                    concept_frames[each] = set()
                concept_frames[each].update(entry_cluster_names)

    all_concept_set = set()
    for k, v in concept_frames.items():
        all_concept_set.update(v)

    concept_to_id = {
        concept: idx for idx, concept in enumerate(sorted(list(all_concept_set)))
    }

    filepath_to_concept_indices = dict()
    for k, v in concept_frames.items():
        indices = [concept_to_id[each] for each in v]
        filepath_to_concept_indices[k] = indices

    return filepath_to_concept_indices, concept_to_id


def is_excluded_fake_path(rel_path):
    return any(excluded in rel_path for excluded in EXCLUDED_FAKE_SUBSETS)


def build_on_disk_fake_frame_paths(datarootpath, fake_subsets=None):
    if fake_subsets is None:
        fake_subsets = [
            "Deepfakes",
            "Face2Face",
            "FaceSwap",
            "NeuralTextures",
            "FaceShifter",
        ]
    frame_paths = set()
    for filepath in get_filepaths(datarootpath, exts=[".png", ".jpg"]):
        rel_path = normalize_rel_frame_path(os.path.relpath(filepath, datarootpath))
        if "masks" in rel_path:
            continue
        if "youtube" in rel_path or "actors" in rel_path:
            continue
        if is_excluded_fake_path(rel_path):
            continue
        if not any(subset in rel_path for subset in fake_subsets):
            continue
        frame_paths.add(rel_path)
    return frame_paths


def build_concept_fake_frame_paths(concept_json_file):
    frame_paths = set()
    for entry in load_json(concept_json_file):
        video = normalize_rel_frame_path(entry["ff_video"])
        if is_excluded_fake_path(video):
            continue
        for frame in entry.get("frames", []):
            frame_paths.add(
                normalize_rel_frame_path(os.path.join(video, f"{frame}.png"))
            )
    return frame_paths


def build_valid_fake_frame_paths(
    datarootpath, concept_json_file, split_names, fake_subsets=None
):
    """Fake frames must be in concept_labels.json, on disk, and belong to a train/test split video."""
    on_disk = build_on_disk_fake_frame_paths(datarootpath, fake_subsets=fake_subsets)
    in_concepts = build_concept_fake_frame_paths(concept_json_file)
    candidate = on_disk & in_concepts

    split_ids = set()
    for split in split_names:
        split_ids.update(get_ff_split_ids(os.path.join(datarootpath, f"{split}.json")))

    return {fp for fp in candidate if get_ff_filepath_video_id(fp) in split_ids}


def get_filepath_to_concept_indices(
    datarootpath,
    concept_json_label,
    min_appearances,
    train_splits,
    test_splits,
    fake_subsets=None,
):
    valid_fake_frames = build_valid_fake_frame_paths(
        datarootpath,
        concept_json_label,
        split_names=train_splits + test_splits,
        fake_subsets=fake_subsets,
    )
    filepath_to_concept_id, concept_name_to_id = get_concept_per_filepath(
        concept_json_file=concept_json_label,
        valid_fake_frames=valid_fake_frames,
    )

    train_split_ids = list(
        chain.from_iterable(
            [
                get_ff_split_ids(os.path.join(datarootpath, each + ".json"))
                for each in train_splits
            ]
        )
    )
    test_split_ids = list(
        chain.from_iterable(
            [
                get_ff_split_ids(os.path.join(datarootpath, each + ".json"))
                for each in test_splits
            ]
        )
    )

    train_counts = Counter()
    test_counts = Counter()

    for k, concept_indices in filepath_to_concept_id.items():
        fp_id = get_ff_filepath_video_id(k)

        if fp_id in train_split_ids:
            train_counts.update(concept_indices)
        if fp_id in test_split_ids:
            test_counts.update(concept_indices)

    valid_ids = {
        idx
        for idx in concept_name_to_id.values()
        if train_counts[idx] >= min_appearances and test_counts[idx] >= min_appearances
    }

    valid_concepts = [
        name for name, idx in concept_name_to_id.items() if idx in valid_ids
    ]

    fresh_concept_name_to_id = {
        name: idx for idx, name in enumerate(sorted(valid_concepts))
    }

    old_to_new_id = {
        concept_name_to_id[name]: fresh_concept_name_to_id[name]
        for name in valid_concepts
    }

    fresh_filepath_to_concept_id = {}

    for fp, old_indices in filepath_to_concept_id.items():
        new_indices = [
            old_to_new_id[old_idx]
            for old_idx in old_indices
            if old_idx in old_to_new_id
        ]

        if new_indices:
            fresh_filepath_to_concept_id[fp] = new_indices

    return fresh_filepath_to_concept_id, fresh_concept_name_to_id


def ordered_concept_names(concept_name_to_id: dict) -> list:
    """Return concept names in dataset label order (index 0 .. K-1)."""
    return [
        name for name, _ in sorted(concept_name_to_id.items(), key=lambda item: item[1])
    ]


def filter_concept_region_matrix(
    concept_region_matrix,
    matrix_concept_names,
    selected_concept_names,
):
    """Keep only rows for selected concepts, in the same order as selected_concept_names."""
    name_to_row = {name: idx for idx, name in enumerate(matrix_concept_names)}
    missing = [name for name in selected_concept_names if name not in name_to_row]
    if missing:
        raise ValueError(
            "Selected concepts are missing from concept_region_matrix: "
            + ", ".join(missing)
        )

    row_indices = [name_to_row[name] for name in selected_concept_names]
    return concept_region_matrix[row_indices]


def load_filtered_concept_region_matrix(
    concept_region_matrix_path: str,
    selected_concept_names: list,
):
    """Load the full concept-region matrix and subset rows for selected concepts."""
    concept_region_matrix, matrix_concept_names, _ = load_concept_region_assets(
        concept_region_matrix_path
    )
    return filter_concept_region_matrix(
        concept_region_matrix,
        matrix_concept_names,
        selected_concept_names,
    )
