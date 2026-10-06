"""Load frame paths and labels from DeepfakeBench-style dataset JSON files."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from utils.ioutils import load_json

COMPRESSION_NESTED_DATASETS = {
    "FF-DF",
    "FF-F2F",
    "FF-FS",
    "FF-NT",
    "FaceForensics++",
    "DeepFakeDetection",
    "FaceShifter",
}


def resolve_frame_path(file_path: str, dataset_root_rgb: str) -> str:
    """Resolve a JSON frame path to an on-disk RGB image path."""
    if not file_path:
        return file_path

    normalized = os.path.normpath(str(file_path).replace("\\", "/"))
    if os.path.exists(normalized):
        return normalized

    root = os.path.normpath(dataset_root_rgb)
    joined = os.path.normpath(os.path.join(root, normalized))
    return joined


def _resolve_mode_block(label_block: dict, mode: str) -> Tuple[str, dict]:
    if mode in label_block:
        return mode, label_block[mode]
    for fallback in ("test", "val", "train"):
        if fallback in label_block:
            return fallback, label_block[fallback]
    raise KeyError(
        f"No split block found for mode={mode!r}. Available: {list(label_block.keys())}"
    )


def _resolve_split_block(
    split_block: dict,
    *,
    dataset_name: str,
    compression: Optional[str],
) -> dict:
    if dataset_name in COMPRESSION_NESTED_DATASETS:
        if compression is None:
            raise ValueError(
                f"Dataset {dataset_name!r} requires compression (e.g. c23) in evaluation config."
            )
        if compression not in split_block:
            raise KeyError(
                f"Compression {compression!r} not found for dataset {dataset_name!r}. "
                f"Available: {list(split_block.keys())}"
            )
        return split_block[compression]
    return split_block


def load_json_dataset_samples(
    *,
    json_path: str,
    dataset_root_rgb: str,
    label_dict: Dict[str, int],
    mode: str = "test",
    compression: Optional[str] = "c23",
    skip_missing_files: bool = True,
    dataset_key: Optional[str] = None,
) -> Tuple[List[str], List[int], str]:
    """Return frame paths, binary labels, and resolved dataset key."""
    raw = load_json(json_path)
    if dataset_key is None:
        dataset_key = Path(json_path).stem
    if dataset_key not in raw:
        if len(raw) == 1:
            dataset_key = next(iter(raw))
        else:
            raise KeyError(
                f"Dataset key {dataset_key!r} not found in {json_path}. "
                f"Available keys: {list(raw.keys())}"
            )

    dataset_info = raw[dataset_key]
    frame_paths: List[str] = []
    labels: List[int] = []
    missing = 0

    for _subset_name, subset_block in dataset_info.items():
        resolved_mode, mode_block = _resolve_mode_block(subset_block, mode)
        video_block = _resolve_split_block(
            mode_block,
            dataset_name=dataset_key,
            compression=compression,
        )

        for _video_name, video_info in video_block.items():
            label_name = video_info["label"]
            if label_name not in label_dict:
                raise ValueError(
                    f"Label {label_name!r} missing from evaluation label_dict "
                    f"while loading {json_path}."
                )
            label = int(label_dict[label_name])
            for frame_path in video_info.get("frames", []):
                resolved = resolve_frame_path(frame_path, dataset_root_rgb)
                if skip_missing_files and not os.path.exists(resolved):
                    missing += 1
                    continue
                frame_paths.append(resolved)
                labels.append(label)

    if not frame_paths:
        raise ValueError(
            f"No frames loaded from {json_path} (mode={mode!r}, missing_files={missing})."
        )

    if missing:
        print(
            f"[{dataset_key}] skipped {missing} missing frame paths "
            f"(loaded {len(frame_paths)} frames, mode={resolved_mode!r})."
        )

    return frame_paths, labels, dataset_key
