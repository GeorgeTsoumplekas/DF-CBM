"""Helpers for stable PyTorch DataLoader usage on large datasets."""

from __future__ import annotations

import resource

import torch
from torch.utils.data import DataLoader, Dataset

_DEFAULT_MIN_OPEN_FILES = 65_536


def configure_torch_dataloader_env(
    *, min_open_files: int = _DEFAULT_MIN_OPEN_FILES
) -> None:
    """Prepare the process for multiprocessing DataLoaders on large datasets."""
    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
    except RuntimeError:
        # Strategy can only be set once per process.
        pass

    _raise_open_file_limit(min_open_files)


def _raise_open_file_limit(min_open_files: int) -> None:
    """Raise the soft RLIMIT_NOFILE when the OS allows it."""
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    except (ValueError, resource.error):
        return

    target_soft = min(max(min_open_files, soft), hard)
    if target_soft <= soft:
        return

    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target_soft, hard))
        print(
            f"[dataloader] raised open-file limit: {soft} -> {target_soft} "
            f"(hard limit {hard})"
        )
    except (ValueError, resource.error) as exc:
        print(
            f"[dataloader] could not raise open-file limit above {soft} "
            f"(hard limit {hard}): {exc}. "
            "If evaluation fails, run `ulimit -n 65536` in your shell."
        )


def build_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 4,
) -> DataLoader:
    """Create a DataLoader with conservative multiprocessing defaults."""
    workers = max(int(num_workers), 0)
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": False,
    }
    if workers > 0:
        kwargs["persistent_workers"] = False
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)
