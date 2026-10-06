from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import Dataset

from utils.dataloader_utils import build_dataloader
from utils.dsutils import read_rgb_uint8_square
from utils.json_dataset_loader import load_json_dataset_samples


class JsonBenchmarkDataset(Dataset):
    """Evaluation dataset backed by a single DeepfakeBench-style JSON file."""

    def __init__(
        self,
        *,
        json_path: str,
        dataset_root_rgb: str,
        label_dict: dict,
        num_concepts: int = 0,
        resolution: int = 224,
        mode: str = "test",
        compression: Optional[str] = "c23",
        skip_missing_files: bool = True,
        dataset_key: Optional[str] = None,
    ):
        super().__init__()
        self.resolution = resolution
        self.num_concepts = int(num_concepts)

        frame_paths, labels, dataset_key = load_json_dataset_samples(
            json_path=json_path,
            dataset_root_rgb=dataset_root_rgb,
            label_dict=label_dict,
            mode=mode,
            compression=compression,
            skip_missing_files=skip_missing_files,
            dataset_key=dataset_key,
        )
        self.dataset_key = dataset_key
        self.frame_paths = frame_paths
        self.labels = labels
        self.samples = list(zip(self.frame_paths, self.labels))

        real_count = sum(1 for label in self.labels if label == 0)
        fake_count = len(self.labels) - real_count
        print(
            f"[{self.dataset_key}] loaded {len(self.samples)} frames "
            f"(real={real_count}, fake={fake_count}, mode={mode!r})"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        img_path, label = self.samples[index]
        img_rgb = read_rgb_uint8_square(img_path, self.resolution)
        concept_labels = torch.zeros(self.num_concepts, dtype=torch.float32)
        return img_rgb, label, concept_labels, img_path


def build_json_benchmark_dataloader(
    *,
    json_path: str,
    dataset_root_rgb: str,
    label_dict: dict,
    batch_size: int,
    num_concepts: int,
    resolution: int = 224,
    mode: str = "test",
    compression: Optional[str] = "c23",
    skip_missing_files: bool = True,
    dataset_key: Optional[str] = None,
    num_workers: int = 4,
):
    dataset = JsonBenchmarkDataset(
        json_path=json_path,
        dataset_root_rgb=dataset_root_rgb,
        label_dict=label_dict,
        num_concepts=num_concepts,
        resolution=resolution,
        mode=mode,
        compression=compression,
        skip_missing_files=skip_missing_files,
        dataset_key=dataset_key,
    )
    dataloader = build_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return dataloader, dataset
