from utils.ioutils import get_filepaths
from utils.dsutils import read_rgb_uint8_square
from utils.dsutils import split_filter
from utils.concept_utils import (
    get_filepath_to_concept_indices,
    normalize_rel_frame_path,
)
from torch.utils.data import Dataset
import torch
import os


class FFWithConcepts(Dataset):
    def __init__(self, rootpath, concept_label_filepath, train, **kwargs):
        super().__init__()
        self.rootpath = rootpath
        self.concept_label_filepath = concept_label_filepath
        self.resolution = kwargs.get("resolution", 224)
        self.min_samples_per_concept = kwargs.get("min_samples_per_concept", 500)
        self.pos_weight_ignore_fake_negatives = kwargs.get(
            "pos_weight_ignore_fake_negatives", False
        )

        self._parse_splits_and_subsets(train, kwargs)

        real_paths, fake_paths = self._get_valid_filepaths()

        self._build_dataset(real_paths, fake_paths)

        self._print_summary()

    def _parse_splits_and_subsets(self, train, kwargs):
        train_splits = kwargs.get("train_splits", ["train"])
        test_splits = kwargs.get("test_splits", ["test"])
        explicit_splits = kwargs.get("splits")
        if explicit_splits is not None:
            self.splits = explicit_splits
        elif train is None:
            raise ValueError(
                "Either train=True/False or an explicit splits list must be provided."
            )
        else:
            self.splits = train_splits if train else test_splits
        if isinstance(self.splits, str):
            self.splits = [self.splits]

        fake_subsets = kwargs.get(
            "fake_subsets",
            ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter"],
        )
        self.fake_subsets = fake_subsets
        self.subsets = fake_subsets + ["youtube"]
        self.train_splits = train_splits
        self.test_splits = test_splits

    def _get_valid_filepaths(self, split=None):
        if split is None:
            split = self.splits
        img_paths = get_filepaths(self.rootpath, exts=[".png", ".jpg"])

        valid_paths = [
            p
            for p in img_paths
            if any(sub in p for sub in self.subsets) and "masks" not in p
        ]
        valid_paths = split_filter(self.rootpath, valid_paths, split)

        real_paths = [p for p in valid_paths if "youtube" in p]
        fake_paths = [p for p in valid_paths if "youtube" not in p]
        return real_paths, fake_paths

    def _build_dataset(self, real_paths, fake_paths):
        filepath_to_concept_id, concept_name_to_id = get_filepath_to_concept_indices(
            datarootpath=self.rootpath,
            concept_json_label=self.concept_label_filepath,
            min_appearances=self.min_samples_per_concept,
            train_splits=self.train_splits,
            test_splits=self.test_splits,
            fake_subsets=self.fake_subsets,
        )
        self.concept_name_to_id = concept_name_to_id

        fake_paths = [
            p
            for p in fake_paths
            if normalize_rel_frame_path(os.path.relpath(p, self.rootpath))
            in filepath_to_concept_id
        ]

        self.img_paths = real_paths + fake_paths
        self.labels = [0] * len(real_paths) + [1] * len(fake_paths)

        real_concepts = [torch.zeros(len(self.concept_name_to_id)) for _ in real_paths]
        fake_concepts = [
            self._create_multi_hot_label(
                filepath_to_concept_id[
                    normalize_rel_frame_path(os.path.relpath(p, self.rootpath))
                ]
            )
            for p in fake_paths
        ]
        self.concept_labels = real_concepts + fake_concepts

        self.samples = list(zip(self.img_paths, self.labels, self.concept_labels))

        self.pos_weights = self._calculate_dataset_pos_weights()

    def _create_multi_hot_label(self, indices):
        concept_label = torch.zeros(len(self.concept_name_to_id))
        concept_label[indices] += 1
        return concept_label

    def _print_summary(self):
        print(f"Total Real Samples: {self.labels.count(0)}")
        print(f"Total Fake Samples: {self.labels.count(1)}")

        concept_counts = torch.zeros(len(self.concept_name_to_id))
        for each in self.concept_labels:
            concept_counts += each

        torch.set_printoptions(sci_mode=False)
        for name, count in zip(self.concept_name_to_id, concept_counts):
            print(f"Concept '{name}': {count.long().item()}")

    def _calculate_dataset_pos_weights(self):
        """Calculates dataset-wide dynamic positive weights for class imbalance.

        By default, every zero concept label counts as a negative.
        When ``pos_weight_ignore_fake_negatives`` is True, zeros on fake samples
        are treated as untrusted and excluded from the negative count (only real
        samples contribute negatives).
        """
        # Shape: (num_concepts,)
        concept_counts = torch.zeros(len(self.concept_name_to_id))
        for label in self.concept_labels:
            concept_counts += label

        if self.pos_weight_ignore_fake_negatives:
            neg_counts = torch.zeros(len(self.concept_name_to_id))
            for sample_label, concept_label in zip(self.labels, self.concept_labels):
                if sample_label == 0:  # real: trusted negatives
                    neg_counts += 1.0 - concept_label
        else:
            total_samples = len(self.samples)
            neg_counts = total_samples - concept_counts

        # Safe division to prevent division by zero for unrepresented concepts
        pos_weights = neg_counts / (concept_counts + 1e-6)
        return pos_weights

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label, concept_labels = self.samples[index]
        img_rgb = read_rgb_uint8_square(img_path, self.resolution)
        return img_rgb, label, concept_labels, img_path
