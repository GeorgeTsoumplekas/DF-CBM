from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from metrics.metrics import (
    compute_binary_metrics,
    compute_concept_report,
    compute_video_level_class_metrics,
    compute_video_level_concept_report,
)


def _unpack_batch(batch) -> Tuple:
    if len(batch) == 4:
        imgs, labels, concept_labels, frame_paths = batch
        return imgs, labels, concept_labels, list(frame_paths)
    imgs, labels, concept_labels = batch
    return imgs, labels, concept_labels, None


def run_cbm_evaluation(
    model: torch.nn.Module,
    feature_encoder: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    *,
    concept_names: Optional[Sequence[str]] = None,
    desc: str = "[Evaluating]",
) -> Dict:
    """Run frame- and video-level class/concept evaluation on a dataloader."""
    model.eval()
    feature_encoder.eval()

    all_labels: List[np.ndarray] = []
    all_preds: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []
    all_concept_labels: List[np.ndarray] = []
    all_concept_logits: List[np.ndarray] = []
    all_frame_paths: List[str] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            imgs, labels, concept_labels, frame_paths = _unpack_batch(batch)
            imgs = imgs.to(device)

            result = feature_encoder.encode(imgs)
            tokens = result.get("clip_patch_tokens")
            masks = result.get("region_masks")

            model_out = model(tokens, masks)
            if isinstance(model_out, tuple):
                class_logits, concept_logits = model_out
            else:
                class_logits = model_out
                concept_logits = tokens.new_zeros(
                    tokens.shape[0], concept_labels.shape[-1]
                )

            probs = torch.softmax(class_logits, dim=1)
            preds = torch.argmax(class_logits, dim=1)

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs[:, 1].cpu().numpy())
            all_concept_labels.append(concept_labels.cpu().numpy())
            all_concept_logits.append(concept_logits.cpu().numpy())
            if frame_paths is not None:
                all_frame_paths.extend(frame_paths)

    labels = np.concatenate(all_labels)
    preds = np.concatenate(all_preds)
    probs = np.concatenate(all_probs)
    concept_labels = np.concatenate(all_concept_labels)
    concept_logits = np.concatenate(all_concept_logits)

    frame_class_metrics = compute_binary_metrics(labels, preds, probs)
    frame_concept_report = compute_concept_report(
        concept_labels, concept_logits, concept_names=concept_names
    )

    video_class_metrics = None
    video_concept_report = None
    if all_frame_paths and len(all_frame_paths) == len(labels):
        video_class_metrics = compute_video_level_class_metrics(
            labels, probs, all_frame_paths
        )
        video_concept_report = compute_video_level_concept_report(
            concept_labels, concept_logits, all_frame_paths, concept_names=concept_names
        )

    return {
        "frame_class_metrics": frame_class_metrics,
        "frame_concept_report": frame_concept_report,
        "video_class_metrics": video_class_metrics,
        "video_concept_report": video_concept_report,
    }
