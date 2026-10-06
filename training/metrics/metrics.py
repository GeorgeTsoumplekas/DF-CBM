import numpy as np
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from utils.dsutils import video_id_from_frame_path


def compute_binary_metrics(y_true, y_pred, y_prob):
    """
    Computes a comprehensive set of classification metrics for a single target array.
    Safely handles edge cases like single-class presence in the validation batch.
    """
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    try:
        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_prob)
        else:
            auc = float("nan")
    except ValueError:
        auc = float("nan")

    return {
        "acc": acc,
        "bal_acc": bal_acc,
        "precision": prec,
        "recall": rec,
        "macro_f1": macro_f1,
        "auc": auc,
    }


def aggregate_frame_predictions_to_videos(y_pred, y_true, frame_paths):
    """Mean-pool predictions and max-pool labels to one row per video."""
    y_pred = np.asarray(y_pred, dtype=np.float32)
    y_true = np.asarray(y_true, dtype=np.float32)
    if y_pred.shape != y_true.shape:
        raise ValueError(
            f"Shape mismatch between predictions and labels: {y_pred.shape} vs {y_true.shape}."
        )
    if len(frame_paths) != y_pred.shape[0]:
        raise ValueError(
            f"Expected {y_pred.shape[0]} frame paths, got {len(frame_paths)}."
        )

    path_groups = defaultdict(list)
    for index, frame_path in enumerate(frame_paths):
        path_groups[str(frame_path)].append(index)

    frame_pred_rows = []
    frame_true_rows = []
    frame_video_ids = []
    for frame_path, indices in path_groups.items():
        index_array = np.asarray(indices, dtype=np.int64)
        frame_pred_rows.append(y_pred[index_array].mean(axis=0))
        frame_true_rows.append(y_true[index_array].max(axis=0))
        frame_video_ids.append(video_id_from_frame_path(frame_path))

    frame_pred = np.stack(frame_pred_rows, axis=0)
    frame_true = np.stack(frame_true_rows, axis=0)

    video_groups = defaultdict(list)
    for index, video_id in enumerate(frame_video_ids):
        video_groups[video_id].append(index)

    video_pred_rows = []
    video_true_rows = []
    for video_id in sorted(video_groups):
        index_array = np.asarray(video_groups[video_id], dtype=np.int64)
        video_pred_rows.append(frame_pred[index_array].mean(axis=0))
        video_true_rows.append(frame_true[index_array].max(axis=0))

    return np.stack(video_pred_rows, axis=0), np.stack(video_true_rows, axis=0)


def compute_concept_report(all_concept_labels, all_concept_logits, concept_names=None):
    """
    Slices evaluation data horizontally across each individual concept dimension.
    Returns:
        report: dict containing per-concept metric splits and their collective average 'mean'.
    """
    all_concept_probs = 1 / (1 + np.exp(-all_concept_logits))
    all_concept_preds = (all_concept_probs > 0.5).astype(int)

    num_concepts = all_concept_labels.shape[1]
    report = {}

    metric_keys = ["acc", "bal_acc", "precision", "recall", "macro_f1"]
    accumulators = {k: [] for k in metric_keys}
    auc_accumulator = []

    for i in range(num_concepts):
        c_labels = all_concept_labels[:, i]
        c_preds = all_concept_preds[:, i]
        c_probs = all_concept_probs[:, i]

        metrics = compute_binary_metrics(c_labels, c_preds, c_probs)
        if concept_names is not None and i < len(concept_names):
            key = concept_names[i]
        else:
            key = f"concept_{i}"
        report[key] = metrics

        for k in metric_keys:
            accumulators[k].append(metrics[k])
        if not np.isnan(metrics["auc"]):
            auc_accumulator.append(metrics["auc"])

    report["mean"] = {k: np.mean(v) for k, v in accumulators.items()}
    report["mean"]["auc"] = (
        np.mean(auc_accumulator) if auc_accumulator else float("nan")
    )

    return report


def compute_video_level_concept_report(
    all_concept_labels, all_concept_logits, frame_paths, concept_names=None
):
    """Per-concept metrics after mean-pooling predictions per video."""
    all_concept_probs = 1 / (1 + np.exp(-all_concept_logits))
    video_probs, video_labels = aggregate_frame_predictions_to_videos(
        all_concept_probs, all_concept_labels, frame_paths
    )
    video_logits = np.log(
        np.clip(video_probs, 1e-7, 1 - 1e-7) / np.clip(1 - video_probs, 1e-7, 1 - 1e-7)
    )
    return compute_concept_report(
        video_labels, video_logits, concept_names=concept_names
    )


def compute_video_level_class_metrics(labels, probs, frame_paths):
    """Binary class metrics after mean-pooling frame probabilities per video."""
    labels = np.asarray(labels)
    probs = np.asarray(probs, dtype=np.float32)
    video_probs, video_labels = aggregate_frame_predictions_to_videos(
        probs.reshape(-1, 1), labels.reshape(-1, 1), frame_paths
    )
    video_probs = video_probs[:, 0]
    video_labels = video_labels[:, 0]
    video_preds = (video_probs > 0.5).astype(int)
    return compute_binary_metrics(video_labels, video_preds, video_probs)
