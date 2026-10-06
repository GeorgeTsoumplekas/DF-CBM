from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import torch
from torch.utils.data import DataLoader

from datasets.data_module import FFDataModule
from datasets.json_benchmark import build_json_benchmark_dataloader
from evaluation.cbm_evaluator import run_cbm_evaluation
from logger.helper_logger import log_evaluation_report
from models_.lightning_module import SegClipCBMLightningModule


def resolve_device(device_cfg: str) -> torch.device:
    device_str = str(device_cfg).lower()
    if device_str in {"cuda", "gpu"} or "cuda" in device_str:
        if torch.cuda.is_available():
            if ":" in device_str:
                return torch.device(device_str)
            return torch.device("cuda:0")
    return torch.device("cpu")


def load_eval_module(
    checkpoint_path: str, device: torch.device
) -> SegClipCBMLightningModule:
    module = SegClipCBMLightningModule.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
    )
    module.to(device)
    module.eval()
    return module


def _normalize_dataset_entry(entry: Union[str, dict]) -> dict:
    if isinstance(entry, str):
        return {"name": entry, "json_file": f"{entry}.json"}
    normalized = dict(entry)
    name = normalized.get("name") or Path(normalized.get("json_file", "")).stem
    normalized["name"] = name
    if "json_file" not in normalized:
        normalized["json_file"] = f"{name}.json"
    return normalized


def _require_ffpp_dataset_config(ds_cfg: dict) -> None:
    missing = [
        key for key in ("rootpath", "concept_label_filepath") if not ds_cfg.get(key)
    ]
    if missing:
        raise ValueError(
            "FF++ evaluation requires dataset_config fields: "
            + ", ".join(missing)
            + ". Use configs/config.yaml, add them to your eval config, "
            "or run with --target cross for JSON benchmark datasets only."
        )


def evaluate_ffpp_splits(
    module: SegClipCBMLightningModule,
    cfg: dict,
    *,
    splits: Sequence[Union[str, Sequence[str]]],
    device: torch.device,
) -> Dict[str, dict]:
    ds_cfg = cfg.get("dataset_config", {})
    train_cfg = cfg.get("training_config", {})
    _require_ffpp_dataset_config(ds_cfg)

    dm = FFDataModule(ds_cfg=ds_cfg, batch_size=train_cfg.get("batch_size", 32))

    results = {}
    for split_name in splits:
        split_key = split_name if isinstance(split_name, str) else "_".join(split_name)
        dataloader, _ = dm.get_split_dataloader(split_name)
        results[split_key] = _run_and_log(
            module=module,
            dataloader=dataloader,
            device=device,
            split_name=split_key,
        )
    return results


def evaluate_json_datasets(
    module: SegClipCBMLightningModule,
    eval_cfg: dict,
    *,
    device: torch.device,
    datasets: Optional[Sequence[Union[str, dict]]] = None,
) -> Dict[str, dict]:
    dataset_entries = datasets or eval_cfg.get("datasets", [])
    if not dataset_entries:
        raise ValueError("No JSON benchmark datasets configured for evaluation.")

    label_dict = eval_cfg.get("label_dict")
    if not label_dict:
        raise ValueError("evaluation_config.label_dict is required for JSON datasets.")

    json_folder = Path(eval_cfg["dataset_json_folder"])
    dataset_root_rgb = eval_cfg["dataset_root_rgb"]
    mode = eval_cfg.get("mode", "test")
    compression = eval_cfg.get("compression", "c23")
    skip_missing_files = bool(eval_cfg.get("skip_missing_files", True))
    batch_size = int(eval_cfg.get("batch_size", 32))
    num_workers = int(eval_cfg.get("num_workers", 4))
    resolution = int(eval_cfg.get("resolution", 224))
    num_concepts = int(
        eval_cfg.get("num_concepts")
        or getattr(module, "num_concepts", 0)
        or len(module.concept_names or [])
    )

    results = {}
    for entry in dataset_entries:
        dataset_cfg = _normalize_dataset_entry(entry)
        json_path = json_folder / dataset_cfg["json_file"]
        if not json_path.is_file():
            raise FileNotFoundError(f"Dataset JSON not found: {json_path}")

        dataloader, dataset = build_json_benchmark_dataloader(
            json_path=str(json_path),
            dataset_root_rgb=dataset_root_rgb,
            label_dict=label_dict,
            batch_size=batch_size,
            num_concepts=num_concepts,
            resolution=resolution,
            mode=dataset_cfg.get("mode", mode),
            compression=dataset_cfg.get("compression", compression),
            skip_missing_files=skip_missing_files,
            dataset_key=dataset_cfg.get("dataset_key"),
            num_workers=num_workers,
        )
        split_name = dataset_cfg["name"]
        results[split_name] = _run_and_log(
            module=module,
            dataloader=dataloader,
            device=device,
            split_name=split_name,
        )
    return results


def _run_and_log(
    *,
    module: SegClipCBMLightningModule,
    dataloader: DataLoader,
    device: torch.device,
    split_name: str,
) -> dict:
    eval_result = run_cbm_evaluation(
        model=module.model,
        feature_encoder=module.feature_encoder,
        dataloader=dataloader,
        device=device,
        concept_names=module.concept_names,
        desc=f"[Eval {split_name}]",
    )
    log_evaluation_report(
        class_m=eval_result["frame_class_metrics"],
        concept_report=eval_result["frame_concept_report"],
        epoch=0,
        train_loss=0.0,
        split_name=split_name,
        video_class_m=eval_result["video_class_metrics"],
        video_concept_report=eval_result["video_concept_report"],
    )
    return eval_result


def serialize_eval_results(results: Dict[str, dict]) -> dict:
    serializable = {}
    for split_key, eval_result in results.items():
        serializable[split_key] = {
            metric_group: {
                key: float(value) if hasattr(value, "item") else value
                for key, value in metrics.items()
            }
            if metric_group.endswith("_metrics")
            else {
                concept_key: {
                    metric_key: float(metric_value)
                    if hasattr(metric_value, "item")
                    else metric_value
                    for metric_key, metric_value in concept_metrics.items()
                }
                for concept_key, concept_metrics in metrics.items()
            }
            for metric_group, metrics in eval_result.items()
            if metrics is not None
        }
    return serializable


def write_eval_results_json(results: Dict[str, dict], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(serialize_eval_results(results), handle, indent=2)
