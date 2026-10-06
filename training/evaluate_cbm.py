import sys
from pathlib import Path

_TRAINING_ROOT = Path(__file__).resolve().parent
_training_root = str(_TRAINING_ROOT)
if _training_root not in sys.path:
    sys.path.insert(0, _training_root)

_REPO_ROOT = _TRAINING_ROOT.parent
_repo_root = str(_REPO_ROOT)
if _repo_root not in sys.path:
    sys.path.insert(1, _repo_root)

import argparse

from config.parse_config import load_and_parse_config, parse_segclip_config
from utils.dataloader_utils import configure_torch_dataloader_env
from evaluation.eval_runner import (
    evaluate_ffpp_splits,
    evaluate_json_datasets,
    load_eval_module,
    resolve_device,
    write_eval_results_json,
)


def _parse_dataset_overrides(dataset_arg: str | None) -> list | None:
    if not dataset_arg:
        return None
    return [item.strip() for item in dataset_arg.split(",") if item.strip()]


def main():
    configure_torch_dataloader_env()

    parser = argparse.ArgumentParser(
        description="Evaluate a saved CBM Lightning checkpoint."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent / "configs" / "config.yaml"),
        help="Path to the configuration YAML file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a Lightning checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="ffpp",
        choices=["ffpp", "cross", "both"],
        help="Evaluation target: FaceForensics++ splits, JSON benchmark datasets, or both",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="test",
        help="Comma-separated FF++ split aliases for --target ffpp/both: train, test, all",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Optional comma-separated override of evaluation_config.datasets",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to write evaluation metrics as JSON",
    )
    args = parser.parse_args()

    cfg = load_and_parse_config(config_path=args.config)
    cfg = parse_segclip_config(cfg)

    train_cfg = cfg.get("training_config", {})
    eval_cfg = cfg.get("evaluation_config", {})
    device = resolve_device(train_cfg.get("device", eval_cfg.get("device", "cuda")))

    module = load_eval_module(args.checkpoint, device)
    results = {}

    if args.target in {"ffpp", "both"}:
        ds_cfg = cfg.get("dataset_config", {})
        split_aliases = [
            item.strip() for item in args.splits.split(",") if item.strip()
        ]
        resolved_splits = []
        for split_alias in split_aliases:
            if split_alias == "train":
                resolved_splits.extend(ds_cfg.get("train_splits", ["train"]))
            elif split_alias == "test":
                resolved_splits.extend(ds_cfg.get("test_splits", ["test"]))
            elif split_alias == "all":
                resolved_splits.extend(ds_cfg.get("train_splits", ["train"]))
                resolved_splits.extend(ds_cfg.get("test_splits", ["test"]))
            else:
                resolved_splits.append(split_alias)
        resolved_splits = list(dict.fromkeys(resolved_splits))
        results.update(
            evaluate_ffpp_splits(
                module,
                cfg,
                splits=resolved_splits,
                device=device,
            )
        )

    if args.target in {"cross", "both"}:
        dataset_override = _parse_dataset_overrides(args.datasets)
        results.update(
            evaluate_json_datasets(
                module,
                eval_cfg,
                device=device,
                datasets=dataset_override,
            )
        )

    if args.output_json:
        write_eval_results_json(results, args.output_json)
        print(f"Wrote evaluation metrics to {args.output_json}")


if __name__ == "__main__":
    main()
