import os
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any

_TRAINING_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _TRAINING_ROOT.parent
_DEFAULT_CONFIG = _TRAINING_ROOT / "configs" / "config.yaml"

_REPO_PATH_FIELDS = (
    ("dataset_config", "rootpath"),
    ("dataset_config", "concept_label_filepath"),
    ("feature_encoder_config", "segface_model_path"),
    ("evaluation_config", "dataset_json_folder"),
    ("evaluation_config", "dataset_root_rgb"),
)


def _resolve_repo_path(value: str) -> str:
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    return str((_REPO_ROOT / path).resolve())


def resolve_repo_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve config paths relative to the repository root."""
    for section, key in _REPO_PATH_FIELDS:
        block = config.get(section)
        if isinstance(block, dict) and block.get(key):
            block[key] = _resolve_repo_path(block[key])

    params = config.get("model_config", {}).get("params", {})
    matrix_path = (
        params.get("concept_region_matrix_path") if isinstance(params, dict) else None
    )
    if matrix_path:
        params["concept_region_matrix_path"] = _resolve_repo_path(matrix_path)
    return config


def load_and_parse_config(config_path: str | None = None):
    """Reads the nested YAML directly without flattening or rewriting keys."""
    parser = argparse.ArgumentParser(description="Train DF-CBM")
    parser.add_argument(
        "--config",
        type=str,
        default=config_path or str(_DEFAULT_CONFIG),
        help="Path to the configuration YAML file",
    )
    args, _ = parser.parse_known_args()
    resolved_path = config_path or args.config

    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"Configuration file not found at: {resolved_path}")

    with open(resolved_path, "r") as f:
        raw_cfg = yaml.safe_load(f)

    return resolve_repo_paths(raw_cfg)


def parse_segclip_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitizes and prepares the configuration dictionary for legacy functions.
    Injects required fallback keys to prevent KeyError crashes in frozen code
    while keeping your main pipeline clean.
    """
    # 1. Extract the feature encoder block safely
    enc_cfg = config.get("feature_encoder_config", {})

    # If the config passed is already the flat encoder block itself, use it directly
    if not enc_cfg and "clip_model" in config:
        enc_cfg = config

    # 2. Extract base resolution for the fallback baseline
    resolution_val = enc_cfg.get("resolution", 224)

    # 3. Force-inject the absolute bare minimum bracket keys the legacy code demands
    enc_cfg["resolution"] = resolution_val
    enc_cfg["segface_model_path"] = enc_cfg.get("segface_model_path")

    # 4. Explicitly map sub-resolutions so the .get() fallbacks short-circuit immediately
    enc_cfg["clip_input_resolution"] = enc_cfg.get(
        "clip_input_resolution", resolution_val
    )
    enc_cfg["segface_input_resolution"] = enc_cfg.get(
        "segface_input_resolution", resolution_val
    )

    return config
