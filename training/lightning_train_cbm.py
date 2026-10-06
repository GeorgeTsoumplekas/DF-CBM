import sys
from pathlib import Path

# Ensure local packages (`metrics`, `datasets`, `config`, etc.) resolve here first.
_TRAINING_ROOT = Path(__file__).resolve().parent
_training_root = str(_TRAINING_ROOT)
if _training_root not in sys.path:
    sys.path.insert(0, _training_root)

# Repo-root `network/` package lives outside `training/`.
_REPO_ROOT = _TRAINING_ROOT.parent
_repo_root = str(_REPO_ROOT)
if _repo_root not in sys.path:
    sys.path.insert(1, _repo_root)


from config.parse_config import load_and_parse_config, parse_segclip_config
from pl_trainers.join_cbm_trainer import LightningTrainerRunner
from pytorch_lightning import seed_everything
import torch
import os


def enforce_determinism(seed: int = 42):
    """Enforces strict determinism across Python, NumPy, PyTorch, and CUDA."""
    seed_everything(seed, workers=True)
    torch.use_deterministic_algorithms(False)

    # CUBLAS workspace environment variable is required for determinism in certain PyTorch versions
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # or ":16:8"
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def main():

    GLOBAL_SEED = 42
    enforce_determinism(seed=GLOBAL_SEED)

    # Load and parse configurations
    cfg = load_and_parse_config()
    cfg = parse_segclip_config(cfg)

    # Isolate relevant sub-configurations cleanly
    runner_configs = {
        "dataset_config": cfg.get("dataset_config", {}),
        "feature_encoder_config": cfg.get("feature_encoder_config", {}),
        "model_config": cfg.get("model_config", {}),
        "training_config": cfg.get("training_config", {}),
    }

    # Pass the isolated configs explicitly into your orchestrator
    runner = LightningTrainerRunner(**runner_configs)

    # Run execution lifecycle
    runner.fit()


if __name__ == "__main__":
    main()
