import json
import warnings
from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml
from datasets.data_module import FFDataModule
from models_.lightning_module import SegClipCBMLightningModule
from pytorch_lightning.callbacks import ModelCheckpoint


class LightningTrainerRunner:
    def __init__(
        self,
        dataset_config: dict,
        feature_encoder_config: dict,
        model_config: dict,
        training_config: dict,
    ):
        """
        Main runner class that dynamically pairs configuration segments
        with PyTorch Lightning components.
        """
        self.ds_cfg = dataset_config
        self.enc_cfg = feature_encoder_config
        self.model_cfg = model_config.copy()
        self.train_cfg = training_config

        self.dm = None
        self.system = None
        self.trainer = None
        self.output_dir = None

    def setup_pipeline(self):
        self.dm = FFDataModule(
            ds_cfg=self.ds_cfg,
            batch_size=self.train_cfg.get("batch_size", 32),
        )
        self.dm.setup()

        self.model_cfg["params"]["num_concepts"] = self.dm.num_concepts
        self.model_cfg["params"]["concept_names"] = self.dm.concept_names

        if (
            hasattr(self.dm, "dataset_pos_weights")
            and self.dm.dataset_pos_weights is not None
        ):
            self.train_cfg["losses"]["concept_loss"]["params"]["default_pos_weight"] = (
                self.dm.dataset_pos_weights
            )

        hparams = {
            "model_cfg": self.model_cfg,
            "enc_cfg": self.enc_cfg,
            **self.train_cfg,
        }
        self.system = SegClipCBMLightningModule(hparams)

    def _resolve_output_dir(self) -> Path:
        base_dir = Path(self.train_cfg.get("output_dir", "runs/cbm"))
        run_name = self.train_cfg.get("run_name")
        if not run_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = self.model_cfg.get("name", "model")
            run_name = f"{model_name}_{timestamp}"
        output_dir = base_dir / run_name
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _save_run_config(self, output_dir: Path):
        run_config = {
            "dataset_config": self.ds_cfg,
            "feature_encoder_config": self.enc_cfg,
            "model_config": self.model_cfg,
            "training_config": self.train_cfg,
        }
        with open(output_dir / "config.yaml", "w", encoding="utf-8") as handle:
            yaml.safe_dump(run_config, handle, sort_keys=False)

        with open(output_dir / "concept_names.json", "w", encoding="utf-8") as handle:
            json.dump(self.dm.concept_names, handle, indent=2)

    def _build_checkpoint_callbacks(self, checkpoint_dir: Path):
        checkpoint_cfg = self.train_cfg.get("checkpoint", {})
        callbacks = []

        if checkpoint_cfg.get("save_best_frame_auc", True):
            callbacks.append(
                ModelCheckpoint(
                    dirpath=str(checkpoint_dir),
                    filename="best_frame_auc",
                    monitor="test_class_auc_frame",
                    mode="max",
                    save_top_k=1,
                    save_last=False,
                )
            )

        if checkpoint_cfg.get("save_best_video_auc", True):
            callbacks.append(
                ModelCheckpoint(
                    dirpath=str(checkpoint_dir),
                    filename="best_video_auc",
                    monitor="test_class_auc_video",
                    mode="max",
                    save_top_k=1,
                    save_last=False,
                )
            )

        return callbacks

    def _parse_trainer_settings(self):
        device_str = str(self.train_cfg.get("device", "cuda:0")).lower()
        num_gpus = int(self.train_cfg.get("num_gpus", 1))
        gpu_ids = self.train_cfg.get("gpu_ids")
        strategy = self.train_cfg.get("strategy")

        wants_gpu = device_str in {"cuda", "gpu"} or "cuda" in device_str
        if not wants_gpu:
            return "cpu", "auto", "auto"

        if not torch.cuda.is_available():
            warnings.warn(
                f"Requested GPU training (device={device_str!r}, num_gpus={num_gpus}) "
                "but CUDA is not available. Falling back to CPU.",
                stacklevel=2,
            )
            return "cpu", "auto", "auto"

        if gpu_ids is not None:
            devices = [int(device_id) for device_id in gpu_ids]
            if num_gpus != len(devices):
                raise ValueError(
                    f"num_gpus ({num_gpus}) must match gpu_ids length ({len(devices)})."
                )
        elif num_gpus > 1:
            devices = list(range(num_gpus))
        elif ":" in device_str:
            devices = [int(device_str.rsplit(":", maxsplit=1)[-1])]
        else:
            devices = [0]

        available = torch.cuda.device_count()
        for device_id in devices:
            if device_id < 0 or device_id >= available:
                raise ValueError(
                    f"Invalid gpu id {device_id}; this machine exposes {available} GPU(s)."
                )

        if len(devices) > 1:
            resolved_strategy = strategy or "ddp"
        else:
            resolved_strategy = strategy or "auto"

        return "gpu", devices, resolved_strategy

    def fit(self):
        if not self.dm or not self.system:
            self.setup_pipeline()

        self.output_dir = self._resolve_output_dir()
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._save_run_config(self.output_dir)

        accelerator, devices, strategy = self._parse_trainer_settings()
        callbacks = self._build_checkpoint_callbacks(checkpoint_dir)

        self.trainer = pl.Trainer(
            max_epochs=self.train_cfg.get("num_epochs", 20),
            accelerator=accelerator,
            devices=devices,
            strategy=strategy,
            default_root_dir=str(self.output_dir),
            callbacks=callbacks,
            enable_checkpointing=bool(callbacks),
            logger=True,
        )

        self.trainer.fit(self.system, datamodule=self.dm)
