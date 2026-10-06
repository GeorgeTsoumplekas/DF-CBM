from pathlib import Path

import pytorch_lightning as pl
import torch
import numpy as np

from segclip.preprocess import ConceptFeatureEncoder
from metrics.metrics import (
    compute_binary_metrics,
    compute_concept_report,
    compute_video_level_class_metrics,
    compute_video_level_concept_report,
)
from logger.helper_logger import log_evaluation_report

from models_.registry import get_model_class
from losses import get_loss_criterion

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SEGFACE_WEIGHT = _REPO_ROOT / "weights" / "segface_celeba_swin_base_224.pt"
_CONCEPT_MATRIX = _REPO_ROOT / "data" / "concept_region_mapping_matrix.csv"


def _stored_path(stored: str) -> Path:
    path = Path(str(stored))
    if path.is_absolute():
        return path
    return _REPO_ROOT / path


def _stored_in_repo(stored: str) -> bool:
    try:
        _stored_path(stored).absolute().relative_to(_REPO_ROOT.resolve())
    except ValueError:
        return False
    return True


def _bind_asset(container, key: str, repo_file: Path, legacy_names: set[str]) -> None:
    """Point a checkpoint path at the copy of that file in this repository."""
    stored = container[key]
    if not stored:
        container[key] = str(repo_file)
        return
    resolved = _stored_path(str(stored))
    legacy = Path(str(stored)).name in legacy_names
    if repo_file.is_file() and (
        not resolved.is_file() or legacy or not _stored_in_repo(str(stored))
    ):
        container[key] = str(repo_file)
    elif not resolved.is_file():
        container[key] = str(repo_file)


def bind_release_checkpoint_paths(hparams):
    """Rewrite SegFace and concept-matrix paths saved in the released checkpoint.

    ``best_video_auc.ckpt`` stores ``models/segface_224.pt`` and an absolute
    concept-region matrix path from the training machine. Evaluation loads
    those strings back before the state dict, so map them onto
    ``weights/segface_celeba_swin_base_224.pt`` and
    ``data/concept_region_mapping_matrix.csv``.
    """
    enc_cfg = hparams["enc_cfg"]
    _bind_asset(enc_cfg, "segface_model_path", _SEGFACE_WEIGHT, {"segface_224.pt"})
    params = hparams["model_cfg"]["params"]
    _bind_asset(
        params,
        "concept_region_matrix_path",
        _CONCEPT_MATRIX,
        {"concept_region_mapping_matrix_v2.csv"},
    )
    return hparams


class SegClipCBMLightningModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        hparams = bind_release_checkpoint_paths(hparams)
        self.save_hyperparameters(hparams)

        self.feature_encoder = ConceptFeatureEncoder(
            config=self.hparams.enc_cfg, device=self.device
        )

        model_cfg = self.hparams.model_cfg
        model_class = get_model_class(model_cfg.get("name"))
        self.model = model_class(**model_cfg.get("params", {}))

        model_params = model_cfg.get("params", {})
        self.concept_names = model_params.get("concept_names")
        self.num_concepts = model_params.get("num_concepts", 0)

        loss_cfg = self.hparams.get("losses", {})

        cls_loss_info = loss_cfg.get("class_loss", {"name", "CrossEntropyLoss"})
        cls_loss_class = get_loss_criterion(cls_loss_info.get("name"))
        self.criterion_class = cls_loss_class(**cls_loss_info.get("params", {}))

        cpt_loss_info = loss_cfg.get("concept_loss", {"name": "BCEWithLogitsLoss"})
        cpt_loss_class = get_loss_criterion(cpt_loss_info.get("name"))
        self.criterion_concept = cpt_loss_class(**cpt_loss_info.get("params", {}))

        self.validation_step_outputs = []

    def setup(self, _stage=None):
        self.feature_encoder.device = self.device

    def forward(self, tokens, masks):
        out = self.model(tokens, masks)
        if isinstance(out, tuple):
            return out
        concept_logits = tokens.new_zeros(tokens.shape[0], self.num_concepts)
        return out, concept_logits

    def _unpack_batch(self, batch):
        if len(batch) == 4:
            imgs, labels, concept_labels, frame_paths = batch
            return imgs, labels, concept_labels, list(frame_paths)
        imgs, labels, concept_labels = batch
        return imgs, labels, concept_labels, None

    def _shared_step(self, batch):
        imgs, labels, concept_labels, frame_paths = self._unpack_batch(batch)

        with torch.no_grad():
            result = self.feature_encoder.encode(imgs)
            tokens = result.get("clip_patch_tokens")
            masks = result.get("region_masks")

        class_logits, concept_logits = self(tokens, masks)
        return class_logits, concept_logits, labels, concept_labels, frame_paths

    def training_step(self, batch, _batch_idx):
        class_logits, concept_logits, labels, concept_labels, _ = self._shared_step(
            batch
        )

        is_real_flag = (labels == 0).float()

        loss_class = self.criterion_class(class_logits, labels)
        loss_concept = self.criterion_concept(
            concept_logits, concept_labels.float(), is_real_sample=is_real_flag
        )
        total_loss = loss_class + (self.hparams.concept_loss_weight * loss_concept)

        self.log(
            "train_loss",
            total_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log("train_loss_class", loss_class, prog_bar=True, sync_dist=True)
        self.log("train_loss_concept", loss_concept, prog_bar=True, sync_dist=True)

        return total_loss

    def validation_step(self, batch, _batch_idx):
        class_logits, concept_logits, labels, concept_labels, frame_paths = (
            self._shared_step(batch)
        )

        probs = torch.softmax(class_logits, dim=1)
        preds = torch.argmax(class_logits, dim=1)

        output = {
            "labels": labels.cpu().numpy(),
            "preds": preds.cpu().numpy(),
            "probs": probs[:, 1].cpu().numpy(),
            "concept_labels": concept_labels.cpu().numpy(),
            "concept_logits": concept_logits.cpu().numpy(),
            "frame_paths": frame_paths,
        }
        self.validation_step_outputs.append(output)
        return output

    def _gather_validation_outputs(self):
        outputs = list(self.validation_step_outputs)
        if self.trainer.world_size <= 1:
            return outputs

        gathered = [None] * self.trainer.world_size
        torch.distributed.all_gather_object(gathered, outputs)
        return [step for rank_outputs in gathered for step in rank_outputs]

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return

        step_outputs = self._gather_validation_outputs()
        self.validation_step_outputs.clear()

        labels = np.concatenate([x["labels"] for x in step_outputs])
        preds = np.concatenate([x["preds"] for x in step_outputs])
        probs = np.concatenate([x["probs"] for x in step_outputs])
        concept_labels = np.concatenate([x["concept_labels"] for x in step_outputs])
        concept_logits = np.concatenate([x["concept_logits"] for x in step_outputs])
        frame_paths = [path for x in step_outputs for path in (x["frame_paths"] or [])]

        class_m = compute_binary_metrics(labels, preds, probs)
        concept_report = compute_concept_report(
            concept_labels, concept_logits, concept_names=self.concept_names
        )

        video_class_m = None
        video_concept_report = None
        if frame_paths and len(frame_paths) == len(labels):
            video_class_m = compute_video_level_class_metrics(
                labels, probs, frame_paths
            )
            video_concept_report = compute_video_level_concept_report(
                concept_labels,
                concept_logits,
                frame_paths,
                concept_names=self.concept_names,
            )

        self.log(
            "test_class_auc_frame",
            class_m["auc"] if not np.isnan(class_m["auc"]) else 0.0,
            prog_bar=True,
            sync_dist=False,
        )
        if video_class_m is not None:
            self.log(
                "test_class_auc_video",
                video_class_m["auc"] if not np.isnan(video_class_m["auc"]) else 0.0,
                prog_bar=True,
                sync_dist=False,
            )
            if video_concept_report is not None:
                mean_video_concept_auc = video_concept_report["mean"]["auc"]
                self.log(
                    "test_concept_auc_video_mean",
                    mean_video_concept_auc
                    if not np.isnan(mean_video_concept_auc)
                    else 0.0,
                    sync_dist=False,
                )

        if self.trainer.global_rank != 0:
            return

        avg_train_loss = self.trainer.callback_metrics.get("train_loss_epoch", 0.0)
        log_evaluation_report(
            class_m=class_m,
            concept_report=concept_report,
            epoch=self.current_epoch + 1,
            train_loss=avg_train_loss,
            logger=None,
            split_name="Test",
            video_class_m=video_class_m,
            video_concept_report=video_concept_report,
        )

    def configure_optimizers(self):
        opt_cfg = self.hparams.get("optimizer", {})
        opt_name = opt_cfg.get("name", "AdamW")
        opt_params = opt_cfg.get("params", {})

        optimizer_class = getattr(torch.optim, opt_name)
        optimizer = optimizer_class(
            self.parameters(),
            lr=self.hparams.learning_rate,
            **opt_params,
        )

        sched_cfg = self.hparams.get("lr_scheduler")
        if not sched_cfg:
            return optimizer

        sched_name = sched_cfg.get("name", "CosineAnnealingLR")
        sched_params = sched_cfg.get("params", {})

        scheduler_class = getattr(torch.optim.lr_scheduler, sched_name)
        scheduler = scheduler_class(optimizer, **sched_params)

        lr_scheduler_dict = {
            "scheduler": scheduler,
            "interval": sched_cfg.get("interval", "epoch"),
            "frequency": sched_cfg.get("frequency", 1),
        }

        if "monitor" in sched_cfg:
            lr_scheduler_dict["monitor"] = sched_cfg["monitor"]

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_dict,
        }
