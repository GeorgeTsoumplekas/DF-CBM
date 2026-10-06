from datasets.FF import FFWithConcepts
from utils.dataloader_utils import build_dataloader
import pytorch_lightning as pl


class FFDataModule(pl.LightningDataModule):
    def __init__(self, ds_cfg, batch_size=32):
        super().__init__()
        self.ds_cfg = ds_cfg
        self.batch_size = batch_size
        self.num_concepts = None
        self.concept_names = None
        self.dataset_pos_weights = None
        self.train_ds = None
        self.val_ds = None

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_ds = FFWithConcepts(train=True, **self.ds_cfg)
            self.val_ds = FFWithConcepts(train=False, **self.ds_cfg)
            self.num_concepts = len(self.train_ds.concept_name_to_id)
            from utils.concept_utils import ordered_concept_names

            self.concept_names = ordered_concept_names(self.train_ds.concept_name_to_id)
            self.dataset_pos_weights = self.train_ds.pos_weights.tolist()

    def train_dataloader(self):
        return build_dataloader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
        )

    def val_dataloader(self):
        return build_dataloader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
        )

    def get_split_dataloader(self, splits, shuffle=False):
        """Build a dataloader for arbitrary split names (e.g. train, test)."""
        dataset = FFWithConcepts(train=None, splits=splits, **self.ds_cfg)
        return build_dataloader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=4,
        ), dataset
