from typing import TypedDict

from lightning.pytorch import LightningDataModule
from lightning.pytorch.utilities.types import (
    EVAL_DATALOADERS,
    TRAIN_DATALOADERS,
)
from torch.utils.data import DataLoader

from traincore.config_stores.datamodules import datamodule_store
from traincore.data.sets.protocol import DatasetProtocol


class BatchSizes(TypedDict):
    train: int
    validation: int
    test: int


class DatasetInputType(TypedDict):
    train: dict[str, DatasetProtocol] | None = None
    validation: dict[str, DatasetProtocol] | None = None
    test: dict[str, DatasetProtocol] | None = None
    batch_size: BatchSizes = BatchSizes(train=1, validation=1, test=1)


def seed_worker(worker_id):
    import random

    import numpy as np
    import torch

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


@datamodule_store(name="generic")
class GenericDataModule(LightningDataModule):
    def __init__(self, datasets: DatasetInputType, num_workers: int = 1) -> None:
        super().__init__()
        self.datasets = datasets
        self.num_workers = num_workers

    def prepare_data(self) -> None:
        # Download and tokenize data here
        if self.datasets.get("train", None):
            for train_dataset in self.datasets["train"].values():
                train_dataset.prepare_data()
        if self.datasets.get("validation", None):
            for validation_dataset in self.datasets["validation"].values():
                validation_dataset.prepare_data()
        if self.datasets.get("test", None):
            for test_dataset in self.datasets["test"].values():
                test_dataset.prepare_data()

    def setup(self, stage: str | None) -> None:
        # Load and split data here
        if stage == "fit":
            if self.datasets.get("train", None):
                for train_dataset in self.datasets["train"].values():
                    train_dataset.setup(stage)
            if self.datasets.get("validation", None):
                for validation_dataset in self.datasets["validation"].values():
                    validation_dataset.setup(stage)
        if stage == "test":
            if self.datasets.get("test", None):
                for test_dataset in self.datasets["test"].values():
                    test_dataset.setup(stage)
        if stage == "predict":
            if self.datasets.get("predict", None):
                for predict_dataset in self.datasets["predict"].values():
                    predict_dataset.setup(stage)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        # Return train dataloader here
        if self.datasets.get("train", None):
            return {
                name: DataLoader(
                    dataset,
                    batch_size=self.datasets.get("batch_sizes", {}).get("train", 1),
                    drop_last=True,
                    num_workers=self.num_workers,
                    timeout=600,
                    shuffle=True,
                    worker_init_fn=seed_worker,
                )
                for name, dataset in self.datasets["train"].items()
            }
        return None

    def val_dataloader(self) -> EVAL_DATALOADERS:
        # Return validation dataloader here
        if self.datasets.get("validation", None):
            return {
                name: DataLoader(
                    dataset,
                    batch_size=self.datasets.get("batch_sizes", {}).get(
                        "validation", 1
                    ),
                    drop_last=True,
                    num_workers=self.num_workers,
                    timeout=600,
                    shuffle=False,
                )
                for name, dataset in self.datasets["validation"].items()
            }
        return None

    def test_dataloader(self) -> EVAL_DATALOADERS:
        # Return test dataloader here
        if self.datasets.get("test", None):
            return {
                name: DataLoader(
                    dataset,
                    batch_size=self.datasets.get("batch_sizes", {}).get("test", 1),
                    drop_last=True,
                    num_workers=self.num_workers,
                    timeout=600,
                    shuffle=False,
                )
                for name, dataset in self.datasets["test"].items()
            }
        return None

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        # Return predict dataloader here
        pass
