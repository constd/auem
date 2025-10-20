from collections.abc import Mapping
from functools import partial
from typing import Any

import torch
from lightning import LightningModule
from torch import Tensor, nn, optim

from traincore.config_stores.recipes import recipe_store
from torch.optim.lr_scheduler import LRScheduler


@recipe_store(name="simple")
class SimpleRecipe(LightningModule):
    def __init__(
        self,
        model: nn.Module | nn.ModuleDict,
        loss: nn.Module | dict[str, nn.Module],
        optimizer: optim.Optimizer | partial | dict[str, optim.Optimizer | partial],
        scheduler: LRScheduler
        | partial
        | dict[str, LRScheduler | partial]
        | None = None,
    ):
        super().__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler

    def training_step(
        self,
        batch: dict[str, Any],
        batch_idx: int | None = None,
        *args,
        **kwargs,
    ) -> Tensor | Mapping[str, Any] | None:
        total_loss = torch.tensor(0.0, device=self.device)
        for _, dataset in batch.items():
            x, y = dataset["audio"], dataset["class"]
            y_hat = self.model(x)
            loss = self.loss(y_hat, y.float())  # ty: ignore[possibly-missing-attribute]
            total_loss += loss
        return {"loss": total_loss}

    def validation_step(
        self,
        batch: dict[str, Any],
        batch_idx: int | None = None,
        *args,
        **kwargs,
    ) -> Tensor | Mapping[str, Any] | None:
        x, y = batch["audio"], batch["class"]
        y_hat = self.model(x)
        loss = self.loss(y_hat.float(), y.float())  # ty: ignore[possibly-missing-attribute]
        return {"loss": loss}

    def test_step(
        self,
        batch: dict[str, Any],
        batch_idx: int | None = None,
        *args,
        **kwargs,
    ) -> Tensor | Mapping[str, Any] | None:
        pass

    def configure_optimizers(self) -> Any:
        return self.optimizer(self.model.parameters())
