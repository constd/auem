from collections.abc import Mapping
from functools import partial
from typing import Any

import torch
from lightning import LightningModule
from torch import Tensor, nn, optim

from traincore.config_stores.recipes import recipe_store


@recipe_store(name="simple")
class SimpleRecipe(LightningModule):
    def __init__(
        self, model: nn.Module, loss: nn.Module, optimizer: optim.Optimizer | partial
    ):
        super().__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer

    def training_step(
        self,
        batch: dict[str, dict[str, str | Tensor] | Tensor],
        batch_idx: int | None = None,
        *args,
        **kwargs,
    ) -> Tensor | Mapping[str, Any] | None:
        total_loss = torch.tensor(0.0, device=self.device)
        for _, dataset in batch.items():
            x, y = dataset["audio"], dataset["class"]
            y_hat = self.model(x)
            loss = self.loss(y_hat, y.float())  # ty: ignore[possibly-unbound-attribute]
            total_loss += loss
        return {"loss": total_loss}

    def validation_step(
        self,
        batch: dict[str, str | Tensor] | Tensor,
        batch_idx: int | None = None,
        *args,
        **kwargs,
    ) -> Tensor | Mapping[str, Any] | None:
        x, y = batch["audio"], batch["class"]
        y_hat = self.model(x)
        loss = self.loss(y_hat.float(), y.float())  # ty: ignore[possibly-unbound-attribute]
        return {"loss": loss}

    def test_step(
        self,
        batch: dict[str, str | Tensor] | Tensor,
        batch_idx: int | None = None,
        *args,
        **kwargs,
    ) -> Tensor | Mapping[str, Any] | None:
        pass

    def configure_optimizers(self) -> Any:
        return self.optimizer(self.model.parameters())
