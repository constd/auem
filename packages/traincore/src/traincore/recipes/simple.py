from collections.abc import Mapping
from typing import Any

from lightning import LightningModule
from torch import Tensor, nn, optim

from traincore.config_stores.recipes import recipe_store


@recipe_store(name="simple")
class SimpleRecipe(LightningModule):
    def __init__(self, model: nn.Module, loss: nn.Module, optimizer: optim.Optimizer):
        super().__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer

    def training_step(
        self,
        batch: dict[str, str | Tensor] | Tensor,
        batch_idx: int | None = None,
        *args,
        **kwargs,
    ) -> Tensor | Mapping[str, Any] | None:
        x, y = batch["audio"], batch["class"]
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        return {"loss": loss}

    def validation_step(
        self,
        batch: dict[str, str | Tensor] | Tensor,
        batch_idx: int | None = None,
        *args,
        **kwargs,
    ) -> Tensor | Mapping[str, Any] | None:
        x, y = batch["audio"], batch["class"]
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        return {"loss": loss}

    def test_step(
        self,
        batch: dict[str, str | Tensor] | Tensor,
        batch_idx: int | None = None,
        *args,
        **kwargs,
    ) -> Tensor | Mapping[str, Any] | None:
        pass

    def configure_optimizers(self) -> Any: ...
