from typing import Any, Mapping
from torch import Tensor, nn
from lightning import LightningModule
from collections.abc import Mapping


class SimpleRecipe(LightningModule):
    def __init__(self, model: nn.Module, loss: nn.Module):
        super().__init__()
        self.model = model
        self.loss = loss

    def training_step(
        self, batch: dict[str, Any], batch_idx: int | None = None, *args, **kwargs
    ) -> Tensor | Mapping[str, Any] | None:
        x, y = batch["audio"], batch["class"]
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        return {"loss": loss}

    def validation_step(
        self, batch: dict[str, Any], batch_idx: int | None = None, *args, **kwargs
    ) -> Tensor | Mapping[str, Any] | None:
        x, y = batch["audio"], batch["class"]
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        return {"loss": loss}

    def test_step(
        self, batch: dict[str, Any], batch_idx: int | None = None, *args, **kwargs
    ) -> Tensor | Mapping[str, Any] | None:
        pass

    def configure_optimizers(self) -> Any: ...
