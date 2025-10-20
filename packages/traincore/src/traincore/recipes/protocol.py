from functools import partial
from typing import Protocol, Any, runtime_checkable
from torch import Tensor, optim, nn
from collections.abc import Mapping

from torch.optim.lr_scheduler import LRScheduler


@runtime_checkable
class AuemRecipeProtocol(Protocol):
    model: nn.Module | nn.ModuleDict
    loss: nn.Module | dict[str, nn.Module]
    optimizer: optim.Optimizer | partial | dict[str, optim.Optimizer | partial]
    scheduler: LRScheduler | partial | dict[str, LRScheduler | partial] | None = None

    def training_step(
        self, batch: dict[str, Any], batch_idx: int | None = None, *args, **kwargs
    ) -> Tensor | Mapping[str, Any] | None: ...

    def validation_step(
        self, batch: dict[str, Any], batch_idx: int | None = None, *args, **kwargs
    ) -> Tensor | Mapping[str, Any] | None: ...

    def test_step(
        self, batch: dict[str, Any], batch_idx: int | None = None, *args, **kwargs
    ) -> Tensor | Mapping[str, Any] | None: ...

    def configure_optimizers(self) -> Any: ...
