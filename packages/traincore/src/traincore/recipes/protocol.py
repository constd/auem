from typing import Protocol, Any, runtime_checkable
from torch import Tensor
from collections.abc import Mapping


@runtime_checkable
class AuemRecipeProtocol(Protocol):
    def training_step(
        self, batch: dict[str, Any], batch_idx: int | None = None, *args, **kwargs
    ) -> Tensor | Mapping[str, Any] | None: ...

    def validation_step(
        self, batch: dict[str, Any], batch_idx: int | None = None, *args, **kwargs
    ) -> Tensor | Mapping[str, Any] | None:
        pass

    def test_step(
        self, batch: dict[str, Any], batch_idx: int | None = None, *args, **kwargs
    ) -> Tensor | Mapping[str, Any] | None:
        pass

    def configure_optimizers(self) -> Any: ...
