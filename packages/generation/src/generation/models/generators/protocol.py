"""Base protocol for generators."""

from typing import Protocol, runtime_checkable
from torch import Tensor


@runtime_checkable
class AuemGeneratorProtocol(Protocol):
    """Protocol for audio generators."""

    def forward(self, *args, **kwargs) -> dict[str, Tensor | str]: ...
