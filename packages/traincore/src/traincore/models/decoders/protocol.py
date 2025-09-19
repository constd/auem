from typing import Protocol, runtime_checkable

from torch import Tensor


@runtime_checkable
class DecoderProtocol(Protocol):
    def forward(self, *args, **kwargs) -> Tensor: ...
