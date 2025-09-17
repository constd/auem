from typing import Protocol

from torch import Tensor


class DecoderProtocol(Protocol):
    def forward(self, *args, **kwargs) -> Tensor: ...
