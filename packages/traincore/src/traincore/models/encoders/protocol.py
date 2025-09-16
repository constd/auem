from typing import Protocol

from torch import Tensor


class EncoderProtocol(Protocol):
    def forward(self, *args, **kwargs) -> Tensor: ...
