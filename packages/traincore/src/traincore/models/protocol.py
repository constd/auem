"""Base classes for Auem Models."""

from typing import Literal, Protocol

from torch import Tensor

from traincore.models.decoders.protocol import DecoderProtocol
from traincore.models.encoders.protocol import EncoderProtocol


class AuemModelProtocol(Protocol):
    encoder: EncoderProtocol | None
    decoder: DecoderProtocol | None
    # a2a: audio -> audio
    # a2e: audio -> embedding
    # e2a: embedding -> audio
    # e2e: embedding -> embedding
    mtype: Literal["a2a", "e2a", "a2e", "e2e"]

    def forward(self, *args, **kwargs) -> dict[str, Tensor | str]: ...
