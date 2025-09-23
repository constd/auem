"""A simple dense model."""

import numpy as np
from torch import Tensor, nn
from torch.nn import functional as F

from traincore.config_stores.models import model_store
from traincore.models.base import AuemClassifierBase
from traincore.models.decoders.protocol import DecoderProtocol
from traincore.models.encoders.protocol import EncoderProtocol

__all__ = ["SimpleNN"]


DEFAULT_DENSE_LAYER_DEF: tuple[int, ...] = (1024, 512, 256, 128)


@model_store(name="simplenn")
class SimpleNN(AuemClassifierBase):
    """A configurable Dense Network."""

    mtype: str = "e2e"

    def __init__(
        self,
        encoder: EncoderProtocol | None = None,
        decoder: DecoderProtocol | None = None,
        input_shape: tuple[int | None, ...] = (None, 1, 128, 2206),
        num_classes: int = 10,
        dense_layer_def: tuple[int, ...] = DEFAULT_DENSE_LAYER_DEF,
        out_nonlinearity: str = "softmax",
    ) -> None:
        super().__init__(
            dense_layer_def[-1], num_classes, out_nonlinearity=out_nonlinearity
        )
        self.encoder: EncoderProtocol | None = encoder
        self.decoder: EncoderProtocol | None = decoder

        self.dense_layers: nn.ModuleList = nn.ModuleList()

        input_size = int(np.prod(input_shape[1:]))
        previous_shape = input_size
        for layer_def in dense_layer_def:
            self.dense_layers.append(module=nn.Linear(previous_shape, layer_def))
            previous_shape = layer_def

    def get_embedding(self, x: Tensor) -> Tensor:
        """Calculate this model's outputs."""
        out = x.reshape(x.size(0), -1)
        for layer in self.dense_layers:
            out = F.leaky_relu(layer(out))

        return out
