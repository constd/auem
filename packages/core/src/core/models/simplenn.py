"""A simple dense model."""

import numpy as np
from torch import nn
from torch.nn import functional as F

from .base import AuemClassifierBase

__all__ = ["SimpleNN"]


DEFAULT_DENSE_LAYER_DEF = [1024, 512, 256, 128]


class SimpleNN(AuemClassifierBase):
    """A configurable Dense Network."""

    def __init__(
        self,
        input_shape,
        num_classes: int = 10,
        dense_layer_def=DEFAULT_DENSE_LAYER_DEF,
        out_nonlinearity="softmax",
    ):
        super(SimpleNN, self).__init__(
            dense_layer_def[-1], num_classes, out_nonlinearity=out_nonlinearity
        )

        self.dense_layers = []

        input_size = np.prod(input_shape[1:])
        previous_shape = input_size
        for layer_def in dense_layer_def:
            self.dense_layers.append(nn.Linear(previous_shape, layer_def))
            previous_shape = layer_def

    def get_embedding(self, x):
        """Calculate this model's outputs."""
        out = x.reshape(x.size(0), -1)
        for layer in self.dense_layers:
            out = F.leaky_relu(layer(out))

        return out
