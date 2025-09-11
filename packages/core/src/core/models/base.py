"""Base classes for Auem Models."""

from abc import abstractmethod
from typing import Optional

from torch import nn


class AuemClassifierBase(nn.Module):
    """A base class for all Auem classifier models."""

    def __init__(
        self,
        child_embedding_size: int,
        num_classes: int = 10,
        out_nonlinearity: Optional[str] = None,
    ):
        super(AuemClassifierBase, self).__init__()

        self.num_classes = 10
        self.class_layer = nn.Linear(child_embedding_size, num_classes)
        self.out_nonlinearity = out_nonlinearity

    @abstractmethod
    def get_embedding(self, x):
        """Abstract method to get the child output."""
        raise NotImplementedError(
            """Please implement the "embedding" part of the network"""
        )

    def forward(self, x):
        """Calculate the forward pass using hte child class' embedding."""
        out = self.get_embedding(x)
        out = self.class_layer(out)

        if not self.out_nonlinearity:
            return out
        elif self.out_nonlinearity == "softmax":
            return nn.functional.softmax(out, dim=-1)
        elif self.out_nonlinearity == "log_softmax":
            return nn.functional.log_softmax(out, dim=-1)
