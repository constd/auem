"""Base classes for Auem Models."""

from abc import abstractmethod

from torch import Tensor, nn


class AuemClassifierBase(nn.Module):
    """A base class for all Auem classifier models."""

    def __init__(
        self,
        child_embedding_size: int,
        num_classes: int = 10,
        out_nonlinearity: str | None = None,
    ) -> None:
        super().__init__()

        self.num_classes: int = 10
        self.class_layer: nn.Linear = nn.Linear(child_embedding_size, num_classes)
        self.out_nonlinearity: str | None = out_nonlinearity

    @abstractmethod
    def get_embedding(self, x: Tensor) -> Tensor:
        """Abstract method to get the child output."""
        raise NotImplementedError(
            """Please implement the "embedding" part of the network"""
        )

    def forward(self, x: Tensor) -> Tensor:
        """Calculate the forward passes using the child class' embedding."""
        out = self.get_embedding(x)
        out = self.class_layer(out)

        match self.out_nonlinearity:
            case "softmax":
                return nn.functional.softmax(out, dim=-1)
            case "log_softmax":
                return nn.functional.log_softmax(out, dim=-1)
            case _:
                return out
