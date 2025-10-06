from typing import TypedDict

from jaxtyping import Float
from torch import Tensor
from traincore.models.protocol import AuemModelProtocol

__all__ = ["DiscriminatorProtocol", "DiscriminatorReturnType", "MultiDiscriminatorProtocol", "MultiDiscriminatorReturnType"]


class DiscriminatorReturnType(TypedDict):
    x: Float[Tensor, "..."]
    fmap: list[Float[Tensor, "..."]]


class DiscriminatorProtocol(AuemModelProtocol):
    def forward(
        self, x: Float[Tensor, "..."], x_hat: Float[Tensor, "..."]
    ) -> DiscriminatorReturnType: ...


class MultiDiscriminatorReturnType(TypedDict):
    disc_rs: list[Float[Tensor, "..."]]
    fmap_rs: list[list[Float[Tensor, "..."]]]
    disc_gs: list[Float[Tensor, "..."]]
    fmap_gs: list[list[Float[Tensor, "..."]]]


class MultiDiscriminatorProtocol(AuemModelProtocol):
    def forward(
        self, x: Float[Tensor, "..."], x_hat: Float[Tensor, "..."]
    ) -> MultiDiscriminatorReturnType: ...
