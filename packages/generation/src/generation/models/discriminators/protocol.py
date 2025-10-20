from typing import TypedDict, runtime_checkable, Protocol

from jaxtyping import Float
from torch import Tensor

__all__ = [
    "DiscriminatorProtocol",
    "DiscriminatorReturnType",
    "MultiDiscriminatorProtocol",
    "MultiDiscriminatorReturnType",
    "CombinerDiscriminatorReturnType",
    "CombinerDiscriminatorProtocol",
]


class DiscriminatorReturnType(TypedDict):
    estimate: Float[Tensor, "..."]
    feature_map: list[Float[Tensor, "..."]]


@runtime_checkable
class DiscriminatorProtocol(Protocol):
    def forward(
        self, x: Float[Tensor, "batch channel time"]
    ) -> DiscriminatorReturnType: ...


class MultiDiscriminatorReturnType(TypedDict):
    estimates: list[Float[Tensor, "..."]]
    feature_maps: list[list[Float[Tensor, "..."]]]


@runtime_checkable
class MultiDiscriminatorProtocol(Protocol):
    def forward(self, x: Float[Tensor, "..."]) -> MultiDiscriminatorReturnType: ...


class CombinerDiscriminatorReturnType(TypedDict):
    estimates_real: list[Float[Tensor, "..."]]
    estimates_generated: list[Float[Tensor, "..."]]
    feature_maps_real: list[Float[Tensor, "..."]]
    feature_maps_generated: list[Float[Tensor, "..."]]


@runtime_checkable
class CombinerDiscriminatorProtocol(Protocol):
    def forward(self, x: Float[Tensor, "..."]) -> CombinerDiscriminatorReturnType: ...
