from typing import Iterator, Protocol, runtime_checkable

from torch import Tensor


@runtime_checkable
class DatasetProtocol(Protocol):
    def setup(self, stage: str | None) -> None: ...
    def prepare_data(self) -> None: ...


@runtime_checkable
class MapDatasetProtocol(DatasetProtocol, Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> dict[str, str | Tensor]: ...


@runtime_checkable
class IterableDatasetProtocol(DatasetProtocol, Protocol):
    def __iter__(self) -> Iterator[dict[str, str | Tensor]]: ...
