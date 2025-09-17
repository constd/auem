from typing import Iterator, Protocol

from torch import Tensor


class DatasetProtocol(Protocol):
    def setup(self, stage: str | None) -> None: ...
    def prepare_data(self) -> None: ...


class MapDatasetProtocol(DatasetProtocol):
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> dict[str, str, Tensor]: ...


class IterableDatasetProtocol(DatasetProtocol):
    def __iter__(self) -> Iterator[dict[str, str, Tensor]]: ...
