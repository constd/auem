from typing import Protocol, runtime_checkable

from traincore.data.modules.generic import DatasetInputType


@runtime_checkable
class DataModuleProtocol(Protocol):
    datasets: DatasetInputType

    def prepare_data(self) -> None: ...
    def setup(self, stage: str | None) -> None: ...
