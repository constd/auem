from typing import Protocol

from traincore.data.sets.protocol import DatasetProtocol


class DataModuleProtocol(Protocol):
    training_datasets: dict[str, DatasetProtocol] | None
    validation_datasets: dict[str, DatasetProtocol] | None
    test_datasets: dict[str, DatasetProtocol] | None

    def prepare_data(self) -> None: ...
    def setup(self, stage: str | None) -> None: ...
